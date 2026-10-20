"""
Chỉnh kết quả theo góp ý của người dùng (Phase 4), ví dụ "da hơi vàng, trời gắt quá".

Góp ý được phân tích thành các điều chỉnh có cấu trúc (loại, vùng, mức độ): trước hết bằng
luật tiếng Việt (chạy offline), nếu luật không hiểu và có API key thì nhờ VLM. Điều chỉnh
được áp vào PHÁC ĐỒ (danh sách thao tác) rồi render lại từ ảnh gốc, không chồng lên ảnh đã
xử lý, nên góp ý nhiều lần không tích lũy sai số. Góp ý là yêu cầu trực tiếp của người
dùng nên không đi qua preserve guard.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, Field

from src.analyzer_evaluator.no_reference_eval import evaluate_no_reference

from .executor import execute_plan
from .planner import action_region, clamp_parameters, sanitize_actions
from .regions import canonical_region
from .state import RegionOperation, TreatmentPlan
from .vlm_diagnostician import call_gemini_json

logger = logging.getLogger(__name__)

ADJUSTMENT_KINDS: Tuple[str, ...] = (
    "brighter",
    "darker",
    "more_contrast",
    "less_contrast",
    "warmer",
    "cooler",
    "more_saturated",
    "less_saturated",
    "sharper",
    "softer",
    "less_noise",
)


class Adjustment(BaseModel):
    """Một điều chỉnh người dùng yêu cầu: loại, vùng và mức độ (1 nhẹ, 2 rõ)."""

    kind: Literal[ADJUSTMENT_KINDS]  # type: ignore[valid-type]
    region: str = "full"
    strength: int = Field(default=1, ge=1, le=2)


# ---------------------------------------------------------
# Phân tích góp ý bằng luật
# ---------------------------------------------------------
# (loại điều chỉnh, mẫu regex); mẫu cụ thể đặt trước mẫu chung để "sáng quá" không thành "sáng"
FEEDBACK_RULES: Tuple[Tuple[str, str], ...] = (
    # Trời/bầu trời "gắt", "chói" nghĩa là quá sáng, không phải tương phản
    ("darker", r"trời.*(gắt|chói|cháy)"),
    (
        "cooler",
        r"(vàng|ấm|cam) quá|(quá|hơi) (vàng|ấm|cam)|ám (vàng|cam)|bớt (vàng|ấm)|lạnh hơn",
    ),
    (
        "warmer",
        r"(xanh|lạnh) quá|(quá|hơi) (xanh|lạnh)|ám xanh|bớt (xanh|lạnh)|ấm hơn|ấm lên",
    ),
    (
        "less_saturated",
        r"(rực|chói màu|màu gắt|đậm màu|lòe loẹt) quá|quá (rực|đậm màu)|bớt (rực|màu)|màu gắt",
    ),
    ("more_saturated", r"(nhạt màu|màu nhạt|thiếu màu|xỉn)|rực hơn|tươi hơn|đậm màu hơn"),
    ("less_noise", r"nhiễu|sạn|lấm tấm|hạt quá|nhiều hạt|mịn hơn"),
    ("softer", r"nét quá|quá nét|gắt cạnh|viền sáng|halo|bớt nét"),
    ("sharper", r"\bmờ\b|nhòe|chưa nét|không nét|nét hơn|thiếu nét"),
    ("less_contrast", r"tương phản quá|quá tương phản|gắt|cứng quá|bớt tương phản"),
    ("more_contrast", r"nhạt nhòa|bệt|thiếu tương phản|tương phản hơn|phẳng quá|đậm hơn"),
    ("darker", r"(sáng|chói|cháy) quá|quá (sáng|chói)|bị cháy|tối hơn|bớt sáng|tối lại"),
    ("brighter", r"tối quá|quá tối|hơi tối|còn tối|sáng hơn|sáng lên|bớt tối|tối"),
)
STRONG_PATTERN = r"quá|rất|nhiều|hẳn|lắm"
WEAK_PATTERN = r"hơi|chút|một tí|nhẹ"
# Từ chỉ vùng → tên vùng (đi qua canonical_region; 'trời' → 'sky' khớp heuristic D1)
REGION_RULES: Tuple[Tuple[str, str], ...] = (
    (r"\bda\b|khuôn mặt|\bmặt\b|gương mặt", "face"),
    (r"bầu trời|\btrời\b", "sky"),
)
_CLAUSE_SPLIT = re.compile(r"[,;.\n]|\bnhưng\b|\bvà\b|\bcòn\b")


def _clause_region(clause: str) -> str:
    """Vùng được nhắc trong một mệnh đề (mặc định toàn ảnh)."""
    for pattern, region in REGION_RULES:
        if re.search(pattern, clause):
            return region
    return "full"


def _clause_strength(clause: str) -> int:
    """Mức độ: 'quá/rất/nhiều' → 2; 'hơi/chút' → 1; mặc định 1."""
    if re.search(WEAK_PATTERN, clause):
        return 1
    return 2 if re.search(STRONG_PATTERN, clause) else 1


def parse_feedback_rules(feedback: str) -> List[Adjustment]:
    """Tách góp ý thành mệnh đề; mỗi mệnh đề lấy điều chỉnh khớp đầu tiên theo FEEDBACK_RULES."""
    adjustments: List[Adjustment] = []
    for clause in _CLAUSE_SPLIT.split(feedback.casefold()):
        clause = clause.strip()
        if not clause:
            continue
        for kind, pattern in FEEDBACK_RULES:
            if re.search(pattern, clause):
                adjustment = Adjustment(
                    kind=kind, region=_clause_region(clause), strength=_clause_strength(clause)
                )
                same = next(
                    (a for a in adjustments if (a.kind, a.region) == (kind, adjustment.region)),
                    None,
                )
                if same is None:
                    adjustments.append(adjustment)
                else:
                    # Cùng loại và vùng nhắc nhiều lần → một điều chỉnh, giữ mức mạnh hơn
                    same.strength = max(same.strength, adjustment.strength)
                break
    return adjustments


FEEDBACK_PROMPT = """
Bạn chuyển góp ý của người dùng về một bức ảnh đã chỉnh thành các điều chỉnh có cấu trúc.
Mỗi điều chỉnh gồm kind, region ("full" hoặc tên vùng ngắn bằng tiếng Anh như "face", "sky")
và strength (1 nhẹ, 2 rõ). Chỉ dùng các kind cho phép. Nếu góp ý không yêu cầu thay đổi gì
về ảnh, trả về danh sách rỗng.
"""

FEEDBACK_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "adjustments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list(ADJUSTMENT_KINDS)},
                    "region": {"type": "string"},
                    "strength": {"type": "integer", "minimum": 1, "maximum": 2},
                },
                "required": ["kind"],
            },
        }
    },
    "required": ["adjustments"],
}


def parse_feedback(feedback: str) -> Tuple[List[Adjustment], str]:
    """
    Phân tích góp ý: luật trước; luật không hiểu và có API key → VLM.

    Returns:
        (adjustments, nguồn "rules" | "vlm" | "none").
    """
    adjustments = parse_feedback_rules(feedback)
    if adjustments:
        return adjustments, "rules"
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not feedback.strip():
        return [], "none"
    try:
        data = call_gemini_json(api_key, FEEDBACK_PROMPT, [feedback], FEEDBACK_SCHEMA)
        parsed: List[Adjustment] = []
        for raw in data.get("adjustments") or []:
            try:
                parsed.append(
                    Adjustment(
                        kind=raw.get("kind"),
                        region=canonical_region(raw.get("region")),
                        strength=min(2, max(1, int(raw.get("strength") or 1))),
                    )
                )
            except (ValueError, TypeError) as exc:
                logger.warning("Dropping invalid feedback adjustment %s: %s", raw, exc)
        return parsed, "vlm" if parsed else "none"
    except Exception as exc:
        logger.warning("Feedback parsing with Gemini failed (%s).", exc)
        return [], "none"


# ---------------------------------------------------------
# Áp điều chỉnh vào phác đồ
# ---------------------------------------------------------
# Thao tác thêm vào cho mỗi loại điều chỉnh: (operation, tham số theo mức độ 1/2)
ADDED_ACTIONS: Dict[str, Tuple[str, Dict[int, Dict[str, Any]]]] = {
    "brighter": ("gamma_correct", {1: {"gamma": 1.15}, 2: {"gamma": 1.3}}),
    "darker": ("gamma_correct", {1: {"gamma": 0.87}, 2: {"gamma": 0.77}}),
    "more_contrast": ("clahe", {1: {"clip_limit": 1.5}, 2: {"clip_limit": 2.0}}),
    "warmer": (
        "color_correct",
        {1: {"temperature_shift": 0.15, "saturation_scale": 1.0}, 2: {"temperature_shift": 0.3}},
    ),
    "cooler": (
        "color_correct",
        {1: {"temperature_shift": -0.15, "saturation_scale": 1.0}, 2: {"temperature_shift": -0.3}},
    ),
    "more_saturated": (
        "color_correct",
        {1: {"saturation_scale": 1.1}, 2: {"saturation_scale": 1.2}},
    ),
    "less_saturated": (
        "color_correct",
        {1: {"saturation_scale": 0.9}, 2: {"saturation_scale": 0.8}},
    ),
    "sharper": (
        "sharpen",
        {
            1: {"method": "unsharp_mask", "amount": 0.4},
            2: {"method": "unsharp_mask", "amount": 0.7},
        },
    ),
    "less_noise": (
        "denoise",
        {1: {"method": "bilateral", "strength": 1.0}, 2: {"method": "bilateral", "strength": 1.5}},
    ),
}
# Loại điều chỉnh "bớt" một thao tác đã có: (operation, tham số, hệ số theo mức độ 1/2)
REDUCING: Dict[str, Tuple[str, str, Dict[int, float]]] = {
    "less_contrast": ("clahe", "clip_limit", {1: 0.6, 2: 0.3}),
    "softer": ("sharpen", "amount", {1: 0.5, 2: 0.0}),
}

ADJUSTMENT_LABELS: Dict[str, str] = {
    "brighter": "sáng hơn",
    "darker": "tối hơn",
    "more_contrast": "tăng tương phản",
    "less_contrast": "giảm tương phản",
    "warmer": "ấm hơn",
    "cooler": "lạnh hơn",
    "more_saturated": "tăng bão hòa",
    "less_saturated": "giảm bão hòa",
    "sharper": "nét hơn",
    "softer": "bớt nét",
    "less_noise": "khử nhiễu",
}


def _new_action(adjustment: Adjustment, operation: str, params: Dict[str, Any]) -> RegionOperation:
    """Thao tác mới do góp ý thêm vào."""
    region = canonical_region(adjustment.region)
    return RegionOperation(
        region_id=region,
        target_prompt=region,
        region_type="full" if region == "full" else None,
        feather_radius=15 if region == "full" else 30,
        detected_issue=f"feedback:{adjustment.kind}",
        operation=operation,
        parameters=clamp_parameters(operation, dict(params)),
    )


def _reduce(
    actions: List[RegionOperation], adjustment: Adjustment
) -> Tuple[List[RegionOperation], bool]:
    """Bớt cường độ (hoặc bỏ) thao tác đã có trên vùng; False nếu không có gì để bớt."""
    operation, parameter, factors = REDUCING[adjustment.kind]
    factor = factors[adjustment.strength]
    region = canonical_region(adjustment.region)
    neutral = 1.0 if operation == "clahe" else 0.0
    changed = False
    result: List[RegionOperation] = []
    for action in actions:
        if action.operation == operation and region in ("full", action_region(action)):
            value = float(action.parameters.get(parameter, neutral))
            reduced = neutral + factor * (value - neutral)
            changed = True
            if abs(reduced - neutral) < 1e-6:
                continue  # bớt hết → bỏ thao tác
            params = clamp_parameters(operation, {**action.parameters, parameter: reduced})
            action = action.model_copy(update={"parameters": params})
        result.append(action)
    return result, changed


def apply_adjustments(
    actions: Sequence[RegionOperation], adjustments: Sequence[Adjustment]
) -> Tuple[List[RegionOperation], List[str]]:
    """
    Áp điều chỉnh vào phác đồ (bản sao). Khử nhiễu được chèn ĐẦU phác đồ (phải chạy trước
    các bước khuếch đại nhiễu); các thao tác khác nối vào cuối. Điều chỉnh "bớt" giảm cường độ
    thao tác đã có; không có gì để bớt thì ghi chú lại.

    Returns:
        (phác đồ mới, ghi chú tiếng Việt cho từng điều chỉnh).
    """
    result = [action.model_copy(deep=True) for action in actions]
    notes: List[str] = []
    for adjustment in adjustments:
        label = ADJUSTMENT_LABELS[adjustment.kind]
        where = "toàn ảnh" if adjustment.region == "full" else f"vùng '{adjustment.region}'"
        if adjustment.kind in REDUCING:
            result, changed = _reduce(result, adjustment)
            if changed:
                notes.append(f"{label} trên {where}")
            else:
                notes.append(f"không thể {label} trên {where}: phác đồ chưa có thao tác tương ứng")
            continue
        operation, by_strength = ADDED_ACTIONS[adjustment.kind]
        params = {**by_strength[1], **by_strength[adjustment.strength]}
        action = _new_action(adjustment, operation, params)
        if operation == "denoise":
            result.insert(0, action)
        else:
            result.append(action)
        notes.append(f"{label} trên {where}")
    return result, notes


@dataclass
class RefineResult:
    """Kết quả một lần chỉnh theo góp ý."""

    image: np.ndarray
    actions: List[RegionOperation]
    adjustments: List[Adjustment] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    source: str = "none"
    quality_score: Optional[float] = None


def refine(original: np.ndarray, actions: Sequence[RegionOperation], feedback: str) -> RefineResult:
    """
    Chỉnh phác đồ theo góp ý rồi render lại từ ảnh gốc (độ phân giải gốc). Không hiểu góp ý
    → giữ nguyên phác đồ, ảnh vẫn được render lại để trả về nhất quán.
    """
    adjustments, source = parse_feedback(feedback)
    new_actions, notes = apply_adjustments(sanitize_actions(list(actions)), adjustments)
    new_actions = sanitize_actions(new_actions)
    image = execute_plan(original, TreatmentPlan(reasoning=feedback, actions=new_actions))
    evaluation = evaluate_no_reference(image, previous_image=original, iteration=1)
    return RefineResult(
        image=image,
        actions=new_actions,
        adjustments=adjustments,
        notes=notes,
        source=source,
        quality_score=evaluation.estimated_quality_score,
    )
