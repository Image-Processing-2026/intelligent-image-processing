"""
Sinh nhiều phiên bản kết quả để người dùng chọn (Phase 3).

Vòng lặp khép kín tìm ra một "phác đồ" (các thao tác của những vòng không bị rollback).
Mỗi phong cách biến đổi phác đồ đó một cách tất định: đổi cường độ từng loại thao tác, bỏ
hoặc thêm thao tác. Mọi phong cách đều đi qua planner (kẹp tham số) và preserve guard, nên
không phong cách nào phá được đặc điểm cần giữ. Các phiên bản được render song song trên ảnh
preview, chấm điểm bằng Module 1, và nếu có API key thì VLM critic xếp hạng. Critic chỉ
đánh giá, không tạo pixel (ADR-001).
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.analyzer_evaluator.no_reference_eval import evaluate_no_reference

from .executor import execute_plan
from .planner import action_region, apply_preserve_guard, clamp_parameters
from .state import (
    DiagnosisReport,
    HistoryItem,
    PreserveItem,
    RegionOperation,
    TreatmentPlan,
    Variant,
)
from .vlm_diagnostician import call_gemini_json, to_vlm_image

logger = logging.getLogger(__name__)

MAX_VARIANTS = 3
# Điểm chênh không quá mức này so với điểm cao nhất được coi là ngang nhau → ưu tiên theo
# thứ tự phong cách (balanced trước): điểm heuristic không đủ tinh để phân định chênh lệch nhỏ
SCORE_TIE_TOLERANCE = 0.5
# Lỗi nhiễu từ mức này trở lên → phong cách đậm nét không tự thêm CLAHE (khuếch đại nhiễu)
NOISY_SEVERITY = 2


@dataclass(frozen=True)
class StyleProfile:
    """Một phong cách: hệ số cường độ theo loại thao tác, thao tác bị bỏ và thao tác thêm."""

    id: str
    label: str
    description: str
    # Hệ số nhân phần "lệch khỏi trung tính" của tham số, theo operation
    intensity: Dict[str, float] = field(default_factory=dict)
    drop_operations: Tuple[str, ...] = ()
    # Bão hòa toàn ảnh thêm vào khi phác đồ chưa chỉnh màu toàn ảnh (None → không thêm)
    extra_saturation: Optional[float] = None
    # clip_limit CLAHE toàn ảnh thêm vào khi phác đồ chưa có CLAHE toàn ảnh (None → không thêm)
    extra_clahe: Optional[float] = None


STYLES: Dict[str, StyleProfile] = {
    "balanced": StyleProfile(
        id="balanced",
        label="Cân bằng",
        description="Đúng phác đồ agent đã chọn qua các vòng chẩn đoán.",
    ),
    "natural": StyleProfile(
        id="natural",
        label="Tự nhiên",
        description="Chỉnh nhẹ tay, giữ cảm giác ảnh gốc; không làm nét.",
        intensity={
            "gamma_correct": 0.7,
            "clahe": 0.5,
            "denoise": 0.8,
            "color_correct": 0.6,
        },
        drop_operations=("sharpen",),
    ),
    "vivid": StyleProfile(
        id="vivid",
        label="Đậm nét",
        description="Tương phản và màu sắc mạnh hơn, nổi bật chủ thể.",
        intensity={
            "gamma_correct": 1.1,
            "clahe": 1.4,
            "sharpen": 1.2,
            "color_correct": 1.3,
        },
        extra_saturation=1.12,
        extra_clahe=1.5,
    ),
}
# Thứ tự ưu tiên khi điểm ngang nhau, và thứ tự sinh khi num_variants < số phong cách
STYLE_ORDER: Tuple[str, ...] = ("balanced", "natural", "vivid")


# ---------------------------------------------------------
# Phác đồ và phong cách
# ---------------------------------------------------------
def treatment_recipe(history: Sequence[HistoryItem]) -> List[RegionOperation]:
    """Các thao tác đã thực sự giữ lại qua vòng lặp (bỏ vòng bị rollback), theo thứ tự."""
    recipe: List[RegionOperation] = []
    for item in history:
        if item.rolled_back or not item.plan:
            continue
        recipe += [action.model_copy(deep=True) for action in item.plan.actions]
    return recipe


def merged_preserve(history: Sequence[HistoryItem]) -> List[PreserveItem]:
    """Hợp các đặc điểm cần giữ của mọi vòng chẩn đoán (không trùng aspect + vùng)."""
    seen: Dict[Tuple[str, str], PreserveItem] = {}
    for item in history:
        if item.diagnosis is None:
            continue
        for preserved in item.diagnosis.preserve:
            seen.setdefault((preserved.aspect, preserved.region), preserved)
    return list(seen.values())


def is_noisy(history: Sequence[HistoryItem]) -> bool:
    """Có vòng nào chẩn đoán nhiễu từ NOISY_SEVERITY trở lên không."""
    return any(
        defect.type == "noise" and defect.severity >= NOISY_SEVERITY
        for item in history
        if item.diagnosis is not None
        for defect in item.diagnosis.defects
    )


def _scaled_parameters(operation: str, parameters: Dict[str, Any], k: float) -> Dict[str, Any]:
    """Nhân phần lệch khỏi trung tính của tham số với hệ số k (tham số enum giữ nguyên)."""
    scaled = dict(parameters)
    if operation == "gamma_correct" and "gamma" in scaled:
        scaled["gamma"] = 1.0 + k * (float(scaled["gamma"]) - 1.0)
    elif operation == "clahe" and "clip_limit" in scaled:
        scaled["clip_limit"] = 1.0 + k * (float(scaled["clip_limit"]) - 1.0)
    elif operation == "denoise" and "strength" in scaled:
        scaled["strength"] = k * float(scaled["strength"])
    elif operation == "sharpen" and "amount" in scaled:
        scaled["amount"] = k * float(scaled["amount"])
    elif operation == "color_correct":
        if "saturation_scale" in scaled:
            scaled["saturation_scale"] = 1.0 + k * (float(scaled["saturation_scale"]) - 1.0)
        if "temperature_shift" in scaled:
            scaled["temperature_shift"] = k * float(scaled["temperature_shift"])
    return {
        name: round(value, 3) if isinstance(value, float) else value
        for name, value in clamp_parameters(operation, scaled).items()
    }


def _full_action(operation: str, parameters: Dict[str, Any], issue: str) -> RegionOperation:
    """Thao tác toàn ảnh do phong cách thêm vào."""
    return RegionOperation(
        region_id="full_image",
        target_prompt="full",
        region_type="full",
        detected_issue=issue,
        operation=operation,
        parameters=clamp_parameters(operation, parameters),
    )


def stylize(
    recipe: Sequence[RegionOperation],
    style: StyleProfile,
    preserve: Sequence[PreserveItem] = (),
    noisy: bool = False,
) -> List[RegionOperation]:
    """
    Áp một phong cách lên phác đồ: đổi cường độ, bỏ thao tác bị loại, thêm thao tác của
    phong cách nếu phác đồ chưa có, rồi chạy preserve guard. Trả về danh sách thao tác mới
    (phác đồ gốc không bị sửa).
    """
    actions: List[RegionOperation] = []
    for action in recipe:
        if action.operation in style.drop_operations:
            continue
        k = style.intensity.get(action.operation, 1.0)
        parameters = _scaled_parameters(action.operation, action.parameters, k)
        actions.append(action.model_copy(update={"parameters": parameters}, deep=True))

    full_ops = {a.operation for a in actions if action_region(a) == "full"}
    if style.extra_clahe is not None and not noisy and "clahe" not in full_ops:
        actions.append(_full_action("clahe", {"clip_limit": style.extra_clahe}, style.id))
    if style.extra_saturation is not None and "color_correct" not in full_ops:
        actions.append(
            _full_action(
                "color_correct",
                {"saturation_scale": style.extra_saturation, "temperature_shift": 0.0},
                style.id,
            )
        )

    plan = TreatmentPlan(reasoning=style.label, actions=actions)
    return apply_preserve_guard(plan, list(preserve)).actions


# ---------------------------------------------------------
# Render và xếp hạng
# ---------------------------------------------------------
def render_variant(
    original: np.ndarray,
    style_id: str,
    recipe: Sequence[RegionOperation],
    preserve: Sequence[PreserveItem] = (),
    noisy: bool = False,
) -> Variant:
    """Render một phong cách trên ảnh (preview) gốc và chấm điểm bằng Module 1."""
    style = STYLES[style_id]
    actions = stylize(recipe, style, preserve, noisy)
    image = execute_plan(original, TreatmentPlan(reasoning=style.label, actions=actions))
    evaluation = evaluate_no_reference(image, previous_image=original, iteration=1)
    return Variant(
        id=style.id,
        label=style.label,
        description=style.description,
        actions=actions,
        image=image,
        quality_score=evaluation.estimated_quality_score,
        quality_improved=evaluation.quality_improved,
    )


def _deduplicate(variants: Sequence[Variant]) -> List[Variant]:
    """Bỏ phiên bản cho ảnh y hệt một phiên bản đứng trước (theo STYLE_ORDER)."""
    ordered = sorted(variants, key=lambda v: STYLE_ORDER.index(v.id))
    unique: List[Variant] = []
    for variant in ordered:
        if any(np.array_equal(variant.image, kept.image) for kept in unique):
            logger.info("Variant '%s' is identical to an earlier one; dropping it.", variant.id)
            continue
        unique.append(variant)
    return unique


def _score_ranking(variants: Sequence[Variant]) -> List[str]:
    """
    Xếp hạng theo Module 1: phiên bản không cải thiện xếp sau; trong cùng nhóm, điểm cao
    hơn rõ (vượt SCORE_TIE_TOLERANCE) đứng trước, ngang nhau thì theo STYLE_ORDER.
    """
    remaining = sorted(variants, key=lambda v: STYLE_ORDER.index(v.id))
    ranking: List[str] = []
    for improved in (True, False):
        group = [v for v in remaining if (v.quality_improved is not False) == improved]
        while group:
            best = max(v.quality_score or 0.0 for v in group)
            pick = next(v for v in group if (v.quality_score or 0.0) >= best - SCORE_TIE_TOLERANCE)
            ranking.append(pick.id)
            group.remove(pick)
    return ranking


CRITIC_PROMPT = """
Bạn là "AI Image Doctor" - GIÁM KHẢO. Bạn nhận ẢNH GỐC và vài PHIÊN BẢN đã được xử lý bằng
thuật toán kinh điển (không có AI tạo ảnh). Hãy xếp hạng các phiên bản từ tốt nhất đến kém nhất:
1. Đã sửa được các lỗi trong chẩn đoán chưa (sáng, tương phản, nhiễu, màu…)?
2. Có tự nhiên không: không cháy sáng, không quầng, không bệt da, không nhiễu bị khuếch đại?
3. Có giữ được các đặc điểm cần giữ (preserve) và không khí của ảnh gốc không?
Trả về JSON: "ranking" (mọi id, tốt nhất trước), "recommended" (id nên chọn), và "notes"
(mỗi phiên bản một nhận xét ngắn bằng tiếng Việt có dấu, nêu ưu và nhược điểm thấy được trên ảnh).
"""


def _critic_schema(ids: Sequence[str]) -> Dict[str, Any]:
    """Schema JSON cho critic, giới hạn id trong các phiên bản đang xét."""
    id_schema = {"type": "string", "enum": list(ids)}
    return {
        "type": "object",
        "properties": {
            "ranking": {"type": "array", "items": id_schema},
            "recommended": id_schema,
            "notes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": id_schema, "note": {"type": "string"}},
                    "required": ["id", "note"],
                },
            },
        },
        "required": ["ranking", "recommended", "notes"],
    }


def _critic_contents(
    original: np.ndarray, variants: Sequence[Variant], diagnosis: Optional[DiagnosisReport]
) -> List[Any]:
    """Nội dung gửi critic: chẩn đoán, mô tả phiên bản, ảnh gốc và ảnh từng phiên bản."""
    lines = []
    if diagnosis is not None:
        lines.append(f"Chẩn đoán ảnh gốc: {diagnosis.summary}")
        if diagnosis.preserve:
            preserved = ", ".join(f"{p.aspect}@{p.region}" for p in diagnosis.preserve)
            lines.append(f"Cần giữ: {preserved}")
    for variant in variants:
        lines.append(
            f"- Phiên bản '{variant.id}' ({variant.label}): {variant.description} "
            f"Điểm Module 1: {variant.quality_score}"
        )
    contents: List[Any] = ["\n".join(lines), "ẢNH GỐC:", to_vlm_image(original)]
    for variant in variants:
        contents += [f"PHIÊN BẢN '{variant.id}':", to_vlm_image(variant.image)]
    return contents


def _critic_ranking(
    original: np.ndarray, variants: Sequence[Variant], diagnosis: Optional[DiagnosisReport]
) -> Optional[Tuple[List[str], str, Dict[str, str]]]:
    """(ranking, recommended, notes) từ VLM critic; None nếu không có key hoặc lỗi."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or len(variants) < 2:
        return None
    ids = [variant.id for variant in variants]
    try:
        data = call_gemini_json(
            api_key,
            CRITIC_PROMPT,
            _critic_contents(original, variants, diagnosis),
            _critic_schema(ids),
        )
        ranking: List[str] = []
        for variant_id in data.get("ranking") or []:
            if variant_id in ids and variant_id not in ranking:
                ranking.append(variant_id)
        if not ranking:
            raise ValueError("critic ranking has no known variant id")
        # Phiên bản critic bỏ sót được xếp sau theo điểm Module 1
        ranking += [v for v in _score_ranking(variants) if v not in ranking]
        recommended = data.get("recommended")
        if recommended not in ids:
            recommended = ranking[0]
        notes = {
            str(note.get("id")): str(note.get("note") or "")
            for note in data.get("notes") or []
            if isinstance(note, dict) and note.get("id") in ids
        }
        return ranking, recommended, notes
    except Exception as exc:
        logger.warning("Variant critic failed (%s); ranking by quality score.", exc)
        return None


def rank_variants(
    original: np.ndarray,
    variants: Sequence[Variant],
    diagnosis: Optional[DiagnosisReport] = None,
) -> Tuple[List[Variant], Optional[str], str]:
    """
    Bỏ phiên bản trùng ảnh, xếp hạng (VLM critic nếu có, ngược lại điểm Module 1).

    Returns:
        (variants theo thứ hạng, id phiên bản đề xuất, nguồn xếp hạng "critic" | "score").
    """
    unique = _deduplicate(variants)
    if not unique:
        return [], None, "score"
    critic = _critic_ranking(original, unique, diagnosis)
    if critic is not None:
        ranking, recommended, notes = critic
        source = "critic"
    else:
        ranking = _score_ranking(unique)
        recommended, notes, source = ranking[0], {}, "score"
    by_id = {variant.id: variant for variant in unique}
    ranked = [
        by_id[variant_id].model_copy(
            update={"rank": position, "critic_note": notes.get(variant_id, "")}
        )
        for position, variant_id in enumerate(ranking, start=1)
    ]
    return ranked, recommended, source


def styles_for(num_variants: int) -> List[str]:
    """Các phong cách cần sinh cho num_variants (tối đa MAX_VARIANTS, theo STYLE_ORDER)."""
    return list(STYLE_ORDER[: max(0, min(num_variants, MAX_VARIANTS))])
