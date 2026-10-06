"""
Giai đoạn Perceive: hiểu cảnh và chẩn đoán lỗi theo vùng trước khi lập kế hoạch.
1. VLM (hoặc luật dự phòng) trả về DiagnosisReport: loại cảnh, vùng quan trọng, lỗi kèm
   mức độ, và các đặc điểm thẩm mỹ có chủ ý cần giữ (preserve).
2. Đo độ sáng thực tế trên từng vùng bằng mặt nạ mềm của Module 2.
3. Bổ sung lỗi đo được mà VLM bỏ sót (ví dụ khuôn mặt tối do ngược sáng).
"""

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from pydantic import ValidationError

from src.analyzer_evaluator.analyzer import (
    _BRIGHTNESS_HIGH,
    _BRIGHTNESS_LOW,
    _HIGHLIGHT_CLIP_RATIO_THRESH,
    _HIGHLIGHT_CLIP_THRESH,
    _SHADOW_CLIP_RATIO_THRESH,
    _SHADOW_CLIP_THRESH,
)

from .executor import resolve_region_mask
from .knowledge import diagnose_from_metrics
from .regions import canonical_region
from .state import (
    DEFECT_TYPES,
    PRESERVE_ASPECTS,
    SCENE_TYPES,
    Defect,
    DiagnosisReport,
    HistoryItem,
    PreserveItem,
    RegionMetrics,
)
from .vlm_diagnostician import _build_history_feedback, call_gemini_json, image_parts

logger = logging.getLogger(__name__)

MAX_SUBJECTS = 4
# Đo vùng trên ảnh thu nhỏ: thống kê độ sáng không cần độ phân giải gốc, và phát hiện
# khuôn mặt / phân vùng trên ảnh nhỏ rẻ hơn nhiều (ADR-004, CPU-only)
MEASURE_MAX_SIDE = 1024
# Vùng tối hơn phần còn lại ít nhất chừng này mức xám (và bản thân tối) → ngược sáng
BACKLIT_GAP = 40.0
BACKLIT_MAX_MEAN = 100.0
# Phần còn lại nhỏ hơn tỉ lệ này → không so sánh vùng với phần còn lại
MIN_REST_RATIO = 0.05
# Preserve làm cho vùng tối/sáng là chủ ý → không tự thêm lỗi phơi sáng đo được
_DARK_PRESERVES = frozenset(("low_key", "silhouette"))
_BRIGHT_PRESERVES = frozenset(("high_key",))
# Lỗi mâu thuẫn với một đặc điểm cần giữ (VLM đôi khi liệt kê cả hai)
_CONFLICTING_DEFECTS: Dict[str, frozenset[str]] = {
    "warm_tone": frozenset(("color_cast_warm",)),
    "cool_tone": frozenset(("color_cast_cool",)),
    "low_key": frozenset(("underexposed", "backlit_subject")),
    "silhouette": frozenset(("underexposed", "backlit_subject")),
    "high_key": frozenset(("overexposed",)),
    "film_grain": frozenset(("noise",)),
    "soft_focus": frozenset(("blur",)),
    "muted_colors": frozenset(("undersaturated",)),
    "vivid_colors": frozenset(("oversaturated",)),
}

PERCEIVE_PROMPT = """
Bạn là "AI Image Doctor" - GIAI ĐOẠN 1: QUAN SÁT & CHẨN ĐOÁN (chưa lập kế hoạch xử lý).
Nhìn ảnh, đọc chỉ số kỹ thuật toàn cục và trả về bản chẩn đoán JSON gồm:
1. scene_type: loại cảnh.
2. lighting: mô tả ngắn điều kiện ánh sáng (ví dụ: "ngược sáng", "nắng chiều", "đèn trong nhà").
3. subjects: tối đa 4 vùng quan trọng cần đo riêng, là danh từ ngắn bằng tiếng Anh
   ("face" cho khuôn mặt người, "sky", "person", "food", "dog"...). Không dùng "full".
4. defects: các lỗi kỹ thuật THẬT SỰ cần sửa. Mỗi lỗi gồm type, region ("full" hoặc một
   subject), severity (0 không đáng kể, 1 nhẹ, 2 rõ, 3 nặng) và evidence (bằng chứng ngắn).
   Dùng "backlit_subject" khi chủ thể tối vì nguồn sáng phía sau.
5. preserve: đặc điểm thẩm mỹ CÓ CHỦ Ý cần giữ nguyên (aspect, region, reason). Ví dụ:
   ánh hoàng hôn/đèn vàng ấm áp → warm_tone; ảnh tối tâm trạng → low_key;
   bóng đen nghệ thuật → silhouette; hạt film → film_grain; xóa phông mềm mại → soft_focus;
   tông màu nhạt kiểu film → muted_colors.
   Đặc điểm đã nằm trong preserve thì KHÔNG được liệt kê thành defect.
6. summary: tóm tắt chẩn đoán bằng tiếng Việt, 1–3 câu.
Lưu ý: chỉ số toàn cục là trung bình cả ảnh. Một vùng nhỏ (ví dụ khuôn mặt ngược sáng)
có thể tối dù brightness_level toàn cục là "normal" → hãy nhìn ảnh và liệt kê vùng đó.
Từ vòng 2 trở đi bạn nhận ẢNH GỐC và ẢNH HIỆN TẠI: chỉ chẩn đoán lỗi CÒN LẠI trên ẢNH HIỆN TẠI.
Viết lighting, evidence, reason và summary bằng tiếng Việt.
"""

PERCEIVE_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "scene_type": {"type": "string", "enum": list(SCENE_TYPES)},
        "lighting": {"type": "string"},
        "subjects": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_SUBJECTS},
        "defects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(DEFECT_TYPES)},
                    "region": {"type": "string"},
                    "severity": {"type": "integer", "minimum": 0, "maximum": 3},
                    "evidence": {"type": "string"},
                },
                "required": ["type", "region", "severity"],
            },
        },
        "preserve": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "aspect": {"type": "string", "enum": list(PRESERVE_ASPECTS)},
                    "region": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["aspect"],
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["scene_type", "defects", "preserve", "summary"],
}


# ---------------------------------------------------------
# Đo số liệu theo vùng
# ---------------------------------------------------------
def _luma(image: np.ndarray) -> np.ndarray:
    """Độ sáng (luma) float64 của ảnh RGB hoặc ảnh xám."""
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.ndim == 2:
        return image.astype(np.float64)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)


def _brightness_level(mean: float) -> str:
    """Phân loại phơi sáng theo cùng ngưỡng với Module 1."""
    if mean < _BRIGHTNESS_LOW:
        return "underexposed"
    if mean > _BRIGHTNESS_HIGH:
        return "overexposed"
    return "normal"


def compute_region_metrics(
    image: np.ndarray, mask: np.ndarray, region: str, backend: str = "unknown"
) -> Optional[RegionMetrics]:
    """
    Tính độ sáng có trọng số theo mặt nạ mềm (ADR-003) cho một vùng và so với phần còn lại.
    Trả về None nếu mặt nạ gần như rỗng.
    """
    weights = mask[:, :, 0] if mask.ndim == 3 else mask
    weights = weights.astype(np.float64)
    total = float(weights.sum())
    if total < 1.0:
        return None

    luma = _luma(image)
    mean = float((weights * luma).sum() / total)
    std = math.sqrt(float((weights * (luma - mean) ** 2).sum() / total))
    highlight = float((weights * (luma >= _HIGHLIGHT_CLIP_THRESH)).sum() / total)
    shadow = float((weights * (luma <= _SHADOW_CLIP_THRESH)).sum() / total)

    rest = 1.0 - weights
    rest_total = float(rest.sum())
    vs_rest: Optional[float] = None
    if rest_total / weights.size >= MIN_REST_RATIO:
        vs_rest = round(mean - float((rest * luma).sum() / rest_total), 2)

    return RegionMetrics(
        region=region,
        backend=backend,
        area_ratio=round(total / weights.size, 4),
        brightness_mean=round(mean, 2),
        brightness_std=round(std, 2),
        brightness_level=_brightness_level(mean),
        highlight_clip_ratio=round(highlight, 4),
        shadow_clip_ratio=round(shadow, 4),
        brightness_vs_rest=vs_rest,
    )


def _downscale(image: np.ndarray, max_side: int = MEASURE_MAX_SIDE) -> np.ndarray:
    """Thu nhỏ ảnh (giữ tỉ lệ) để cạnh dài không vượt max_side; ảnh nhỏ hơn giữ nguyên."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / longest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def measure_regions(image: np.ndarray, subjects: List[str]) -> Dict[str, RegionMetrics]:
    """
    Đo số liệu cho từng vùng qua Module 2, trên ảnh đã thu nhỏ về MEASURE_MAX_SIDE.
    Vùng không xác định được (không có khuôn mặt, backend ngữ nghĩa không khả dụng…)
    được bỏ qua, không làm hỏng giai đoạn Perceive.
    """
    image = _downscale(image)
    measured: Dict[str, RegionMetrics] = {}
    for subject in subjects[:MAX_SUBJECTS]:
        try:
            found, mask, backend = resolve_region_mask(image, subject)
            if not found or mask is None:
                continue
            region_metrics = compute_region_metrics(image, mask, subject, backend or "unknown")
        except Exception as exc:
            logger.warning("Could not measure region '%s': %s", subject, exc)
            continue
        if region_metrics is not None:
            measured[subject] = region_metrics
    return measured


def _preserved_aspects(report: DiagnosisReport, region: str) -> set[str]:
    """Các aspect cần giữ áp dụng cho một vùng (kể cả preserve trên toàn ảnh)."""
    return {
        item.aspect for item in report.preserve if canonical_region(item.region) in (region, "full")
    }


def _drop_preserved_defects(report: DiagnosisReport) -> List[Defect]:
    """Bỏ lỗi mâu thuẫn với preserve trên cùng vùng (hoặc preserve toàn ảnh): chủ ý thắng."""
    kept: List[Defect] = []
    for defect in report.defects:
        conflicts = {
            item.aspect
            for item in report.preserve
            if item.region in (defect.region, "full")
            and defect.type in _CONFLICTING_DEFECTS[item.aspect]
        }
        if conflicts:
            logger.info(
                "Dropping defect '%s' on '%s': conflicts with preserved %s.",
                defect.type,
                defect.region,
                sorted(conflicts),
            )
            continue
        kept.append(defect)
    return kept


def _measured_defects(report: DiagnosisReport) -> List[Defect]:
    """
    Suy ra lỗi phơi sáng theo vùng từ số đo thực tế mà chẩn đoán chưa nêu,
    trừ khi vùng đó tối/sáng là chủ ý (preserve).
    Chỉ khuôn mặt có mức sáng "đúng" tuyệt đối (da ở vùng trung tính). Vùng khác có thể tối
    hoặc sáng tự nhiên (trời, tuyết, mèo đen) nên cần bằng chứng: tối hơn hẳn phần còn lại
    (ngược sáng), hoặc bệt đen / cháy trắng thật theo ngưỡng clipping của Module 1.
    """
    existing = {(defect.region, defect.type) for defect in report.defects}
    added: List[Defect] = []
    for region, measured in report.region_metrics.items():
        preserved = _preserved_aspects(report, region)
        mean = measured.brightness_mean
        gap = measured.brightness_vs_rest
        is_face = region == "face"
        is_backlit = gap is not None and gap <= -BACKLIT_GAP and mean < BACKLIT_MAX_MEAN
        is_dark = (
            is_backlit
            or (is_face and measured.brightness_level == "underexposed")
            or measured.shadow_clip_ratio >= _SHADOW_CLIP_RATIO_THRESH
        )
        is_bright = (
            is_face and measured.brightness_level == "overexposed"
        ) or measured.highlight_clip_ratio >= _HIGHLIGHT_CLIP_RATIO_THRESH

        if is_dark and not (preserved & _DARK_PRESERVES):
            if (region, "underexposed") in existing or (region, "backlit_subject") in existing:
                continue
            defect_type = "backlit_subject" if is_backlit else "underexposed"
            gap_text = f", tối hơn phần còn lại {abs(gap):.0f} mức" if gap is not None else ""
            added.append(
                Defect(
                    type=defect_type,
                    region=region,
                    severity=3 if mean < 40 else 2,
                    evidence=f"Đo được: độ sáng vùng {mean:.0f}/255{gap_text}.",
                    origin="measured",
                )
            )
        elif is_bright and not (preserved & _BRIGHT_PRESERVES):
            if (region, "overexposed") in existing:
                continue
            added.append(
                Defect(
                    type="overexposed",
                    region=region,
                    severity=2,
                    evidence=(
                        f"Đo được: độ sáng vùng {mean:.0f}/255, "
                        f"cháy sáng {measured.highlight_clip_ratio:.0%}."
                    ),
                    origin="measured",
                )
            )
    return added


# ---------------------------------------------------------
# Chẩn đoán: VLM hoặc luật dự phòng
# ---------------------------------------------------------
def _normalize_region(raw: Any) -> str:
    """Chuẩn hóa tên vùng theo từ vựng chung (rỗng/toàn ảnh → 'full', khuôn mặt → 'face')."""
    return canonical_region(raw)


def _report_from_vlm_json(data: Any, iteration: int) -> DiagnosisReport:
    """
    Dựng DiagnosisReport từ JSON của VLM. Mỗi defect/preserve được dựng riêng: phần tử
    sai bị loại kèm cảnh báo. Thiếu danh sách defects, hoặc mọi defect đều sai → ValueError
    để chuyển sang chẩn đoán bằng luật.
    """
    if not isinstance(data, dict):
        raise ValueError("VLM diagnosis is not a JSON object")
    raw_defects = data.get("defects")
    if not isinstance(raw_defects, list):
        raise ValueError("VLM diagnosis 'defects' is not a list")

    defects: List[Defect] = []
    for index, raw in enumerate(raw_defects):
        try:
            if not isinstance(raw, dict):
                raise ValueError("not a JSON object")
            severity = int(round(float(raw.get("severity", 1))))
            defects.append(
                Defect(
                    type=raw.get("type"),
                    region=_normalize_region(raw.get("region")),
                    severity=max(0, min(3, severity)),
                    evidence=str(raw.get("evidence") or ""),
                    origin="vlm",
                )
            )
        except (ValidationError, ValueError, TypeError) as exc:
            logger.warning("Dropping invalid VLM defect %d: %s", index, exc)

    # Mọi lỗi đều sai → chẩn đoán rỗng sẽ bị hiểu là "ảnh tốt" (SHIP âm thầm): chuyển sang luật
    if raw_defects and not defects:
        raise ValueError(f"all {len(raw_defects)} VLM defects were invalid")

    preserve: List[PreserveItem] = []
    raw_preserve = data.get("preserve")
    for index, raw in enumerate(raw_preserve if isinstance(raw_preserve, list) else []):
        try:
            if not isinstance(raw, dict):
                raise ValueError("not a JSON object")
            preserve.append(
                PreserveItem(
                    aspect=raw.get("aspect"),
                    region=_normalize_region(raw.get("region")),
                    reason=str(raw.get("reason") or ""),
                )
            )
        except (ValidationError, ValueError) as exc:
            logger.warning("Dropping invalid VLM preserve item %d: %s", index, exc)

    # Vùng cần đo = subjects + mọi vùng có lỗi; bỏ trùng, bỏ 'full', giới hạn MAX_SUBJECTS
    raw_subjects = data.get("subjects")
    candidates = list(raw_subjects) if isinstance(raw_subjects, list) else []
    candidates += [defect.region for defect in defects]
    subjects: List[str] = []
    for candidate in candidates:
        region = _normalize_region(candidate)
        if region != "full" and region not in subjects:
            subjects.append(region)

    scene_type = data.get("scene_type")
    lighting = data.get("lighting")
    summary = data.get("summary")
    return DiagnosisReport(
        iteration=iteration,
        scene_type=scene_type if scene_type in SCENE_TYPES else "other",
        lighting=lighting if isinstance(lighting, str) else "",
        subjects=subjects[:MAX_SUBJECTS],
        defects=defects,
        preserve=preserve,
        summary=summary if isinstance(summary, str) else "",
        source="vlm",
    )


def perceive(
    image: np.ndarray,
    metrics: Dict[str, Any],
    iteration: int = 1,
    history: Optional[List[HistoryItem]] = None,
    original_image: Optional[np.ndarray] = None,
) -> DiagnosisReport:
    """
    Giai đoạn Perceive: chẩn đoán (VLM hoặc luật), đo số liệu theo vùng và bổ sung lỗi đo được.
    - Không có GEMINI_API_KEY → chẩn đoán rule-based (source="rule_based").
    - Lỗi gọi/parse Gemini → chẩn đoán rule-based (source="vlm_fallback").
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        report = diagnose_from_metrics(metrics, iteration, source="rule_based")
    else:
        try:
            prompt = (
                f"Chỉ số kỹ thuật toàn cục:\n{json.dumps(metrics, indent=2, default=str)}\n"
                f"Vòng lặp: {iteration}{_build_history_feedback(history)}"
            )
            contents = [prompt, *image_parts(image, iteration, original_image)]
            data = call_gemini_json(api_key, PERCEIVE_PROMPT, contents, PERCEIVE_RESPONSE_SCHEMA)
            report = _report_from_vlm_json(data, iteration)
        except Exception as exc:
            logger.warning("Gemini perception failed (%s); using the rule-based diagnosis.", exc)
            report = diagnose_from_metrics(metrics, iteration, source="vlm_fallback")

    report.defects = _drop_preserved_defects(report)
    report.region_metrics = measure_regions(image, report.subjects)
    report.defects = report.defects + _measured_defects(report)
    return report
