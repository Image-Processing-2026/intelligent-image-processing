"""
Giai đoạn Perceive: hiểu cảnh và chẩn đoán lỗi theo vùng trước khi lập kế hoạch.
1. Đo cục bộ toàn ảnh bằng Module 1 (measure_image, độ phân giải gốc): tóm tắt + bản đồ tile.
2. VLM (hoặc luật dự phòng) trả về DiagnosisReport: loại cảnh, vùng quan trọng kèm bbox, lỗi
   kèm mức độ, và các đặc điểm thẩm mỹ có chủ ý cần giữ (preserve). VLM nhận tóm tắt số đo, crop
   100 % và overlay (evidence.py), prior EXIF; chạy PERCEIVE_SAMPLES lần (self-consistency) rồi
   gộp theo đa số.
3. Phân giải mask từng vùng bằng region_engine.resolve_subject_mask (dùng chung với executor,
   cache theo ảnh gốc), gộp bản đồ đo theo mask → RegionStats; giữ RegionMetrics cũ (độ sáng).
4. Bổ sung lỗi đo được mà VLM bỏ sót (ví dụ khuôn mặt tối do ngược sáng).
"""

import json
import logging
import math
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from src.analyzer_evaluator.measurement import (
    Estimate,
    MeasurementResult,
    RegionStats,
    aggregate,
    measure_image,
)
from src.analyzer_evaluator.metadata import ImageMetadata, metadata_priors
from src.region_engine import SubjectMask, resolve_subject_mask, subject_capabilities

from .evidence import (
    EvidenceCrop,
    EvidenceOverlay,
    build_overlays,
    describe_crops,
    select_evidence,
    summarize_measurement,
)
from .executor import image_cache_key, subject_cache_key
from .imaging import PREVIEW_MAX_SIDE, downscale
from .knowledge import diagnose_from_metrics
from .regions import canonical_region
from .state import (
    DEFECT_TYPES,
    PRESERVE_ASPECTS,
    SCENE_TYPES,
    Defect,
    DiagnosisReport,
    Evidence,
    HistoryItem,
    PreserveItem,
    RegionMetrics,
)
from .vlm_diagnostician import (
    _build_history_feedback,
    call_gemini_json,
    image_parts,
    to_vlm_image,
)

logger = logging.getLogger(__name__)

MAX_SUBJECTS = 4
# RegionMetrics cũ (độ sáng) tính trên ảnh thu nhỏ: thống kê độ sáng không cần độ phân giải gốc
MEASURE_MAX_SIDE = PREVIEW_MAX_SIDE
# Vùng tối hơn phần còn lại ít nhất chừng này mức xám (và bản thân tối) → ngược sáng
BACKLIT_GAP = 40.0
BACKLIT_MAX_MEAN = 100.0
# Phần còn lại nhỏ hơn tỉ lệ này → không so sánh vùng với phần còn lại
MIN_REST_RATIO = 0.05
# Self-consistency (plan D3): số lần quan sát mở, temperature, tỉ lệ đồng thuận tối thiểu
PERCEIVE_SAMPLES = 3
PERCEIVE_TEMPERATURE = 0.7
CONSENSUS_FRACTION = 2.0 / 3.0
# Ảnh toàn khung gửi VLM: None = độ phân giải gốc. Câu hỏi mở plan §7 (đo ở P5); ghi đè bằng
# biến môi trường PERCEIVE_FULL_FRAME_MAX_SIDE để benchmark so sánh.
FULL_FRAME_MAX_SIDE: Optional[int] = None
FULL_FRAME_ENV = "PERCEIVE_FULL_FRAME_MAX_SIDE"
# Tile của bản đồ đo cục bộ (C1)
MEASURE_TILE_SIZE = 64
# Skin locus trong YCrCb (Chai & Ngan 1999): da người thường nằm trong khoảng này
SKIN_CR_RANGE = (133.0, 173.0)
SKIN_CB_RANGE = (77.0, 127.0)
# Pixel da dùng để kiểm tra: bỏ pixel quá tối hoặc cháy (màu không tin được)
SKIN_MIN_LUMA = 30.0
SKIN_MAX_CHANNEL = 250
# Đủ chừng này pixel (theo trọng số mask) thì độ tin cậy kiểm tra skin locus đạt tối đa
SKIN_FULL_CONFIDENCE_PIXELS = 2000.0
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
Bạn nhận: ẢNH TOÀN KHUNG; các CROP 100 % độ phân giải có nhãn (A, B, …) do hệ thống đo chọn
(patch phẳng để soi nhiễu, cạnh mạnh để soi độ nét, khuôn mặt, vùng cháy sáng); OVERLAY tô màu
vùng clip và heatmap nhiễu; TÓM TẮT SỐ ĐO classical (giá trị kèm độ tin cậy); có thể có GỢI Ý
TỪ EXIF. Số đo và EXIF là bằng chứng hỗ trợ, ẢNH là căn cứ chính: số đo có thể sai trên nội
dung đặc biệt (cỏ, hoàng hôn, tường màu), EXIF chỉ là xác suất.
Trả về bản chẩn đoán JSON gồm:
1. scene_type: loại cảnh.
2. lighting: mô tả ngắn điều kiện ánh sáng (ví dụ: "ngược sáng", "nắng chiều", "đèn trong nhà").
3. subjects: tối đa 4 vùng quan trọng THẬT SỰ CÓ trong ảnh, mỗi vùng gồm name (danh từ ngắn
   tiếng Anh: "face" cho khuôn mặt người, "sky", "person", "food", "dog"…; không dùng "full")
   và box_2d = [ymin, xmin, ymax, xmax] chuẩn hóa 0–1000 bao trọn vùng đó.
   Không liệt kê vùng không có (ví dụ ảnh trong nhà không có "sky").
4. defects: GIẢ THUYẾT lỗi kỹ thuật THẬT SỰ cần sửa. Mỗi lỗi gồm type, region ("full" hoặc name
   của một subject), severity (0 không đáng kể, 1 nhẹ, 2 rõ, 3 nặng), evidence (bằng chứng ngắn,
   nêu số đo nếu dùng) và crops (nhãn crop/overlay bạn đã nhìn để kết luận, ví dụ ["A", "C"]).
   Dùng "backlit_subject" khi chủ thể tối vì nguồn sáng phía sau.
   Chỉ báo lỗi khi THẤY rõ trên ảnh hoặc crop. Ảnh tốt thì trả "defects": [] — "không có lỗi"
   là câu trả lời đúng và được khuyến khích khi ảnh ổn; đừng cố tìm lỗi.
   Mờ do xóa phông/chuyển động có chủ ý, hạt film, tông màu nghệ thuật KHÔNG phải lỗi.
5. preserve: đặc điểm thẩm mỹ CÓ CHỦ Ý cần giữ nguyên (aspect, region, reason). Ví dụ:
   ánh hoàng hôn/đèn vàng ấm áp → warm_tone; ảnh tối tâm trạng → low_key;
   bóng đen nghệ thuật → silhouette; hạt film → film_grain; xóa phông mềm mại → soft_focus;
   tông màu nhạt kiểu film → muted_colors.
   Đặc điểm đã nằm trong preserve thì KHÔNG được liệt kê thành defect.
6. summary: tóm tắt chẩn đoán bằng tiếng Việt, 1–3 câu.
Lưu ý: số đo toàn ảnh là trung bình cả ảnh. Một vùng nhỏ (ví dụ khuôn mặt ngược sáng)
có thể tối dù trung bình bình thường → hãy nhìn ảnh và liệt kê vùng đó.
Từ vòng 2 trở đi bạn nhận ẢNH GỐC và ẢNH HIỆN TẠI: chỉ chẩn đoán lỗi CÒN LẠI trên ẢNH HIỆN TẠI
(crop và số đo lấy từ ẢNH HIỆN TẠI).
Viết lighting, evidence, reason và summary bằng tiếng Việt có dấu.
"""

PERCEIVE_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "scene_type": {"type": "string", "enum": list(SCENE_TYPES)},
        "lighting": {"type": "string"},
        "subjects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "box_2d": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "required": ["name"],
            },
            "maxItems": MAX_SUBJECTS,
        },
        "defects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(DEFECT_TYPES)},
                    "region": {"type": "string"},
                    "severity": {"type": "integer", "minimum": 0, "maximum": 3},
                    "evidence": {"type": "string"},
                    "crops": {"type": "array", "items": {"type": "string"}},
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


@dataclass
class RegionMeasurement:
    """Kết quả đo theo vùng: RegionStats (v2), RegionMetrics cũ và mask đã phân giải."""

    stats: Dict[str, RegionStats] = field(default_factory=dict)
    metrics: Dict[str, RegionMetrics] = field(default_factory=dict)
    masks: Dict[str, SubjectMask] = field(default_factory=dict)


def _mask2d(mask: np.ndarray) -> np.ndarray:
    return mask[:, :, 0] if mask.ndim == 3 else mask


def _usable(subject: SubjectMask) -> bool:
    """Mask định vị được vùng: không rỗng và không phải phương án cuối 'full' (D1)."""
    return subject.source != "full" and subject.confidence > 0.0 and bool(subject.mask.max() > 0)


def resolve_subject_masks(
    image: np.ndarray,
    subjects: Sequence[str],
    boxes: Optional[Dict[str, List[int]]] = None,
    cache_key: Optional[str] = None,
) -> Dict[str, SubjectMask]:
    """
    Mask của từng subject qua resolver chung (C2), kèm bbox Perceive nếu có. Vùng không định vị
    được (không thấy mặt, rơi về 'full') hoặc lỗi được bỏ qua, không làm hỏng Perceive.
    """
    boxes = boxes or {}
    masks: Dict[str, SubjectMask] = {}
    for subject in subjects[:MAX_SUBJECTS]:
        box = boxes.get(subject)
        try:
            resolved = resolve_subject_mask(
                image, subject, tuple(box) if box else None, cache_key=cache_key
            )
        except Exception as exc:
            logger.warning("Could not resolve region '%s': %s", subject, exc)
            continue
        if _usable(resolved):
            masks[subject] = resolved
        else:
            logger.info("Region '%s' not located (source %s).", subject, resolved.source)
    return masks


def skin_locus_stats(image: np.ndarray, mask: np.ndarray, confidence: float) -> Dict[str, Estimate]:
    """
    Kiểm tra skin locus trên vùng mặt (plan §1.4): tỉ lệ pixel da (theo trọng số mask) nằm trong
    vùng Cr/Cb của da người, và Cr, Cb trung bình. Ảnh xám hoặc quá ít pixel tin được → {}.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        return {}
    rgb = image.astype(np.uint8)
    if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2]):
        return {}
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float64)
    luma, cr, cb = ycrcb[..., 0], ycrcb[..., 1], ycrcb[..., 2]
    trusted = (luma >= SKIN_MIN_LUMA) & (rgb.max(axis=2) < SKIN_MAX_CHANNEL)
    weights = _mask2d(mask).astype(np.float64) * trusted
    total = float(weights.sum())
    if total < 1.0:
        return {}
    inside = (
        (cr >= SKIN_CR_RANGE[0])
        & (cr <= SKIN_CR_RANGE[1])
        & (cb >= SKIN_CB_RANGE[0])
        & (cb <= SKIN_CB_RANGE[1])
    )
    estimate_confidence = float(np.clip(confidence * total / SKIN_FULL_CONFIDENCE_PIXELS, 0, 1))
    samples = int(round(total))
    return {
        "color.skin_locus_fraction": Estimate(
            value=float((weights * inside).sum() / total),
            confidence=min(1.0, estimate_confidence),
            n_samples=samples,
        ),
        "color.skin_cr_mean": Estimate(
            value=float((weights * cr).sum() / total),
            confidence=min(1.0, estimate_confidence),
            n_samples=samples,
        ),
        "color.skin_cb_mean": Estimate(
            value=float((weights * cb).sum() / total),
            confidence=min(1.0, estimate_confidence),
            n_samples=samples,
        ),
    }


def measure_regions(
    image: np.ndarray,
    subjects: Sequence[str],
    boxes: Optional[Dict[str, List[int]]] = None,
    *,
    measurement: Optional[MeasurementResult] = None,
    cache_key: Optional[str] = None,
) -> RegionMeasurement:
    """
    Đo theo vùng: mask từ resolver chung (resolve_subject_mask, bbox Perceive nếu có, cache theo
    cache_key) → RegionStats = aggregate(bản đồ đo, mask) cho mỗi subject, luôn kèm 'full' và
    'background' (phần bù của hợp các subject) khi có measurement. RegionMetrics cũ (độ sáng)
    tính trên ảnh thu nhỏ về MEASURE_MAX_SIDE cho từng subject (UI/API cũ vẫn đọc).
    Khuôn mặt có thêm kiểm tra skin locus trong RegionStats.values.
    """
    result = RegionMeasurement(masks=resolve_subject_masks(image, subjects, boxes, cache_key))
    small = downscale(image, MEASURE_MAX_SIDE)
    small_size = (small.shape[1], small.shape[0])

    for name, subject in result.masks.items():
        mask = subject.mask
        try:
            if measurement is not None:
                stats = aggregate(measurement, mask, name).model_copy(
                    update={"mask_source": subject.source, "mask_confidence": subject.confidence}
                )
                if name == "face":
                    stats.values.update(skin_locus_stats(image, mask, subject.confidence))
                result.stats[name] = stats
            small_mask = (
                mask
                if small.shape[:2] == mask.shape[:2]
                else cv2.resize(_mask2d(mask), small_size, interpolation=cv2.INTER_AREA)
            )
            metrics = compute_region_metrics(small, small_mask, name, subject.source)
        except Exception as exc:
            logger.warning("Could not measure region '%s': %s", name, exc)
            continue
        if metrics is not None:
            result.metrics[name] = metrics

    if measurement is not None:
        shape = (measurement.report.height, measurement.report.width)
        full = np.ones(shape, dtype=np.float32)
        result.stats["full"] = aggregate(measurement, full, "full").model_copy(
            update={"mask_source": "full", "mask_confidence": 1.0}
        )
        if "background" not in result.stats:
            union = np.zeros(shape, dtype=np.float32)
            for subject in result.masks.values():
                union = np.maximum(union, _mask2d(subject.mask).astype(np.float32))
            confidence = min((s.confidence for s in result.masks.values()), default=1.0)
            result.stats["background"] = aggregate(
                measurement, 1.0 - union, "background"
            ).model_copy(
                update={
                    "mask_source": "complement" if result.masks else "full",
                    "mask_confidence": confidence,
                }
            )
    return result


def _measure(image: np.ndarray) -> Optional[MeasurementResult]:
    """Đo cục bộ toàn ảnh (Module 1, C1); lỗi đo không làm hỏng Perceive."""
    try:
        return measure_image(image, tile_size=MEASURE_TILE_SIZE)
    except Exception as exc:
        logger.warning("Local measurement failed: %s", exc)
        return None


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
                    evidence_items=[
                        Evidence(source="measured", metric="brightness_mean", value=mean),
                        *(
                            [Evidence(source="measured", metric="brightness_vs_rest", value=gap)]
                            if gap is not None
                            else []
                        ),
                    ],
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
                    evidence_items=[
                        Evidence(source="measured", metric="brightness_mean", value=mean),
                        Evidence(
                            source="measured",
                            metric="highlight_clip_ratio",
                            value=measured.highlight_clip_ratio,
                        ),
                    ],
                )
            )
    return added


# ---------------------------------------------------------
# Chẩn đoán: VLM hoặc luật dự phòng
# ---------------------------------------------------------
def _normalize_region(raw: Any) -> str:
    """Chuẩn hóa tên vùng theo từ vựng chung (rỗng/toàn ảnh → 'full', khuôn mặt → 'face')."""
    return canonical_region(raw)


def _parse_box(raw: Any) -> Optional[List[int]]:
    """bbox Gemini [ymin, xmin, ymax, xmax] 0..1000 hợp lệ (có diện tích) hoặc None."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = (int(round(float(value))) for value in raw)
    except (TypeError, ValueError):
        return None
    if not all(0 <= value <= 1000 for value in (ymin, xmin, ymax, xmax)):
        return None
    if ymin >= ymax or xmin >= xmax:
        return None
    return [ymin, xmin, ymax, xmax]


def _parse_subjects(raw_subjects: Any) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    subjects của VLM: [{name, box_2d}] (C6) hoặc danh sách tên (định dạng cũ).
    Trả về (tên đã chuẩn hóa theo thứ tự, bbox theo tên); bỏ 'full' và tên rỗng.
    """
    names: List[str] = []
    boxes: Dict[str, List[int]] = {}
    for raw in raw_subjects if isinstance(raw_subjects, list) else []:
        raw_name = raw.get("name") if isinstance(raw, dict) else raw
        name = _normalize_region(raw_name)
        if name == "full" or not isinstance(raw_name, str):
            continue
        if name not in names:
            names.append(name)
        box = _parse_box(raw.get("box_2d")) if isinstance(raw, dict) else None
        if box is not None and name not in boxes:
            boxes[name] = box
    return names, boxes


def _crop_labels(raw: Any) -> List[str]:
    """Nhãn crop VLM dẫn chứng (chuỗi ngắn, viết hoa)."""
    if not isinstance(raw, list):
        return []
    return [label.strip().upper() for label in raw if isinstance(label, str) and label.strip()]


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
            evidence = str(raw.get("evidence") or "")
            crops = _crop_labels(raw.get("crops"))
            items = [Evidence(source="vlm", text=evidence)] if evidence else []
            if crops:
                items.append(Evidence(source="vlm", metric="crops", text=", ".join(crops)))
            defects.append(
                Defect(
                    type=raw.get("type"),
                    region=_normalize_region(raw.get("region")),
                    severity=max(0, min(3, severity)),
                    evidence=evidence,
                    origin="vlm",
                    evidence_items=items,
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
    subjects, boxes = _parse_subjects(data.get("subjects"))
    for defect in defects:
        if defect.region != "full" and defect.region not in subjects:
            subjects.append(defect.region)
    subjects = subjects[:MAX_SUBJECTS]

    scene_type = data.get("scene_type")
    lighting = data.get("lighting")
    summary = data.get("summary")
    return DiagnosisReport(
        iteration=iteration,
        scene_type=scene_type if scene_type in SCENE_TYPES else "other",
        lighting=lighting if isinstance(lighting, str) else "",
        subjects=subjects,
        defects=defects,
        preserve=preserve,
        summary=summary if isinstance(summary, str) else "",
        subject_boxes={name: box for name, box in boxes.items() if name in subjects},
        source="vlm",
    )


# ---------------------------------------------------------
# Self-consistency: gộp N lần quan sát
# ---------------------------------------------------------
def _consensus_count(samples: int) -> int:
    """Số lần tối thiểu một giả thuyết phải xuất hiện (≥ 2/3 số lần quan sát hợp lệ)."""
    return max(1, math.ceil(CONSENSUS_FRACTION * samples - 1e-9))


def _median_box(boxes: List[List[int]]) -> List[int]:
    """bbox trung vị theo từng tọa độ."""
    return [int(round(float(np.median([box[i] for box in boxes])))) for i in range(4)]


def merge_samples(reports: Sequence[DiagnosisReport], iteration: int = 1) -> DiagnosisReport:
    """
    Gộp các lần quan sát (self-consistency, plan D3): defect (type, region) xuất hiện ở
    ≥ 2/3 số lần thì giữ, severity = trung vị, confidence = tỉ lệ đồng thuận; còn lại vào
    `rejected`. scene_type, preserve và subjects theo đa số; lighting/summary lấy từ lần quan
    sát khớp kết luận nhất. Một lần quan sát → trả lại đúng lần đó (confidence 1).
    """
    if not reports:
        raise ValueError("no diagnosis samples to merge")
    total = len(reports)
    needed = _consensus_count(total)

    defect_votes: Dict[Tuple[str, str], List[Defect]] = {}
    for report in reports:
        seen = set()
        for defect in report.defects:
            key = (defect.type, defect.region)
            if key not in seen:
                seen.add(key)
                defect_votes.setdefault(key, []).append(defect)

    kept: List[Defect] = []
    rejected: List[Defect] = []
    for (defect_type, region), votes in defect_votes.items():
        agreement = len(votes) / total
        severity = int(math.floor(float(np.median([d.severity for d in votes])) + 0.5))
        crops = sorted(
            {
                label
                for d in votes
                for item in d.evidence_items
                if item.metric == "crops"
                for label in item.text.split(", ")
                if label
            }
        )
        items = [
            Evidence(
                source="vlm",
                metric="self_consistency",
                value=round(agreement, 3),
                text=f"{len(votes)}/{total} lần quan sát",
            )
        ]
        items += [item for item in votes[0].evidence_items if item.metric != "crops"]
        if crops:
            items.append(Evidence(source="vlm", metric="crops", text=", ".join(crops)))
        merged = votes[0].model_copy(
            update={
                "severity": max(0, min(3, severity)),
                "confidence": round(agreement, 3),
                "evidence_items": items,
            }
        )
        (kept if len(votes) >= needed else rejected).append(merged)

    preserve_votes: Dict[Tuple[str, str], List[PreserveItem]] = {}
    for report in reports:
        for item in {(p.aspect, p.region): p for p in report.preserve}.values():
            preserve_votes.setdefault((item.aspect, item.region), []).append(item)
    preserve = [votes[0] for votes in preserve_votes.values() if len(votes) >= needed]

    scenes = Counter(report.scene_type for report in reports)
    top = max(scenes.values())
    scene_type = next(report.scene_type for report in reports if scenes[report.scene_type] == top)

    subject_votes: Counter = Counter()
    box_votes: Dict[str, List[List[int]]] = {}
    order: List[str] = []
    for report in reports:
        for name in report.subjects:
            subject_votes[name] += 1
            if name not in order:
                order.append(name)
        for name, box in report.subject_boxes.items():
            box_votes.setdefault(name, []).append(box)
    required = {d.region for d in kept if d.region != "full"}
    subjects = [n for n in order if subject_votes[n] >= needed or n in required][:MAX_SUBJECTS]
    boxes = {name: _median_box(box_votes[name]) for name in subjects if name in box_votes}

    kept_keys = {(d.type, d.region) for d in kept}

    def overlap(report: DiagnosisReport) -> int:
        """Số giả thuyết của lần quan sát trùng kết luận chung, trừ số giả thuyết lệch."""
        keys = {(d.type, d.region) for d in report.defects}
        return len(keys & kept_keys) - len(keys - kept_keys)

    representative = max(reports, key=overlap)
    return DiagnosisReport(
        iteration=iteration,
        scene_type=scene_type,
        lighting=representative.lighting,
        subjects=subjects,
        defects=kept,
        preserve=preserve,
        summary=representative.summary,
        subject_boxes=boxes,
        rejected=rejected,
        source="vlm",
    )


# ---------------------------------------------------------
# Nội dung gửi VLM
# ---------------------------------------------------------
def _full_frame_max_side() -> Optional[int]:
    """Cạnh dài tối đa của ảnh toàn khung gửi VLM (None = gốc), cho phép ghi đè bằng env."""
    raw = os.getenv(FULL_FRAME_ENV, "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            logger.warning("Ignoring invalid %s=%r.", FULL_FRAME_ENV, raw)
        else:
            return value if value > 0 else None
    return FULL_FRAME_MAX_SIDE


def _priors_text(metadata: Optional[ImageMetadata]) -> str:
    """Khối gợi ý từ EXIF (metadata_priors, C3); rỗng nếu không có prior."""
    if metadata is None:
        return ""
    priors = metadata_priors(metadata)
    if not priors:
        return ""
    lines = ["GỢI Ý TỪ EXIF (chỉ là khả năng, ảnh và crop mới là căn cứ):"]
    lines += [f"- {prior.kind} (độ mạnh {prior.strength:.2f}): {prior.reason}" for prior in priors]
    return "\n".join(lines)


def _face_stats_text(stats: Optional[RegionStats]) -> str:
    """Một dòng số đo vùng mặt (nếu đã định vị được trước khi gọi VLM)."""
    if stats is None or not stats.values:
        return ""
    parts = []
    for key, label, digits in (
        ("exposure.mean_l", "L*", 0),
        ("exposure.clip_high", "tỉ lệ cháy", 2),
        ("noise.sigma_luma", "σ nhiễu", 1),
        ("color.skin_locus_fraction", "tỉ lệ pixel trong skin locus", 2),
    ):
        estimate = stats.values.get(key)
        if estimate is not None:
            parts.append(f"{label} {estimate.value:.{digits}f} (tin cậy {estimate.confidence:.2f})")
    return f"- Khuôn mặt (MediaPipe): {'; '.join(parts)}." if parts else ""


def build_perceive_contents(
    image: np.ndarray,
    metrics: Dict[str, Any],
    iteration: int,
    history: Optional[List[HistoryItem]],
    original_image: Optional[np.ndarray],
    measurement: Optional[MeasurementResult],
    crops: Sequence[EvidenceCrop],
    overlays: Sequence[EvidenceOverlay],
    metadata: Optional[ImageMetadata] = None,
    face_stats: Optional[RegionStats] = None,
) -> List[Any]:
    """
    Nội dung một lần gọi Perceive v2: tóm tắt bằng chứng có cấu trúc (thay dump JSON chỉ số),
    mô tả crop, prior EXIF, lịch sử; rồi ảnh toàn khung (vòng ≥ 2 kèm ảnh gốc), từng crop có
    nhãn và overlay. Không có số đo cục bộ → dùng lại chỉ số toàn cục dạng JSON.
    """
    blocks: List[str] = []
    if measurement is not None:
        blocks.append(summarize_measurement(measurement.report))
        face_line = _face_stats_text(face_stats)
        if face_line:
            blocks.append(face_line)
    else:
        blocks.append(f"Chỉ số kỹ thuật toàn cục:\n{json.dumps(metrics, indent=2, default=str)}")
    crop_text = describe_crops(crops)
    if crop_text:
        blocks.append(crop_text)
    if overlays:
        blocks.append(
            "OVERLAY (ảnh thu nhỏ, chỉ để tham khảo vị trí):\n"
            + "\n".join(f"- Overlay {overlay.label}: {overlay.description}" for overlay in overlays)
        )
    priors = _priors_text(metadata)
    if priors:
        blocks.append(priors)
    blocks.append(f"Vòng lặp: {iteration}{_build_history_feedback(history)}")

    max_side = _full_frame_max_side()
    frame = downscale(image, max_side) if max_side else image
    original = (
        downscale(original_image, max_side)
        if max_side and original_image is not None
        else original_image
    )
    contents: List[Any] = ["\n\n".join(blocks), "ẢNH TOÀN KHUNG:"]
    contents += image_parts(frame, iteration, original)
    for crop in crops:
        contents += [f"Crop {crop.label}: {crop.description}", to_vlm_image(crop.image)]
    for overlay in overlays:
        contents += [f"Overlay {overlay.label}: {overlay.description}", to_vlm_image(overlay.image)]
    return contents


def _sample_diagnoses(api_key: str, contents: List[Any], iteration: int) -> List[DiagnosisReport]:
    """
    Gọi Perceive PERCEIVE_SAMPLES lần song song (temperature PERCEIVE_TEMPERATURE khi N > 1).
    Lần lỗi (mạng, parse, mọi defect sai) bị bỏ; tất cả lỗi → raise lỗi cuối để fallback luật.
    """
    samples = max(1, int(PERCEIVE_SAMPLES))
    temperature = PERCEIVE_TEMPERATURE if samples > 1 else None

    def one(_: int) -> DiagnosisReport:
        data = call_gemini_json(
            api_key, PERCEIVE_PROMPT, contents, PERCEIVE_RESPONSE_SCHEMA, temperature=temperature
        )
        return _report_from_vlm_json(data, iteration)

    reports: List[DiagnosisReport] = []
    errors: List[BaseException] = []
    if samples == 1:
        return [one(0)]
    with ThreadPoolExecutor(max_workers=samples, thread_name_prefix="perceive") as pool:
        futures = [pool.submit(one, index) for index in range(samples)]
        for future in futures:
            try:
                reports.append(future.result())
            except Exception as exc:
                logger.warning("Perceive sample failed: %s", exc)
                errors.append(exc)
    if not reports:
        raise errors[-1]
    if errors:
        logger.info("Merging %d of %d perceive samples.", len(reports), samples)
    return reports


def _evidence_masks(
    image: np.ndarray, history: Optional[List[HistoryItem]], cache_key: str
) -> Dict[str, SubjectMask]:
    """
    Mask đã biết trước khi gọi VLM để chọn crop: khuôn mặt (MediaPipe) và, từ vòng 2, các subject
    của chẩn đoán vòng trước (trúng cache resolver).
    """
    previous = history[-1].diagnosis if history and history[-1].diagnosis else None
    subjects = ["face"] + [s for s in (previous.subjects if previous else []) if s != "face"]
    boxes = previous.subject_boxes if previous else {}
    return resolve_subject_masks(image, subjects, boxes, cache_key)


def _vlm_diagnosis(
    api_key: str,
    image: np.ndarray,
    metrics: Dict[str, Any],
    iteration: int,
    history: Optional[List[HistoryItem]],
    original_image: Optional[np.ndarray],
    measurement: Optional[MeasurementResult],
    metadata: Optional[ImageMetadata],
    cache_key: str,
) -> DiagnosisReport:
    """Chẩn đoán VLM v2: bằng chứng (crop, overlay, số đo, EXIF) + self-consistency."""
    crops: List[EvidenceCrop] = []
    overlays: List[EvidenceOverlay] = []
    face_stats: Optional[RegionStats] = None
    if measurement is not None:
        try:
            masks = _evidence_masks(image, history, cache_key)
            region_masks = {name: subject.mask for name, subject in masks.items()}
            crops = select_evidence(image, measurement, region_masks)
            overlays = build_overlays(image, measurement)
            if "face" in masks:
                face = masks["face"]
                face_stats = aggregate(measurement, face.mask, "face")
                face_stats.values.update(skin_locus_stats(image, face.mask, face.confidence))
        except Exception as exc:
            logger.warning("Evidence selection failed (%s); sending the full frame only.", exc)
            crops, overlays = [], []
    contents = build_perceive_contents(
        image,
        metrics,
        iteration,
        history,
        original_image,
        measurement,
        crops,
        overlays,
        metadata,
        face_stats,
    )
    return merge_samples(_sample_diagnoses(api_key, contents, iteration), iteration)


def perceive(
    image: np.ndarray,
    metrics: Dict[str, Any],
    iteration: int = 1,
    history: Optional[List[HistoryItem]] = None,
    original_image: Optional[np.ndarray] = None,
    metadata: Optional[ImageMetadata] = None,
    cache_key: Optional[str] = None,
) -> DiagnosisReport:
    """
    Giai đoạn Perceive: đo cục bộ, chẩn đoán (VLM v2 hoặc luật), đo theo vùng và bổ sung lỗi
    đo được.
    - Không có GEMINI_API_KEY → chẩn đoán rule-based (source="rule_based").
    - Lỗi gọi/parse Gemini ở mọi lần quan sát → chẩn đoán rule-based (source="vlm_fallback").
    metadata: EXIF của file gốc; prior được đưa vào prompt như gợi ý.
    cache_key: khóa cache mask (image_cache_key của ảnh gốc của lượt chạy); None → tự tính từ
    original_image (hoặc image) để vòng 2–3 dùng lại mask của vòng 1.
    """
    base_key = cache_key or image_cache_key(original_image if original_image is not None else image)
    mask_key = subject_cache_key(base_key, image.shape)
    measurement = _measure(image)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        report = diagnose_from_metrics(metrics, iteration, source="rule_based")
    else:
        try:
            report = _vlm_diagnosis(
                api_key,
                image,
                metrics,
                iteration,
                history,
                original_image,
                measurement,
                metadata,
                mask_key,
            )
        except Exception as exc:
            logger.warning("Gemini perception failed (%s); using the rule-based diagnosis.", exc)
            report = diagnose_from_metrics(metrics, iteration, source="vlm_fallback")

    report.defects = _drop_preserved_defects(report)
    regions = measure_regions(
        image,
        report.subjects,
        report.subject_boxes,
        measurement=measurement,
        cache_key=mask_key,
    )
    report.region_metrics = regions.metrics
    report.region_stats = regions.stats
    if measurement is not None:
        report.measurement = measurement.report
        report.measurement_maps = measurement.maps
    report.defects = report.defects + _measured_defects(report)
    return report


# ---------------------------------------------------------
# Nạp sẵn model phân vùng
# ---------------------------------------------------------
def preload_region_models(background: bool = True) -> Optional[threading.Thread]:
    """
    Nạp sẵn DINO + MobileSAM khi có model (subject_capabilities()["semantic_models"]) để lời
    gọi Perceive đầu tiên không chờ ~13 s. Chạy một lần phân giải trên ảnh nhỏ; lỗi chỉ ghi log.
    Trả về thread nạp (background=True) hoặc None.
    """
    if not subject_capabilities()["semantic_models"]:
        logger.info("Semantic segmentation models not installed; skipping preload.")
        return None

    def load() -> None:
        try:
            resolve_subject_mask(np.zeros((64, 64, 3), dtype=np.uint8), "object", use_semantic=True)
            logger.info("Semantic segmentation models preloaded.")
        except Exception as exc:
            logger.warning("Semantic model preload failed: %s", exc)

    if not background:
        load()
        return None
    thread = threading.Thread(target=load, name="region-model-preload", daemon=True)
    thread.start()
    return thread
