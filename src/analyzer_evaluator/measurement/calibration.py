"""Hiệu chỉnh severity từ số đo (Analysis v2 P2, hợp đồng C4 trong docs/analysis_v2/contracts.md).

Mỗi loại lỗi có một hàm điểm (càng lớn càng nặng) tính từ `MeasurementReport` hoặc `RegionStats`,
và ba ngưỡng trong `calibration.yaml`: severity = số ngưỡng mà điểm đạt tới (≥). Ngưỡng được khớp
trên bộ synthetic bằng `scripts/calibrate_severity.py`; hàm điểm ở đây là nguồn duy nhất, script
khớp dùng lại đúng các hàm này. `TechnicalMetrics.*_level` của `analyze_image` cũng suy ra từ đây
(quyết định D4: một nguồn ngưỡng).

Confidence = độ tin cậy của số đo × khoảng cách tới ngưỡng gần nhất (0.5–1) × độ tin cậy của
bảng hiệu chỉnh (`reliability` từ kiểm chéo bỏ-một-nguồn, quy về 0.5–1); ngưỡng prior (không khớp)
nhân thêm PRIOR_CONFIDENCE_SCALE.

Lỗi không đo được (thiếu số đo, loại lỗi không có hàm điểm, vùng không có key cần thiết) trả
`(0, 0.0)`. Không dùng generative AI (ADR-001).
"""

import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import yaml

from .schemas import Estimate, MeasurementReport, RegionStats

logger = logging.getLogger("img_doctor.analyzer_evaluator")

CALIBRATION_PATH = Path(__file__).with_name("calibration.yaml")
Stats = Union[RegionStats, MeasurementReport]
# (điểm, độ tin cậy của số đo) hoặc None khi không đo được
Score = Optional[Tuple[float, float]]
ScoreFn = Callable[[Stats, Dict[str, Any], Optional[int]], Score]

# Mức xám giữa trong L* (18 % phản xạ) làm gốc cho điểm phơi sáng
MID_GRAY_L = 50.0
# Độ rộng vùng mềm của confidence quanh ngưỡng, tính theo phần của khoảng ngưỡng
CONFIDENCE_SOFT_FRACTION = 0.25
PRIOR_CONFIDENCE_SCALE = 0.5
NOT_MEASURED: Tuple[int, float] = (0, 0.0)


def _estimate(stats: Stats, key: str) -> Optional[Estimate]:
    """Ước lượng theo key dạng "nhóm.tên" ("noise.sigma_luma").

    `MeasurementReport` tra theo nhóm; `RegionStats` chỉ có các key của MAP_KEYS.
    """
    if isinstance(stats, RegionStats):
        return stats.values.get(key)
    section, name = key.split(".", 1)
    return getattr(stats, section, {}).get(name)


def _value(stats: Stats, key: str) -> Optional[float]:
    estimate = _estimate(stats, key)
    return None if estimate is None else estimate.value


def _resolution_scale(stats: Stats, params: Dict[str, Any], long_side: Optional[int]) -> float:
    """Quy độ mờ (px) về cạnh dài tham chiếu của bộ hiệu chỉnh; không phóng to ảnh nhỏ.

    `MeasurementReport` tự biết kích thước; `RegionStats` cần `long_side` từ người gọi.
    """
    if long_side is None and isinstance(stats, MeasurementReport):
        long_side = max(stats.width, stats.height)
    reference = float(params.get("reference_long_side", 0.0))
    if not long_side or reference <= 0.0:
        return 1.0
    return min(1.0, reference / float(long_side))


def _score_noise(stats: Stats, params: Dict[str, Any], long_side: Optional[int]) -> Score:
    """σ nhiễu luma (mức 0–255) có trọng số cảm nhận.

    Nhiễu tập trung ở vùng tối (σ tối / σ sáng = r > 1) ít thấy hơn: chia cho
    1 + dark_weight·(r − 1). Nhiễu chroma quy về thang luma bằng chroma_weight; lấy giá trị lớn hơn.
    Vùng (RegionStats) chỉ có σ luma.
    """
    luma = _estimate(stats, "noise.sigma_luma")
    if luma is None:
        return None
    sigma = luma.value
    ratio = _value(stats, "noise.sigma_dark_ratio")
    if ratio is not None and ratio > 1.0:
        sigma /= 1.0 + float(params.get("dark_weight", 0.0)) * (ratio - 1.0)
    confidence = luma.confidence
    chroma = [
        e for e in (_estimate(stats, "noise.sigma_cb"), _estimate(stats, "noise.sigma_cr")) if e
    ]
    if chroma:
        strongest = max(chroma, key=lambda e: e.value)
        weighted = float(params.get("chroma_weight", 0.0)) * strongest.value
        if weighted > sigma:
            sigma, confidence = weighted, strongest.confidence
    return sigma, confidence


def _score_defocus(stats: Stats, params: Dict[str, Any], long_side: Optional[int]) -> Score:
    """σ defocus đã bù lệch thấp có hệ thống của phép đo, quy về cạnh dài tham chiếu (px)."""
    defocus = _estimate(stats, "sharpness.defocus_sigma")
    if defocus is None:
        return None
    unbiased = defocus.value / (1.0 - float(params.get("defocus_bias", 0.0)))
    return unbiased * _resolution_scale(stats, params, long_side), defocus.confidence


def _score_motion(stats: Stats, params: Dict[str, Any], long_side: Optional[int]) -> Score:
    """Độ dài mờ chuyển động (σ tương đương, px) khi hướng mờ đã được xác nhận, ngược lại None.

    Xác nhận: dị hướng σ defocus theo hướng pháp tuyến cạnh ≥ min_anisotropy và độ dài
    ≥ min_extent (ảnh nhiễu cần thêm noisy_extra_extent). Coherence toàn cục không được dùng.
    """
    anisotropy = _estimate(stats, "sharpness.motion_anisotropy")
    extent = _estimate(stats, "sharpness.motion_extent_sigma")
    if anisotropy is None or extent is None:
        return None
    min_extent = float(params.get("min_extent", math.inf))
    sigma = _value(stats, "noise.sigma_luma")
    if sigma is not None and sigma >= float(params.get("noisy_sigma", math.inf)):
        min_extent += float(params.get("noisy_extra_extent", 0.0))
    if (
        anisotropy.value < float(params.get("min_anisotropy", math.inf))
        or extent.value < min_extent
    ):
        return None
    scale = _resolution_scale(stats, params, long_side)
    return extent.value * scale, min(anisotropy.confidence, extent.confidence)


def _score_darkness(stats: Stats, params: Dict[str, Any], long_side: Optional[int]) -> Score:
    """Độ tối: MID_GRAY_L − L* trung bình."""
    mean = _estimate(stats, "exposure.mean_l")
    return None if mean is None else (MID_GRAY_L - mean.value, mean.confidence)


def _score_brightness(stats: Stats, params: Dict[str, Any], long_side: Optional[int]) -> Score:
    """Độ sáng: L* trung bình − MID_GRAY_L."""
    mean = _estimate(stats, "exposure.mean_l")
    return None if mean is None else (mean.value - MID_GRAY_L, mean.confidence)


def _score_flatness(stats: Stats, params: Dict[str, Any], long_side: Optional[int]) -> Score:
    """Độ phẳng tông: 100 − (p95 − p5) của L*. Vùng không có phân vị nên không đo được."""
    low = _estimate(stats, "exposure.p5_l")
    high = _estimate(stats, "exposure.p95_l")
    if low is None or high is None:
        return None
    return 100.0 - (high.value - low.value), min(low.confidence, high.confidence)


def _illuminant(stats: Stats) -> Optional[Tuple[float, float, float, float]]:
    """(r, g, b, confidence) của nguồn sáng ước lượng; None khi không có pixel trung tính."""
    parts = [_estimate(stats, f"color.illuminant_{c}") for c in "rgb"]
    if any(p is None or p.value <= 0.0 for p in parts):
        return None
    r, g, b = (p.value for p in parts)  # type: ignore[union-attr]
    return r, g, b, min(p.confidence for p in parts)  # type: ignore[union-attr]


def _score_cast_warm(stats: Stats, params: Dict[str, Any], long_side: Optional[int]) -> Score:
    """Trục ấm–lạnh của nguồn sáng (RGB tuyến tính): ln(r / b)."""
    light = _illuminant(stats)
    return None if light is None else (math.log(light[0] / light[2]), light[3])


def _score_cast_cool(stats: Stats, params: Dict[str, Any], long_side: Optional[int]) -> Score:
    """Trục lạnh–ấm của nguồn sáng: ln(b / r)."""
    light = _illuminant(stats)
    return None if light is None else (math.log(light[2] / light[0]), light[3])


def _score_cast_green(stats: Stats, params: Dict[str, Any], long_side: Optional[int]) -> Score:
    """Trục xanh lá–tím của nguồn sáng: ln(g) − (ln r + ln b) / 2."""
    light = _illuminant(stats)
    if light is None:
        return None
    r, g, b, confidence = light
    return math.log(g) - 0.5 * (math.log(r) + math.log(b)), confidence


def _score_chroma(stats: Stats, params: Dict[str, Any], long_side: Optional[int]) -> Score:
    """C* trung vị (CIELAB); vùng dùng C* trung bình của bản đồ."""
    if isinstance(stats, RegionStats):
        estimate = _estimate(stats, "color.chroma")
    else:
        estimate = _estimate(stats, "color.chroma_p50")
    return None if estimate is None else (estimate.value, estimate.confidence)


def _score_low_chroma(stats: Stats, params: Dict[str, Any], long_side: Optional[int]) -> Score:
    """Độ nhạt màu: −C* (càng nhạt điểm càng lớn)."""
    chroma = _score_chroma(stats, params, long_side)
    return None if chroma is None else (-chroma[0], chroma[1])


def _score_dark_channel(stats: Stats, params: Dict[str, Any], long_side: Optional[int]) -> Score:
    """Median dark channel ngoài vùng trời (0–1): sương mù nâng dark channel."""
    estimate = _estimate(stats, "haze.dark_channel_median")
    return None if estimate is None else (estimate.value, estimate.confidence)


def _score_blockiness(stats: Stats, params: Dict[str, Any], long_side: Optional[int]) -> Score:
    """Tỉ số chênh lệch ở biên khối 8×8 so với trong khối (1 = không có khối)."""
    estimate = _estimate(stats, "compression.blockiness")
    return None if estimate is None else (estimate.value, estimate.confidence)


def _score_clip_blob(stats: Stats, params: Dict[str, Any], long_side: Optional[int]) -> Score:
    """Diện tích (tỉ lệ) blob cháy sáng lớn nhất; vùng dùng tỉ lệ pixel cháy trung bình."""
    if isinstance(stats, RegionStats):
        estimate = _estimate(stats, "exposure.clip_high")
    else:
        estimate = _estimate(stats, "exposure.largest_clip_blob")
    return None if estimate is None else (estimate.value, estimate.confidence)


SCORE_FUNCTIONS: Dict[str, ScoreFn] = {
    "noise_sigma": _score_noise,
    "defocus": _score_defocus,
    "motion_extent": _score_motion,
    "darkness": _score_darkness,
    "brightness": _score_brightness,
    "flatness": _score_flatness,
    "cast_warm": _score_cast_warm,
    "cast_cool": _score_cast_cool,
    "cast_green": _score_cast_green,
    "chroma": _score_chroma,
    "low_chroma": _score_low_chroma,
    "dark_channel": _score_dark_channel,
    "blockiness": _score_blockiness,
    "clip_blob": _score_clip_blob,
}


@lru_cache(maxsize=4)
def load_calibration(path: Optional[str] = None) -> Dict[str, Any]:
    """Đọc calibration.yaml (cache theo đường dẫn).

    Raises:
        FileNotFoundError: thiếu file hiệu chỉnh.
        ValueError: file thiếu `version` hoặc `defects`.
    """
    source = Path(path) if path else CALIBRATION_PATH
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "version" not in data or "defects" not in data:
        raise ValueError(f"Invalid calibration file: {source}")
    return data


def calibration_version() -> str:
    """Phiên bản bảng hiệu chỉnh đang dùng."""
    return str(load_calibration()["version"])


def defect_params(calibration: Dict[str, Any], defect_type: str) -> Dict[str, Any]:
    """Tham số hàm điểm của một lỗi, kèm cạnh dài tham chiếu chung của bộ hiệu chỉnh."""
    entry = calibration["defects"].get(defect_type) or {}
    return {
        "reference_long_side": calibration.get("reference_long_side", 0),
        **(entry.get("params") or {}),
    }


def defect_score(
    defect_type: str,
    stats: Stats,
    calibration: Dict[str, Any],
    long_side: Optional[int] = None,
) -> Score:
    """Điểm của một lỗi theo hàm điểm khai báo trong calibration; None khi không đo được."""
    entry = calibration["defects"].get(defect_type) or {}
    score_fn = SCORE_FUNCTIONS.get(entry.get("score") or "")
    if score_fn is None:
        return None
    return score_fn(stats, defect_params(calibration, defect_type), long_side)


def severity_for_score(score: float, thresholds: List[float]) -> int:
    """Severity 0–3 = số ngưỡng (tăng dần) mà điểm đạt tới."""
    return sum(1 for threshold in thresholds if score >= threshold)


def _margin_confidence(score: float, thresholds: List[float]) -> float:
    """Hệ số 0.5–1: điểm càng xa ngưỡng gần nhất thì càng chắc về mức severity."""
    finite = [t for t in thresholds if math.isfinite(t)]
    if not finite:
        return 1.0
    span = max(finite) - min(finite) if len(finite) > 1 else abs(finite[0])
    soft = max(CONFIDENCE_SOFT_FRACTION * span, 1e-6)
    distance = min(abs(score - t) for t in finite)
    return 0.5 + 0.5 * min(1.0, distance / soft)


def severity_from_calibration(
    defect_type: str,
    stats: Stats,
    calibration: Dict[str, Any],
    long_side: Optional[int] = None,
) -> Tuple[int, float]:
    """Như `severity_from_measurement` nhưng với bảng hiệu chỉnh cho trước (script khớp dùng)."""
    entry = calibration["defects"].get(defect_type)
    if not entry:
        return NOT_MEASURED
    thresholds = [float(t) for t in entry.get("thresholds") or []]
    severity, confidence = NOT_MEASURED
    score = defect_score(defect_type, stats, calibration, long_side) if thresholds else None
    if score is not None:
        severity = severity_for_score(score[0], thresholds)
        confidence = score[1] * _margin_confidence(score[0], thresholds)
        # Độ tin cậy của chính bảng hiệu chỉnh (kiểm chéo bỏ-một-nguồn), 0..1
        confidence *= 0.5 + 0.5 * float(entry.get("reliability", 1.0))
        if not entry.get("fitted", True):
            # Ngưỡng prior (không khớp trên dữ liệu có nhãn): hạ độ tin cậy
            confidence *= PRIOR_CONFIDENCE_SCALE
    # Lỗi gộp (ví dụ blur lấy mức cao hơn giữa defocus và mờ chuyển động đã xác nhận)
    for other in entry.get("combine_max") or []:
        other_severity, other_confidence = severity_from_calibration(
            other, stats, calibration, long_side
        )
        if other_severity > severity:
            severity, confidence = other_severity, other_confidence
    return severity, round(confidence, 3)


def severity_from_measurement(
    defect_type: str, stats: Stats, *, long_side: Optional[int] = None
) -> Tuple[int, float]:
    """(severity 0..3, confidence 0..1) của một lỗi từ số đo (hợp đồng C4).

    Args:
        defect_type: loại lỗi trong DEFECT_TYPES (kể cả defect mới của P6); loại không có
            hàm điểm hoặc không đo được trả (0, 0.0).
        stats: báo cáo toàn ảnh hoặc thống kê một vùng.
        long_side: cạnh dài ảnh gốc (px) để quy độ mờ về thang tham chiếu. Báo cáo toàn ảnh tự
            lấy từ width/height; RegionStats không có kích thước nên mặc định không quy đổi.
    """
    return severity_from_calibration(defect_type, stats, load_calibration(), long_side)


def defect_severities(
    report: MeasurementReport, long_side: Optional[int] = None
) -> Dict[str, Tuple[int, float]]:
    """Severity của mọi lỗi có trong bảng hiệu chỉnh, cho một báo cáo toàn ảnh."""
    calibration = load_calibration()
    return {
        defect_type: severity_from_calibration(defect_type, report, calibration, long_side)
        for defect_type in calibration["defects"]
    }
