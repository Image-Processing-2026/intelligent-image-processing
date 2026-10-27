"""Phân tích khác biệt giữa ảnh gốc và ảnh đã xử lý (Analysis v2 P7, plan §Phase 7).

Hai phần:
1. Trạng thái từng lỗi cũ trên đúng vùng của nó: severity đo được (C4) trước và sau →
   resolved / improved / unchanged / worse.
2. Tác dụng phụ do xử lý, đo trên pixel (không cần nhãn):
   - halo: vượt ngưỡng (overshoot) quanh cạnh mạnh, sau khi đã khử phần thay đổi tông toàn cục
     (bảng tra mức xám gốc → mức xám mới) và độ lệch tông cục bộ chậm (CLAHE);
   - dark_noise: σ nhiễu ở vùng tối phẳng của ảnh gốc tăng (gamma, CLAHE khuếch đại nhiễu);
   - banding: tỉ lệ mức xám bị bỏ trống trong vùng chuyển sắc mịn tăng (giãn tông → răng lược);
   - channel_clipping: tỉ lệ pixel cháy/bệt kênh màu mới xuất hiện;
   - oversaturation: độ bão hòa C* (p95) tăng mạnh tới mức rực.
Ngưỡng severity của tác dụng phụ là prior (không có bộ nhãn để khớp), ghi trong hằng số module.
Chỉ đo, không sửa pixel (ADR-001).
"""

from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from pydantic import BaseModel, Field

from .calibration import severity_from_measurement
from .schemas import MeasurementReport, MeasurementResult, RegionStats
from .tiles import aggregate

DefectStatus = Literal["resolved", "improved", "unchanged", "worse"]
SideEffectKind = Literal["halo", "dark_noise", "banding", "channel_clipping", "oversaturation"]
Stats = Union[RegionStats, MeasurementReport]
# (loại lỗi, vùng, báo cáo toàn ảnh, số đo theo vùng) → (severity, confidence) hoặc None
Assessor = Callable[
    [str, str, MeasurementReport, Dict[str, RegionStats]], Optional[Tuple[int, float]]
]

# Cạnh dài tối đa khi đo tác dụng phụ (halo vài pixel cần gần độ phân giải gốc)
SIDE_EFFECT_MAX_SIDE = 3000
# Halo: cạnh mạnh = gradient ≥ max(EDGE_MIN_GRADIENT, phân vị EDGE_PERCENTILE) của ảnh gốc
EDGE_MIN_GRADIENT = 12.0
EDGE_PERCENTILE = 97.0
HALO_WINDOW = 7
HALO_TOLERANCE = 2.0
# σ Gaussian của độ lệch tông cục bộ chậm (CLAHE) cần khử trước khi đo overshoot
LOCAL_TONE_SIGMA = 8.0
# Overshoot trung bình quanh cạnh (mức xám). Đo trên 5 ảnh thật: sharpen 0.5 → 0.2–1.4,
# sharpen 1.0 → 1.2–4.6, sharpen 2.0 → 2.5–11.8, CLAHE 2 → 0–6 (ảnh nhiễu)
HALO_THRESHOLDS = (1.5, 4.0, 8.0)
# Nhiễu vùng tối: pixel tối (luma gốc ≤ DARK_LUMA) và phẳng (gradient ≤ FLAT_GRADIENT)
DARK_LUMA = 70.0
FLAT_GRADIENT = 4.0
MIN_DARK_PIXELS = 2000
DARK_NOISE_MIN_SIGMA = 2.0  # σ sau xử lý dưới mức này không nhìn thấy được
# σ sau / σ trước. Đo trên 5 ảnh thật: gamma 2.0 → 1.2–1.6, CLAHE 2 → 1.9–2.6, sharpen 1.0 → 1.9,
# sharpen 2.0 → 2.6–2.9, khử nhiễu → 0.4–0.6
DARK_NOISE_THRESHOLDS = (1.5, 2.2, 3.0)
# Banding: vùng chuyển sắc mịn có gradient trong khoảng này (mức xám/pixel)
GRADIENT_BAND = (0.3, 4.0)
MIN_GRADIENT_PIXELS = 5000
BANDING_THRESHOLDS = (0.12, 0.25, 0.45)  # tăng tỉ lệ mức xám bỏ trống
# Cháy kênh màu mới xuất hiện (tỉ lệ diện tích, sau khi bỏ đốm nhỏ hơn 3×3). Chỉ phía sáng:
# kênh chạm 0 ở vùng tối bão hòa (tăng bão hòa, làm nét) không nhìn thấy được
CLIP_HIGH_LEVEL = 254
CLIPPING_THRESHOLDS = (0.01, 0.03, 0.08)
# Bão hòa: C* p95 tăng ít nhất chừng này, và sau xử lý vượt OVERSATURATION_MIN_CHROMA
OVERSATURATION_MIN_CHROMA = 60.0
OVERSATURATION_THRESHOLDS = (8.0, 15.0, 25.0)


class DefectChange(BaseModel):
    """Thay đổi của một lỗi cũ sau xử lý."""

    type: str
    region: str = "full"
    before: int
    after: int
    status: DefectStatus
    confidence: float = 0.0


class SideEffect(BaseModel):
    """Một tác dụng phụ do xử lý."""

    kind: SideEffectKind
    region: str = "full"
    severity: int = Field(ge=0, le=3)
    value: float
    before: Optional[float] = None
    after: Optional[float] = None
    evidence: str = ""


class DiffReport(BaseModel):
    """Kết quả so sánh ảnh hiện tại với ảnh trước xử lý."""

    defects: List[DefectChange] = Field(default_factory=list)
    side_effects: List[SideEffect] = Field(default_factory=list)

    @property
    def max_side_effect(self) -> int:
        """Severity lớn nhất trong các tác dụng phụ (0 nếu không có)."""
        return max((effect.severity for effect in self.side_effects), default=0)

    def status_of(self, defect_type: str, region: str = "full") -> Optional[DefectChange]:
        """Trạng thái của một lỗi cũ, None nếu không theo dõi."""
        return next((c for c in self.defects if c.type == defect_type and c.region == region), None)


# ---------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------
def _rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    return np.ascontiguousarray(arr.astype(np.uint8))


def _limit(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1.0:
        return image
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _gradient(luma: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return cv2.magnitude(gx, gy)


def _severity(value: float, thresholds: Sequence[float]) -> int:
    return sum(1 for threshold in thresholds if value >= threshold)


def _tone_map(original: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Bảng tra 256 mức: mức xám gốc → trung vị mức xám mới (khử thay đổi tông toàn cục)."""
    levels = np.clip(np.round(original), 0, 255).astype(np.int32).ravel()
    values = current.ravel()
    order = np.argsort(levels, kind="stable")
    sorted_levels, sorted_values = levels[order], values[order]
    bounds = np.searchsorted(sorted_levels, np.arange(257))
    lut = np.full(256, np.nan, dtype=np.float32)
    for level in range(256):
        start, end = bounds[level], bounds[level + 1]
        if end > start:
            lut[level] = float(np.median(sorted_values[start:end]))
    known = np.flatnonzero(~np.isnan(lut))
    if known.size == 0:
        return np.arange(256, dtype=np.float32)
    lut = np.interp(np.arange(256), known, lut[known]).astype(np.float32)
    return lut


# ---------------------------------------------------------
# Tác dụng phụ
# ---------------------------------------------------------
def halo_overshoot(original: np.ndarray, current: np.ndarray) -> Optional[float]:
    """
    Overshoot trung bình (mức xám) quanh cạnh mạnh của ảnh gốc: phần ảnh mới vượt ra ngoài
    khoảng [min, max] cục bộ của ảnh gốc đã ánh xạ tông (toàn cục + lệch tông cục bộ chậm).
    Không có cạnh mạnh → None.
    """
    gradient = _gradient(original)
    threshold = max(EDGE_MIN_GRADIENT, float(np.percentile(gradient, EDGE_PERCENTILE)))
    edges = gradient >= threshold
    if not edges.any():
        return None
    expected = _tone_map(original, current)[np.clip(original, 0, 255).astype(np.uint8)]
    # Độ lệch tông cục bộ chậm (CLAHE, mask vùng) không phải halo: cộng vào kỳ vọng
    expected = expected + cv2.GaussianBlur(current - expected, (0, 0), LOCAL_TONE_SIGMA)
    kernel = np.ones((HALO_WINDOW, HALO_WINDOW), np.uint8)
    local_max = cv2.dilate(expected, kernel)
    local_min = cv2.erode(expected, kernel)
    over = np.maximum(0.0, current - local_max - HALO_TOLERANCE)
    under = np.maximum(0.0, local_min - HALO_TOLERANCE - current)
    band = cv2.dilate(edges.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return float((over + under)[band].mean())


def _robust_sigma(residual: np.ndarray) -> float:
    """σ ước lượng bền (MAD) của phần dư thông cao."""
    median = float(np.median(residual))
    return 1.4826 * float(np.median(np.abs(residual - median)))


def dark_noise_ratio(
    original: np.ndarray, current: np.ndarray
) -> Optional[Tuple[float, float, float]]:
    """
    (σ sau / σ trước, σ trước, σ sau) của nhiễu trên pixel tối và phẳng của ảnh gốc; σ đo bằng
    MAD của phần dư thông cao (ảnh − Gaussian σ 1.5). Quá ít pixel tối phẳng → None.
    """
    smooth = cv2.GaussianBlur(original, (0, 0), 2.0)
    pixels = (smooth <= DARK_LUMA) & (_gradient(smooth) <= FLAT_GRADIENT)
    if int(pixels.sum()) < MIN_DARK_PIXELS:
        return None
    before = _robust_sigma((original - cv2.GaussianBlur(original, (0, 0), 1.5))[pixels])
    after = _robust_sigma((current - cv2.GaussianBlur(current, (0, 0), 1.5))[pixels])
    return after / max(before, 0.25), before, after


def _empty_level_fraction(values: np.ndarray) -> Optional[float]:
    """Tỉ lệ mức xám (nguyên) bị bỏ trống trong khoảng [p1, p99] của các giá trị."""
    if values.size == 0:
        return None
    low, high = np.percentile(values, [1, 99])
    low, high = int(round(low)), int(round(high))
    if high - low < 8:
        return None
    counts = np.bincount(np.clip(np.round(values), 0, 255).astype(np.int32), minlength=256)
    return float((counts[low : high + 1] == 0).mean())


def banding_increase(
    original: np.ndarray, current: np.ndarray
) -> Optional[Tuple[float, float, float]]:
    """
    (tăng tỉ lệ mức xám bỏ trống, trước, sau) trong vùng chuyển sắc mịn của ảnh gốc.
    Không đủ pixel chuyển sắc → None.
    """
    gradient = _gradient(cv2.GaussianBlur(original, (0, 0), 1.0))
    pixels = (gradient >= GRADIENT_BAND[0]) & (gradient <= GRADIENT_BAND[1])
    if int(pixels.sum()) < MIN_GRADIENT_PIXELS:
        return None
    before = _empty_level_fraction(original[pixels])
    after = _empty_level_fraction(current[pixels])
    if before is None or after is None:
        return None
    return after - before, before, after


def clipping_increase(original: np.ndarray, current: np.ndarray) -> Tuple[float, float, float]:
    """(tỉ lệ diện tích mới cháy kênh, đã bỏ đốm nhỏ; tỉ lệ cháy trước; tỉ lệ cháy sau)."""

    def clipped(rgb: np.ndarray) -> np.ndarray:
        return (rgb >= CLIP_HIGH_LEVEL).any(axis=2)

    before, after = clipped(original), clipped(current)
    new = cv2.morphologyEx(
        (after & ~before).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    return float(new.mean()), float(before.mean()), float(after.mean())


def chroma_p95(rgb: np.ndarray) -> float:
    """C* (CIELAB) phân vị 95."""
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    return float(np.percentile(np.hypot(lab[..., 1], lab[..., 2]), 95))


def side_effects(original: np.ndarray, current: np.ndarray) -> List[SideEffect]:
    """Các tác dụng phụ do xử lý (severity ≥ 1), đo trên ảnh ≤ SIDE_EFFECT_MAX_SIDE."""
    first = _limit(_rgb(original), SIDE_EFFECT_MAX_SIDE)
    second = _limit(_rgb(current), SIDE_EFFECT_MAX_SIDE)
    if first.shape != second.shape:
        second = cv2.resize(second, (first.shape[1], first.shape[0]), interpolation=cv2.INTER_AREA)
    luma_before, luma_after = _luma(first), _luma(second)
    effects: List[SideEffect] = []

    halo = halo_overshoot(luma_before, luma_after)
    if halo is not None and _severity(halo, HALO_THRESHOLDS):
        effects.append(
            SideEffect(
                kind="halo",
                severity=_severity(halo, HALO_THRESHOLDS),
                value=round(halo, 3),
                evidence=f"Overshoot quanh cạnh mạnh {halo:.1f} mức xám.",
            )
        )

    noise = dark_noise_ratio(luma_before, luma_after)
    if noise is not None:
        ratio, before, after = noise
        severity = _severity(ratio, DARK_NOISE_THRESHOLDS) if after >= DARK_NOISE_MIN_SIGMA else 0
        if severity:
            effects.append(
                SideEffect(
                    kind="dark_noise",
                    severity=severity,
                    value=round(ratio, 3),
                    before=round(before, 3),
                    after=round(after, 3),
                    evidence=f"σ nhiễu vùng tối {before:.1f} → {after:.1f} (×{ratio:.2f}).",
                )
            )

    banding = banding_increase(luma_before, luma_after)
    if banding is not None and _severity(banding[0], BANDING_THRESHOLDS):
        increase, before, after = banding
        effects.append(
            SideEffect(
                kind="banding",
                severity=_severity(increase, BANDING_THRESHOLDS),
                value=round(increase, 3),
                before=round(before, 3),
                after=round(after, 3),
                evidence=f"Mức xám bỏ trống trong vùng chuyển sắc {before:.0%} → {after:.0%}.",
            )
        )

    new_clip, clip_before, clip_after = clipping_increase(first, second)
    if _severity(new_clip, CLIPPING_THRESHOLDS):
        effects.append(
            SideEffect(
                kind="channel_clipping",
                severity=_severity(new_clip, CLIPPING_THRESHOLDS),
                value=round(new_clip, 4),
                before=round(clip_before, 4),
                after=round(clip_after, 4),
                evidence=f"Vùng cháy kênh màu mới: {new_clip:.1%} diện tích.",
            )
        )

    chroma_before, chroma_after = chroma_p95(first), chroma_p95(second)
    increase = chroma_after - chroma_before
    if chroma_after >= OVERSATURATION_MIN_CHROMA and _severity(increase, OVERSATURATION_THRESHOLDS):
        effects.append(
            SideEffect(
                kind="oversaturation",
                severity=_severity(increase, OVERSATURATION_THRESHOLDS),
                value=round(increase, 2),
                before=round(chroma_before, 2),
                after=round(chroma_after, 2),
                evidence=f"C* p95 {chroma_before:.0f} → {chroma_after:.0f}.",
            )
        )
    return effects


# ---------------------------------------------------------
# Trạng thái lỗi cũ
# ---------------------------------------------------------
def _region_stats(
    result: MeasurementResult, regions: Dict[str, np.ndarray]
) -> Dict[str, RegionStats]:
    """RegionStats theo mask (kèm 'background' = phần bù hợp các vùng)."""
    stats: Dict[str, RegionStats] = {}
    shape = (result.report.height, result.report.width)
    union = np.zeros(shape, dtype=np.float32)
    for name, mask in regions.items():
        mask2d = mask[..., 0] if mask.ndim == 3 else mask
        if mask2d.shape != shape:
            mask2d = cv2.resize(mask2d.astype(np.float32), (shape[1], shape[0]))
        stats[name] = aggregate(result, mask2d.astype(np.float32), name)
        union = np.maximum(union, mask2d.astype(np.float32))
    if regions and "background" not in stats:
        stats["background"] = aggregate(result, 1.0 - union, "background")
    return stats


def calibrated_assessor(
    defect_type: str,
    region: str,
    report: MeasurementReport,
    region_stats: Dict[str, RegionStats],
) -> Optional[Tuple[int, float]]:
    """Mặc định: severity_from_measurement trên vùng (hoặc toàn ảnh); không đo được → None."""
    stats: Stats = region_stats.get(region, report) if region != "full" else report
    if region != "full" and region not in region_stats:
        return None
    severity, confidence = severity_from_measurement(
        defect_type, stats, long_side=max(report.width, report.height)
    )
    return None if severity == 0 and confidence <= 0.0 else (severity, confidence)


def _status(before: int, after: int) -> DefectStatus:
    if after == 0 and before > 0:
        return "resolved"
    if after < before:
        return "improved"
    if after > before:
        return "worse"
    return "unchanged"


def compare(
    original: np.ndarray,
    current: np.ndarray,
    regions: Optional[Dict[str, np.ndarray]] = None,
    *,
    defects: Sequence[Tuple[str, str]] = (),
    original_measurement: Optional[MeasurementResult] = None,
    current_measurement: Optional[MeasurementResult] = None,
    assess: Optional[Assessor] = None,
) -> DiffReport:
    """
    So sánh ảnh hiện tại với ảnh gốc (plan §Phase 7).

    Args:
        original, current: ảnh RGB uint8 cùng nội dung (ảnh xám cũng được).
        regions: mask float32 [0, 1] theo tên vùng (cùng kích thước ảnh).
        defects: các lỗi cũ (loại, vùng) cần theo dõi trạng thái.
        original_measurement, current_measurement: kết quả measure_image đã có (đỡ đo lại);
            thiếu thì không theo dõi được trạng thái lỗi (chỉ đo tác dụng phụ).
        assess: hàm severity cho (loại, vùng); mặc định severity_from_measurement (C4).

    Returns:
        DiffReport: trạng thái từng lỗi đo được trước/sau và các tác dụng phụ severity ≥ 1.
    """
    report = DiffReport(side_effects=side_effects(original, current))
    if not defects or original_measurement is None or current_measurement is None:
        return report
    assess = assess or calibrated_assessor
    regions = regions or {}
    before_stats = _region_stats(original_measurement, regions)
    after_stats = _region_stats(current_measurement, regions)
    for defect_type, region in defects:
        before = assess(defect_type, region, original_measurement.report, before_stats)
        after = assess(defect_type, region, current_measurement.report, after_stats)
        if before is None or after is None:
            continue
        report.defects.append(
            DefectChange(
                type=defect_type,
                region=region,
                before=before[0],
                after=after[0],
                status=_status(before[0], after[0]),
                confidence=round(min(before[1], after[1]), 3),
            )
        )
    return report
