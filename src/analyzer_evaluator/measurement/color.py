"""Đo màu: nguồn sáng ước lượng từ pixel trung tính, độ lệch so với trung tính, và C*.

Pixel tin được là pixel không cháy kênh, L* > 15 và C* < NEUTRAL_CHROMA. Bốn thuật toán (Gray-World,
White-Patch p99, Shades-of-Gray p = 6, Gray-Edge bậc 1, p = 6, σ = 2) ước lượng nguồn sáng trên
RGB tuyến tính của tập đó. Độ lệch góc giữa các thuật toán là độ đồng thuận: lệch quá 5° thì độ
tin cậy giảm. Ước lượng không dùng pixel có màu thật (cảnh vàng vẫn cho nguồn sáng trung tính nếu
có patch trung tính).
"""

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ._util import Section, put_estimate, srgb_to_linear
from .tiles import tile_mean

CLIP_HIGH = 250
CLIP_LOW = 5
MIN_LIGHTNESS = 15.0
NEUTRAL_CHROMA = 12.0
MIN_NEUTRAL_PIXELS = 100
FULL_CONFIDENCE_NEUTRAL = 2000
SPREAD_TOLERANCE_DEG = 5.0
SPREAD_PENALTY_DEG = 10.0
WHITE_PATCH_PERCENTILE = 99.0
SHADES_POWER = 6.0
GRAY_EDGE_POWER = 6.0
GRAY_EDGE_SIGMA = 2.0
# Bán kính loại trừ quanh pixel không trung tính: đuôi Gaussian (cắt ở 3σ, có sai số ~1e-3 của
# gradient cạnh) cộng sai phân Sobel, có dư 2 px.
GRADIENT_CONTEXT = 2 * (math.ceil(4 * GRAY_EDGE_SIGMA) + 2) + 1
MIN_EDGE_GRADIENT = 1e-3
# Chọn pixel trung tính lặp: tối đa số vòng bù–lọc lại, dừng khi ước lượng đổi < ngưỡng (độ).
# Pixel chỉ được nhận lại nếu C* gốc < CAST_MAX_CHROMA: xám dưới ám mạnh nhất cần đo (≈ 3000 K)
# có C* ≤ ~32, còn màu thật bão hòa (cỏ: C* ≥ 32 ở phân vị 1) không thể "trôi" vào tập trung tính.
CAST_MAX_CHROMA = 30.0
NEUTRAL_ITERATIONS = 6
NEUTRAL_CONVERGED_DEG = 0.25
_NEUTRAL_UNIT = np.full(3, 1.0 / math.sqrt(3.0))


def _power_mean(values: np.ndarray, power: float) -> np.ndarray:
    """Trung bình Minkowski theo cột: (mean(x^p))^(1/p)."""
    return np.mean(values**power, axis=0) ** (1.0 / power)


def _unit(vector: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        return None
    return vector / norm


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Góc giữa hai vector (độ), kẹp để tránh lỗi làm tròn của arccos."""
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _illuminant_estimates(linear: np.ndarray, gradient: Optional[np.ndarray]) -> List[np.ndarray]:
    """Ước lượng nguồn sáng (chưa chuẩn hóa) từ RGB tuyến tính của pixel tin được.

    Gray-Edge cần gradient ở pixel nằm xa mọi pixel không trung tính; gradient là None thì bỏ qua.
    """
    estimates = [
        np.mean(linear, axis=0),
        np.percentile(linear, WHITE_PATCH_PERCENTILE, axis=0),
        _power_mean(linear, SHADES_POWER),
    ]
    if gradient is not None:
        estimates.append(_power_mean(gradient, GRAY_EDGE_POWER))
    return estimates


def _away_from_non_neutral(neutral: np.ndarray) -> np.ndarray:
    """Pixel trung tính nằm ngoài vùng hỗ trợ của Gray-Edge quanh pixel không trung tính.

    Ranh giới cảnh ấm/trung tính tạo gradient rất lớn; nếu giữ lại, Gray-Edge sẽ đo cạnh đó.
    """
    non_neutral = (~neutral).astype(np.uint8)
    kernel = np.ones((GRADIENT_CONTEXT, GRADIENT_CONTEXT), dtype=np.uint8)
    near = cv2.dilate(non_neutral, kernel) > 0
    return neutral & ~near


def _gradient_magnitude(linear: np.ndarray) -> np.ndarray:
    """Độ lớn gradient theo từng kênh sau làm mịn Gaussian σ (cho Gray-Edge)."""
    channels = []
    for c in range(linear.shape[2]):
        plane = np.ascontiguousarray(linear[..., c])
        smooth = cv2.GaussianBlur(plane, (0, 0), GRAY_EDGE_SIGMA)
        dx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
        channels.append(np.hypot(dx, dy))
    return np.stack(channels, axis=2)


def _consensus(
    linear_all: np.ndarray, gradient_all: np.ndarray, neutral: np.ndarray
) -> Optional[Tuple[np.ndarray, float]]:
    """Nguồn sáng đồng thuận (vector đơn vị) và độ lệch góc lớn nhất giữa các thuật toán.

    Trả None khi tập pixel trung tính rỗng hoặc có ít hơn hai ước lượng hợp lệ.
    """
    if not neutral.any():
        return None
    linear = linear_all[neutral].astype(np.float64)
    # Chỉ dùng pixel có gradient thật: vùng phẳng chỉ có nhiễu số, làm hướng Gray-Edge bị ngẫu nhiên.
    edge_mask = _away_from_non_neutral(neutral) & (gradient_all.max(axis=2) > MIN_EDGE_GRADIENT)
    gradient = gradient_all[edge_mask].astype(np.float64) if edge_mask.any() else None
    units = [_unit(v) for v in _illuminant_estimates(linear, gradient)]
    units = [u for u in units if u is not None]
    if len(units) < 2:
        return None
    spread = max(_angle_deg(a, b) for i, a in enumerate(units) for b in units[i + 1 :])
    consensus = _unit(np.mean(units, axis=0))
    if consensus is None:
        return None
    return consensus, spread


def _neutral_after_correction(
    linear_all: np.ndarray, illuminant: np.ndarray, usable: np.ndarray
) -> np.ndarray:
    """Pixel trung tính (C* < NEUTRAL_CHROMA) sau khi bù nguồn sáng kiểu von Kries.

    Ám màu mạnh đẩy chính các pixel xám ra khỏi cổng C*; bù trước rồi mới lọc thì chúng quay lại.
    `usable` đã chỉ gồm pixel có thể là xám bị ám (C* gốc < CAST_MAX_CHROMA).
    """
    gains = illuminant / float(np.mean(illuminant))
    corrected = np.clip(linear_all / gains.astype(np.float32), 0.0, 1.0)
    encoded = np.where(
        corrected <= 0.0031308, corrected * 12.92, 1.055 * np.power(corrected, 1 / 2.4) - 0.055
    ).astype(np.float32)
    lab = cv2.cvtColor(encoded, cv2.COLOR_RGB2LAB)
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    return usable & (chroma < NEUTRAL_CHROMA)


def _illuminant(
    section: Section, linear_all: np.ndarray, neutral: np.ndarray, usable: np.ndarray
) -> None:
    """Nguồn sáng ước lượng, độ lệch so trung tính, độ đồng thuận giữa thuật toán.

    Chọn pixel trung tính lặp: ước lượng trên tập C* < 12 của ảnh gốc, bù ảnh theo ước lượng đó,
    lọc lại C* trên ảnh đã bù rồi ước lượng lại trên pixel gốc, tới khi hội tụ. Bắt đầu từ tập
    trung tính của ảnh gốc (không từ Gray-World) để cảnh ấm có patch xám không bị kéo lệch; chỉ
    nhận lại pixel có C* gốc < CAST_MAX_CHROMA để màu thật bão hòa (cỏ) không bị coi là xám.
    """
    count = int(neutral.sum())
    if count < MIN_NEUTRAL_PIXELS:
        section.notes.append(f"few neutral pixels for illuminant estimate (n={count})")
    gradient_all = _gradient_magnitude(linear_all) if count else None
    estimate = _consensus(linear_all, gradient_all, neutral) if count else None
    if estimate is None:
        return
    for _ in range(NEUTRAL_ITERATIONS):
        candidate = _neutral_after_correction(linear_all, estimate[0], usable)
        if int(candidate.sum()) < max(1, count // 2):
            break
        refined = _consensus(linear_all, gradient_all, candidate)
        if refined is None:
            break
        moved = _angle_deg(refined[0], estimate[0])
        neutral, count, estimate = candidate, int(candidate.sum()), refined
        if moved < NEUTRAL_CONVERGED_DEG:
            break
    consensus, spread = estimate
    chromaticity = consensus / float(consensus.sum())
    count_factor = min(1.0, count / FULL_CONFIDENCE_NEUTRAL)
    spread_factor = min(
        1.0, max(0.0, 1.0 - max(0.0, spread - SPREAD_TOLERANCE_DEG) / SPREAD_PENALTY_DEG)
    )
    confidence = count_factor * spread_factor
    put_estimate(
        section, "illuminant_angle_deg", _angle_deg(consensus, _NEUTRAL_UNIT), confidence, count
    )
    put_estimate(section, "illuminant_r", float(chromaticity[0]), confidence, count)
    put_estimate(section, "illuminant_g", float(chromaticity[1]), confidence, count)
    put_estimate(section, "illuminant_b", float(chromaticity[2]), confidence, count)
    put_estimate(section, "algorithm_spread_deg", spread, count_factor, count)


def measure_color(rgb: np.ndarray, tile_size: int) -> Section:
    """Đo màu trên ảnh RGB uint8 (H, W, 3). Ảnh xám cho nguồn sáng trung tính."""
    height, width = rgb.shape[:2]
    total = height * width
    section = Section()
    rgb01 = rgb.astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb01, cv2.COLOR_RGB2LAB)
    l_star = lab[..., 0]
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)

    clipped = ((rgb >= CLIP_HIGH) | (rgb <= CLIP_LOW)).any(axis=2)
    usable = (~clipped) & (l_star > MIN_LIGHTNESS)
    neutral = usable & (chroma < NEUTRAL_CHROMA)
    p50, p95 = np.percentile(chroma, [50, 95])
    put_estimate(section, "chroma_p50", float(p50), 1.0, total)
    put_estimate(section, "chroma_p95", float(p95), 1.0, total)
    put_estimate(section, "neutral_fraction", float(neutral.mean()), 1.0, total)
    section.maps["color.chroma"] = tile_mean(chroma, tile_size)

    _illuminant(section, srgb_to_linear(rgb01), neutral, usable & (chroma < CAST_MAX_CHROMA))
    return section
