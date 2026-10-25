"""Đo độ nét và mờ: độ rộng cạnh, defocus (Zhuo & Sim 2011), hướng gradient, Laplacian.

Tách nhiễu khỏi mờ: khi nhiễu đáng kể, ảnh được làm mịn Gaussian σ_s đã biết trước, rồi phần σ_s
được trừ khỏi kết quả (phương sai của các mức mờ cộng dồn). Ngưỡng chọn cạnh nằm trên mức gradient
của nhiễu sau làm mịn, nên nhiễu không tạo cạnh. Bộ lọc song phương không được dùng: nó làm dốc các
chuyển tiếp mờ và đẩy σ defocus giảm khoảng 40 %.

Độ rộng cạnh dùng khoảng cách 10–90 % biên độ dọc pháp tuyến cạnh, là biến thể của Marziliano
(2002): ảnh 8-bit mờ thường đơn điệu, không có cực trị cục bộ ổn định để Marziliano dùng.
"""

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from ._util import Section, put_estimate
from .tiles import tile_mean_at

EDGE_PERCENTILE = 90.0
# Canny so sánh > ngưỡng cao, nên hạ nhẹ để cạnh mạnh nhất (đúng bằng phân vị) vẫn được nhận.
CANNY_HIGH_FACTOR = 0.9
CANNY_LOW_FACTOR = 1.0
# Cạnh phải vượt NOISE_EDGE_SIGMAS lần độ lệch chuẩn của thành phần gradient nhiễu.
NOISE_EDGE_SIGMAS = 5.0
# Sobel 3×3 trên nhiễu trắng σ: mỗi thành phần gradient có độ lệch chuẩn √12·σ.
SOBEL_NOISE_STD = math.sqrt(12.0)
# Làm mịn Gaussian khi nhiễu ≥ PRE_SMOOTH_MIN_NOISE. Gaussian 2D làm độ lệch chuẩn nhiễu trắng giảm
# đi 1/(2√π σ_s).
PRE_SMOOTH_SIGMA = 1.0
PRE_SMOOTH_MAX = 2.0
PRE_SMOOTH_NOISE_DIVISOR = 5.0
PRE_SMOOTH_MIN_NOISE = 3.0
# Độ rộng 10–90 % của cạnh Gaussian σ là 2·1.2816·σ.
GAUSSIAN_WIDTH_10_90 = 2.5631
PROFILE_RADIUS = 12
REMAP_CHUNK = 20_000
MIN_AMPLITUDE = 10.0
MIN_EDGES = 30
FULL_CONFIDENCE_EDGES = 200
MAX_EDGE_SAMPLES = 200_000
NORM_LONG_SIDE = 1000.0
DEFOCUS_SIGMA0 = 1.0
DEFOCUS_MAX = 10.0
EPSILON = 1e-6


def _pre_smooth_sigma(sigma: Optional[float]) -> float:
    """σ của Gaussian làm mịn trước khi đo: 0 khi nhiễu nhỏ hoặc không biết, tăng theo nhiễu."""
    if sigma is None or not math.isfinite(sigma) or sigma < PRE_SMOOTH_MIN_NOISE:
        return 0.0
    return min(PRE_SMOOTH_MAX, max(PRE_SMOOTH_SIGMA, sigma / PRE_SMOOTH_NOISE_DIVISOR))


def _gradient_noise_floor(noise_std: float) -> float:
    """Ngưỡng gradient dưới đó không coi là cạnh vì nhiễu (độ lệch chuẩn nhiễu sau làm mịn)."""
    return NOISE_EDGE_SIGMAS * SOBEL_NOISE_STD * noise_std


def _remove_pre_smoothing(values: np.ndarray, pre_sigma: float, scale: float) -> np.ndarray:
    """Bỏ phần mờ do làm mịn: x_gốc = sqrt(x² − (scale·σ_s)²). Không làm mịn thì giữ nguyên."""
    if pre_sigma <= 0.0:
        return values
    removed = (scale * pre_sigma) ** 2
    return np.sqrt(np.maximum(values * values - removed, 0.0))


def _crossing(profile: np.ndarray, level: float) -> Tuple[np.ndarray, np.ndarray]:
    """Vị trí nội suy tuyến tính của lần đầu profile ≥ level, theo từng hàng.

    Trả (position, reached). Hàng không đạt mức có reached = False.
    """
    above = profile >= level
    reached = above.any(axis=1)
    first = np.argmax(above, axis=1)
    previous = np.maximum(first - 1, 0)
    rows = np.arange(profile.shape[0])
    t_now = profile[rows, first]
    t_prev = profile[rows, previous]
    fraction = np.clip((level - t_prev) / np.maximum(t_now - t_prev, EPSILON), 0.0, 1.0)
    position = previous + fraction * (first - previous)
    return position.astype(np.float64), reached


def _edge_widths(
    image: np.ndarray, ys: np.ndarray, xs: np.ndarray, nx: np.ndarray, ny: np.ndarray
) -> np.ndarray:
    """Độ rộng cạnh 10–90 % dọc pháp tuyến; NaN ở điểm không đo được."""
    widths = np.empty(ys.size, dtype=np.float64)
    # cv2.remap giới hạn số hàng của bản đồ (SHRT_MAX), nên lấy mẫu profile theo khối.
    for start in range(0, ys.size, REMAP_CHUNK):
        block = slice(start, start + REMAP_CHUNK)
        widths[block] = _profile_widths(image, ys[block], xs[block], nx[block], ny[block])
    return widths


def _profile_widths(
    image: np.ndarray, ys: np.ndarray, xs: np.ndarray, nx: np.ndarray, ny: np.ndarray
) -> np.ndarray:
    """Độ rộng 10–90 % cho một khối điểm cạnh."""
    offsets = np.arange(-PROFILE_RADIUS, PROFILE_RADIUS + 1, dtype=np.float32)
    map_x = (xs[:, None] + offsets[None, :] * nx[:, None]).astype(np.float32)
    map_y = (ys[:, None] + offsets[None, :] * ny[:, None]).astype(np.float32)
    profile = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    low = profile.min(axis=1)
    amplitude = profile.max(axis=1) - low
    rising = profile[:, -1] >= profile[:, 0]
    normalized = (profile - low[:, None]) / np.maximum(amplitude, EPSILON)[:, None]
    normalized = np.where(rising[:, None], normalized, 1.0 - normalized)
    pos10, reached10 = _crossing(normalized, 0.1)
    pos90, reached90 = _crossing(normalized, 0.9)
    width = np.abs(pos90 - pos10)
    usable = reached10 & reached90 & (amplitude >= MIN_AMPLITUDE)
    return np.where(usable, width, np.nan)


def _defocus_sigmas(
    image: np.ndarray, gx: np.ndarray, gy: np.ndarray, ys: np.ndarray, xs: np.ndarray
) -> np.ndarray:
    """σ defocus tại cạnh (Zhuo & Sim): σ = σ₀ / sqrt(R² − 1), R = |∇I| / |∇(I * G_σ₀)|."""
    reblur = cv2.GaussianBlur(image, (0, 0), DEFOCUS_SIGMA0)
    rx = cv2.Sobel(reblur, cv2.CV_32F, 1, 0, ksize=3)
    ry = cv2.Sobel(reblur, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx[ys, xs], gy[ys, xs]).astype(np.float64)
    mag_reblur = np.hypot(rx[ys, xs], ry[ys, xs]).astype(np.float64)
    ratio_sq_minus_one = (mag / np.maximum(mag_reblur, EPSILON)) ** 2 - 1.0
    usable = (mag_reblur > EPSILON) & (ratio_sq_minus_one > 1e-3)
    sigma = DEFOCUS_SIGMA0 / np.sqrt(np.maximum(ratio_sq_minus_one, 1e-3))
    return np.where(usable, np.minimum(sigma, DEFOCUS_MAX), np.nan)


def _subsample(count: int) -> np.ndarray:
    """Chỉ số đều để giới hạn số điểm cạnh xử lý."""
    if count <= MAX_EDGE_SAMPLES:
        return np.arange(count)
    return np.linspace(0, count - 1, MAX_EDGE_SAMPLES).astype(np.int64)


def _median(values: np.ndarray) -> Tuple[float, int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan, 0
    return float(np.median(finite)), int(finite.size)


def measure_sharpness(rgb: np.ndarray, tile_size: int, sigma_luma: Optional[float]) -> Section:
    """Đo độ nét trên ảnh RGB uint8. sigma_luma (từ đo nhiễu) dùng để chọn cạnh và làm mịn."""
    height, width = rgb.shape[:2]
    section = Section()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    pre_sigma = _pre_smooth_sigma(sigma_luma)
    image = gray.astype(np.float32)
    noise_std = float(sigma_luma) if sigma_luma is not None and math.isfinite(sigma_luma) else 0.0
    if pre_sigma > 0.0:
        image = cv2.GaussianBlur(image, (0, 0), pre_sigma)
        noise_std /= 2.0 * math.sqrt(math.pi) * pre_sigma

    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    positive = magnitude[magnitude > 0]
    floor = _gradient_noise_floor(noise_std)
    if positive.size:
        strong_threshold = max(float(np.percentile(positive, EDGE_PERCENTILE)), floor)
    else:
        strong_threshold = np.inf
    strong = magnitude >= strong_threshold

    _motion(section, gx, gy, strong)
    _edges(section, image, gx, gy, magnitude, tile_size, height, width, floor, pre_sigma)

    laplacian = cv2.Laplacian(image.astype(np.float64), cv2.CV_64F)
    put_estimate(section, "laplacian_var", float(laplacian.var()), 1.0, gray.size)
    return section


def _motion(section: Section, gx: np.ndarray, gy: np.ndarray, strong: np.ndarray) -> None:
    """Structure tensor trên gradient mạnh: coherence của hướng và hướng trội (độ, 0–180)."""
    n_strong = int(strong.sum())
    if n_strong == 0:
        return
    sx = gx[strong].astype(np.float64)
    sy = gy[strong].astype(np.float64)
    jxx = float((sx * sx).sum())
    jyy = float((sy * sy).sum())
    jxy = float((sx * sy).sum())
    trace = jxx + jyy
    disc = math.hypot((jxx - jyy) / 2.0, jxy)
    lam_max = trace / 2.0 + disc
    lam_min = trace / 2.0 - disc
    coherence = (lam_max - lam_min) / max(lam_max + lam_min, EPSILON)
    angle = math.degrees(0.5 * math.atan2(2.0 * jxy, jxx - jyy)) % 180.0
    confidence = min(1.0, n_strong / FULL_CONFIDENCE_EDGES)
    put_estimate(section, "motion_coherence", coherence, confidence, n_strong)
    put_estimate(section, "motion_angle_deg", angle, confidence, n_strong)


def _edges(
    section: Section,
    image: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    magnitude: np.ndarray,
    tile_size: int,
    height: int,
    width: int,
    floor: float,
    pre_sigma: float,
) -> None:
    """Độ rộng cạnh và σ defocus tại điểm cạnh (Canny trên ngưỡng nhiễu); đưa về bản đồ tile."""
    positive = magnitude[magnitude > 0]
    if positive.size == 0:
        section.notes.append("no gradient: sharpness not measured")
        return
    high = max(CANNY_HIGH_FACTOR * float(np.percentile(positive, EDGE_PERCENTILE)), floor)
    edges = cv2.Canny(image.astype(np.uint8), CANNY_LOW_FACTOR * high, high, L2gradient=True)
    ys_all, xs_all = np.nonzero(edges)
    keep = _subsample(ys_all.size)
    ys, xs = ys_all[keep], xs_all[keep]

    edge_mag = magnitude[ys, xs]
    usable = edge_mag > EPSILON
    ys, xs, edge_mag = ys[usable], xs[usable], edge_mag[usable]
    nx = gx[ys, xs] / edge_mag
    ny = gy[ys, xs] / edge_mag

    raw_widths = _edge_widths(image, ys, xs, nx, ny)
    # Một điểm chỉ là cạnh khi profile có bậc thật; điểm chỉ có nhiễu bị loại khỏi cả hai đo.
    is_step = np.isfinite(raw_widths)
    widths = _remove_pre_smoothing(raw_widths, pre_sigma, GAUSSIAN_WIDTH_10_90)
    defocus = _defocus_sigmas(image, gx, gy, ys, xs)
    defocus = _remove_pre_smoothing(np.where(is_step, defocus, np.nan), pre_sigma, 1.0)

    width_px, n_width = _median(widths)
    confidence = min(1.0, n_width / FULL_CONFIDENCE_EDGES)
    put_estimate(section, "edge_width_px", width_px, confidence, n_width)
    put_estimate(
        section,
        "edge_width_norm",
        width_px * NORM_LONG_SIDE / max(height, width),
        confidence,
        n_width,
    )
    sigma, n_sigma = _median(defocus)
    put_estimate(
        section, "defocus_sigma", sigma, min(1.0, n_sigma / FULL_CONFIDENCE_EDGES), n_sigma
    )
    if min(n_width, n_sigma) < MIN_EDGES:
        section.notes.append(f"few edges for sharpness estimate (n={min(n_width, n_sigma)})")

    section.maps["sharpness.edge_width"], _ = tile_mean_at(ys, xs, widths, height, width, tile_size)
    section.maps["sharpness.defocus_sigma"], _ = tile_mean_at(
        ys, xs, defocus, height, width, tile_size
    )
