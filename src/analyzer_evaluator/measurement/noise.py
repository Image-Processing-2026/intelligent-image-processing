"""Ước lượng nhiễu: Liu, Tanaka, Okutomi (2013) trên luma và chroma, σ(I) theo độ sáng, Immerkær.

Ý tưởng: patch 7×7 (bước 3) có năng lượng gradient thấp được coi là phẳng (chỉ có nhiễu). Trên các
patch phẳng, nhiễu đẳng hướng làm hiệp phương sai có trị riêng nhỏ nhất ≈ σ², còn texture chỉ chiếm
vài chiều. Ngưỡng năng lượng được lặp theo σ hiện tại đến khi hội tụ. Đơn vị σ là mức 0–255.

Trị riêng nhỏ nhất của mẫu hữu hạn bị lệch thấp (định luật Marchenko–Pastur), nên được hiệu chỉnh
theo số mẫu hiệu dụng. Immerkær (1996) giữ làm chỉ số phụ, lệch khi có texture.
"""

import math
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ._util import Section, put_estimate
from .schemas import Estimate
from .tiles import tile_mean_at

PATCH = 7
STRIDE = 3
DIM = PATCH * PATCH
# Năng lượng gradient (sai phân tiến) của nhiễu thuần có kỳ vọng 2σ² + 2σ² = 4σ².
NOISE_GRAD_ENERGY = 4.0
GRAD_THRESHOLD_MARGIN = 3.0
FALLBACK_FRACTION = 0.1
MIN_FLAT_PATCHES = 30
# Dưới số mẫu này, hiệu chỉnh Marchenko–Pastur không đáng tin: không ước lượng σ.
MIN_EFF_SAMPLES = 8 * DIM
FULL_CONFIDENCE_PATCHES = 100
MIN_TILE_PATCHES = 8
MIN_BIN_PATCHES = 30
N_BINS = 8
MAX_ITERATIONS = 12
CONVERGENCE_TOL = 0.01
CHUNK_PATCHES = 100_000
IMMERKAER_CONFIDENCE = 0.5

_IMMERKAER_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


@dataclass
class _PlaneFit:
    """Kết quả ước lượng σ trên một mặt phẳng (luma hoặc chroma)."""

    sigma: float  # NaN nếu không đủ patch phẳng
    selected: np.ndarray  # (nh, nw) bool: patch được coi là phẳng
    energy: np.ndarray  # (nh, nw) năng lượng gradient trung bình tại tâm patch
    n_flat: int


@dataclass
class _Stats:
    """Thống kê thô (đếm, tổng, tổng tích) của các patch theo nhãn."""

    count: np.ndarray  # (L,)
    s1: np.ndarray  # (L, DIM)
    s2: np.ndarray  # (L, DIM, DIM)


def _grid_shape(height: int, width: int) -> Tuple[int, int]:
    """Số patch theo mỗi chiều (0 nếu ảnh nhỏ hơn patch)."""
    nh = (height - PATCH) // STRIDE + 1 if height >= PATCH else 0
    nw = (width - PATCH) // STRIDE + 1 if width >= PATCH else 0
    return nh, nw


def _sample_centers(smooth: np.ndarray, nh: int, nw: int) -> np.ndarray:
    """Lấy giá trị tại tâm của các patch (góc trên trái + 3)."""
    half = PATCH // 2
    return smooth[half : half + STRIDE * nh : STRIDE, half : half + STRIDE * nw : STRIDE]


def _center_coords(nh: int, nw: int) -> Tuple[np.ndarray, np.ndarray]:
    """Tọa độ pixel (hàng, cột) của tâm patch, theo thứ tự lưới (nh, nw)."""
    half = PATCH // 2
    rows = np.arange(nh) * STRIDE + half
    cols = np.arange(nw) * STRIDE + half
    return np.broadcast_to(rows[:, None], (nh, nw)), np.broadcast_to(cols[None, :], (nh, nw))


def _patch_energy(plane: np.ndarray) -> np.ndarray:
    """Năng lượng gradient trung bình (dx² + dy²) trong mỗi patch, tại tâm patch."""
    h, w = plane.shape
    gx = np.zeros_like(plane)
    gy = np.zeros_like(plane)
    gx[:, :-1] = plane[:, 1:] - plane[:, :-1]
    gy[:-1, :] = plane[1:, :] - plane[:-1, :]
    energy = (gx * gx + gy * gy).astype(np.float32)
    smooth = cv2.boxFilter(
        energy, cv2.CV_32F, (PATCH, PATCH), normalize=True, borderType=cv2.BORDER_REFLECT_101
    )
    nh, nw = _grid_shape(h, w)
    return _sample_centers(smooth, nh, nw).astype(np.float64)


def _patch_blocks(plane: np.ndarray) -> Iterator[Tuple[slice, np.ndarray]]:
    """Duyệt patch theo khối hàng của lưới, mỗi khối là (số patch, DIM) float64."""
    windows = sliding_window_view(plane, (PATCH, PATCH))[::STRIDE, ::STRIDE]
    nh, nw = windows.shape[:2]
    rows_per_chunk = max(1, CHUNK_PATCHES // max(1, nw))
    for r0 in range(0, nh, rows_per_chunk):
        r1 = min(nh, r0 + rows_per_chunk)
        block = windows[r0:r1].reshape(-1, DIM).astype(np.float64)
        yield slice(r0, r1), block


def _label_stats(plane: np.ndarray, labels: np.ndarray, n_labels: int) -> _Stats:
    """Gom thống kê patch theo nhãn; nhãn -1 bị bỏ qua."""
    count = np.zeros(n_labels)
    s1 = np.zeros((n_labels, DIM))
    s2 = np.zeros((n_labels, DIM, DIM))
    for rows, block in _patch_blocks(plane):
        flat_labels = labels[rows].reshape(-1)
        for label in range(n_labels):
            members = block[flat_labels == label]
            if members.shape[0] == 0:
                continue
            count[label] += members.shape[0]
            s1[label] += members.sum(axis=0)
            s2[label] += members.T @ members
    return _Stats(count=count, s1=s1, s2=s2)


def _sigma_from_stats(stats: _Stats) -> np.ndarray:
    """σ = sqrt(λ_min đã hiệu chỉnh) cho từng nhãn; NaN khi không đủ mẫu hiệu dụng."""
    sigmas = np.full(stats.count.shape, np.nan)
    for label, n in enumerate(stats.count):
        if n < MIN_EFF_SAMPLES:
            continue
        mean = stats.s1[label] / n
        cov = stats.s2[label] / n - np.outer(mean, mean)
        lambda_min = max(float(np.linalg.eigvalsh(cov)[0]), 0.0)
        # Hiệu chỉnh Marchenko–Pastur: cận dưới của phổ là σ²(1 − sqrt(p / n))².
        correction = (1.0 - math.sqrt(DIM / n)) ** 2
        sigmas[label] = math.sqrt(lambda_min / correction)
    return sigmas


def _fallback_selection(energy: np.ndarray) -> np.ndarray:
    """Chọn FALLBACK_FRACTION patch có năng lượng gradient thấp nhất."""
    k = min(energy.size, max(MIN_EFF_SAMPLES, int(round(FALLBACK_FRACTION * energy.size))))
    threshold = np.partition(energy.ravel(), k - 1)[k - 1]
    return energy <= threshold


def _fit_plane(plane: np.ndarray) -> _PlaneFit:
    """Ước lượng σ bằng lặp chọn patch phẳng (Liu 2013)."""
    nh, nw = _grid_shape(*plane.shape)
    if nh == 0 or nw == 0:
        return _PlaneFit(math.nan, np.zeros((0, 0), bool), np.zeros((0, 0)), 0)
    energy = _patch_energy(plane)
    selected = _fallback_selection(energy)
    sigma = _sigma_of_selection(plane, selected)
    for _ in range(MAX_ITERATIONS):
        if not math.isfinite(sigma):
            break
        candidate = energy <= GRAD_THRESHOLD_MARGIN * NOISE_GRAD_ENERGY * sigma * sigma
        if int(candidate.sum()) < MIN_FLAT_PATCHES:
            candidate = _fallback_selection(energy)
        new_sigma = _sigma_of_selection(plane, candidate)
        selected = candidate
        if not math.isfinite(new_sigma):
            break
        converged = abs(new_sigma - sigma) <= CONVERGENCE_TOL * max(sigma, 1e-6)
        sigma = new_sigma
        if converged:
            break
    return _PlaneFit(sigma=sigma, selected=selected, energy=energy, n_flat=int(selected.sum()))


def _sigma_of_selection(plane: np.ndarray, selected: np.ndarray) -> float:
    labels = np.where(selected, 0, -1)
    return float(_sigma_from_stats(_label_stats(plane, labels, 1))[0])


def _bin_sigmas(plane: np.ndarray, fit: _PlaneFit) -> np.ndarray:
    """σ(I): σ của các patch phẳng chia theo độ sáng trung bình patch thành N_BINS bin."""
    nh, nw = fit.selected.shape
    patch_mean = _sample_centers(
        cv2.boxFilter(plane, cv2.CV_32F, (PATCH, PATCH), normalize=True), nh, nw
    )
    bins = np.clip((patch_mean / 256.0 * N_BINS).astype(np.int64), 0, N_BINS - 1)
    labels = np.where(fit.selected, bins, -1)
    stats = _label_stats(plane, labels, N_BINS)
    return np.where(stats.count >= MIN_BIN_PATCHES, _sigma_from_stats(stats), np.nan)


def _tile_sigma_map(plane: np.ndarray, fit: _PlaneFit, tile_size: int) -> np.ndarray:
    """σ cục bộ theo tile: căn bậc hai của phương sai trung bình các patch phẳng trong tile."""
    h, w = plane.shape
    nh, nw = fit.selected.shape
    p64 = plane.astype(np.float64)
    mean = _sample_centers(cv2.boxFilter(p64, cv2.CV_64F, (PATCH, PATCH), normalize=True), nh, nw)
    mean_sq = _sample_centers(
        cv2.boxFilter(p64 * p64, cv2.CV_64F, (PATCH, PATCH), normalize=True), nh, nw
    )
    variance = (mean_sq - mean * mean) * DIM / (DIM - 1)
    rows, cols = _center_coords(nh, nw)
    flat = fit.selected.reshape(-1)
    means, counts = tile_mean_at(
        rows.reshape(-1)[flat],
        cols.reshape(-1)[flat],
        variance.reshape(-1)[flat],
        h,
        w,
        tile_size,
    )
    sigma_map = np.sqrt(np.maximum(means.astype(np.float64), 0.0))
    sigma_map[counts < MIN_TILE_PATCHES] = np.nan
    return sigma_map.astype(np.float32)


def _immerkaer(plane: np.ndarray) -> float:
    """Ước lượng Immerkær (1996): σ = sqrt(π/2) / (6 (W−2)(H−2)) · Σ|I * M|."""
    h, w = plane.shape
    if h < 3 or w < 3:
        return math.nan
    response = cv2.filter2D(plane.astype(np.float64), -1, _IMMERKAER_KERNEL)[1:-1, 1:-1]
    total = float(np.abs(response).sum())
    return math.sqrt(math.pi / 2.0) * total / (6.0 * (w - 2) * (h - 2))


def _confidence(n_flat: int) -> float:
    return min(1.0, n_flat / FULL_CONFIDENCE_PATCHES)


def _chroma_planes(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cr và Cb (float32). OpenCV xếp YCrCb nên Cr là kênh 1, Cb là kênh 2."""
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    return ycrcb[..., 1].astype(np.float32), ycrcb[..., 2].astype(np.float32)


def measure_noise(
    rgb: np.ndarray, tile_size: int, *, grayscale: bool
) -> Tuple[Section, Optional[float]]:
    """Đo nhiễu. Trả (section, σ luma hoặc None) — σ luma dùng để khử nhiễu trước khi đo độ nét.

    Ảnh xám không có chroma nên sigma_cb/cr = 0 với confidence 0 và có cảnh báo.
    """
    section = Section()
    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fit = _fit_plane(luma)
    confidence = _confidence(fit.n_flat)
    put_estimate(section, "sigma_luma", fit.sigma, confidence, fit.n_flat)
    if not math.isfinite(fit.sigma) or fit.n_flat < MIN_FLAT_PATCHES:
        section.notes.append(f"few flat patches for noise estimate (n={fit.n_flat})")

    if math.isfinite(fit.sigma):
        bins = _bin_sigmas(luma, fit)
        dark = [s for s in bins[:3] if math.isfinite(s)]
        bright = [s for s in bins[N_BINS - 3 :] if math.isfinite(s)]
        if dark and bright and float(np.mean(bright)) > 0:
            ratio = float(np.mean(dark)) / float(np.mean(bright))
            put_estimate(section, "sigma_dark_ratio", ratio, confidence, fit.n_flat)
        section.maps["noise.sigma_luma"] = _tile_sigma_map(luma, fit, tile_size)

    put_estimate(section, "immerkaer", _immerkaer(luma), IMMERKAER_CONFIDENCE, luma.size)

    if grayscale:
        section.notes.append("grayscale input: chroma noise not measured")
        section.estimates["sigma_cb"] = Estimate(value=0.0, confidence=0.0, n_samples=0)
        section.estimates["sigma_cr"] = Estimate(value=0.0, confidence=0.0, n_samples=0)
    else:
        cr, cb = _chroma_planes(rgb)
        for key, plane in (("sigma_cb", cb), ("sigma_cr", cr)):
            chroma = _fit_plane(plane)
            put_estimate(section, key, chroma.sigma, _confidence(chroma.n_flat), chroma.n_flat)

    luma_estimate = section.estimates.get("sigma_luma")
    return section, (luma_estimate.value if luma_estimate is not None else None)
