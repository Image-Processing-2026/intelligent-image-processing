"""Lưới tile và gộp bản đồ theo mask mềm (ADR-003).

Bản đồ tile có hình (ceil(H/T), ceil(W/T)). Tile ở mép có thể nhỏ hơn T nên mọi trung bình
tính theo số pixel thực tế, không theo T².
"""

from typing import Dict, Tuple

import numpy as np

from .schemas import Estimate, MeasurementResult, RegionStats

MAP_KEYS: Tuple[str, ...] = (
    "exposure.mean_l",
    "exposure.clip_high",
    "exposure.clip_low",
    "noise.sigma_luma",
    "sharpness.edge_width",
    "sharpness.defocus_sigma",
    "color.chroma",
)

# Số tile hợp lệ tối thiểu để confidence của một vùng đạt tối đa.
MIN_TILES_FULL_CONFIDENCE = 4


def tile_grid_shape(height: int, width: int, tile_size: int) -> Tuple[int, int]:
    """Kích thước lưới tile (gh, gw) cho ảnh height × width."""
    return -(-height // tile_size), -(-width // tile_size)


def _blocks(values: np.ndarray, tile_size: int, fill: float) -> np.ndarray:
    """Đệm mảng 2D về bội số của tile rồi reshape thành (gh, T, gw, T)."""
    height, width = values.shape
    gh, gw = tile_grid_shape(height, width, tile_size)
    padded = np.full((gh * tile_size, gw * tile_size), fill, dtype=np.float64)
    padded[:height, :width] = values
    return padded.reshape(gh, tile_size, gw, tile_size)


def tile_mean(values: np.ndarray, tile_size: int) -> np.ndarray:
    """Trung bình theo tile, bỏ qua NaN. Tile không có giá trị hữu hạn → NaN."""
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    sums = _blocks(np.where(finite, arr, 0.0), tile_size, 0.0).sum(axis=(1, 3))
    counts = _blocks(finite.astype(np.float64), tile_size, 0.0).sum(axis=(1, 3))
    means = np.full(counts.shape, np.nan)
    np.divide(sums, counts, out=means, where=counts > 0)
    return means.astype(np.float32)


def tile_sum(values: np.ndarray, tile_size: int) -> np.ndarray:
    """Tổng theo tile; giá trị không hữu hạn coi như 0."""
    arr = np.asarray(values, dtype=np.float64)
    return _blocks(np.where(np.isfinite(arr), arr, 0.0), tile_size, 0.0).sum(axis=(1, 3))


def tile_mean_at(
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    height: int,
    width: int,
    tile_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Trung bình theo tile cho mẫu rời rạc đặt tại tọa độ pixel (rows, cols).

    Trả (mean, count): mean là NaN ở tile không có mẫu hữu hạn; count là số mẫu hữu hạn.
    """
    gh, gw = tile_grid_shape(height, width, tile_size)
    index = (rows // tile_size) * gw + (cols // tile_size)
    vals = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(vals)
    sums = np.bincount(index[ok], weights=vals[ok], minlength=gh * gw)
    counts = np.bincount(index[ok], minlength=gh * gw).astype(np.float64)
    means = np.full(gh * gw, np.nan)
    np.divide(sums, counts, out=means, where=counts > 0)
    return means.reshape(gh, gw).astype(np.float32), counts.reshape(gh, gw)


def _as_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Chuẩn hóa mask về (H, W) float64 trong [0, 1]; sai hình thì raise ValueError."""
    arr = np.asarray(mask)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.shape != (height, width):
        raise ValueError(f"mask shape {arr.shape} does not match image size ({height}, {width})")
    return np.clip(np.nan_to_num(arr.astype(np.float64), nan=0.0), 0.0, 1.0)


def aggregate(result: MeasurementResult, mask: np.ndarray, region: str = "full") -> RegionStats:
    """Gộp bản đồ tile theo mask mềm (H, W) thành thống kê của một vùng.

    Trọng số mỗi tile là tổng mask trong tile. Giá trị vùng là trung bình có trọng số, bỏ qua
    NaN. Confidence = tỉ lệ trọng số nằm trên tile đo được × độ đủ số tile. Mask toàn 0 cho
    mask_confidence = 0 và không có key nào trong values.

    Raises:
        ValueError: mask không khớp kích thước ảnh đã đo.
    """
    report = result.report
    soft_mask = _as_mask(mask, report.height, report.width)
    weights = tile_sum(soft_mask, result.tile_size)
    total_weight = float(weights.sum())
    area_fraction = float(soft_mask.mean())
    if total_weight <= 0.0:
        return RegionStats(
            region=region, area_fraction=area_fraction, values={}, mask_confidence=0.0
        )

    values: Dict[str, Estimate] = {}
    for key in MAP_KEYS:
        grid = result.maps.get(key)
        if grid is None:
            continue
        valid = np.isfinite(grid) & (weights > 0)
        valid_weight = float(weights[valid].sum())
        if valid_weight <= 0.0:
            continue
        value = float((grid[valid].astype(np.float64) * weights[valid]).sum() / valid_weight)
        coverage = valid_weight / total_weight
        n_tiles = int(valid.sum())
        confidence = coverage * min(1.0, n_tiles / MIN_TILES_FULL_CONFIDENCE)
        values[key] = Estimate(
            value=value, confidence=min(1.0, max(0.0, confidence)), n_samples=n_tiles
        )

    return RegionStats(
        region=region,
        area_fraction=area_fraction,
        values=values,
        mask_source="unknown",
        mask_confidence=1.0,
    )
