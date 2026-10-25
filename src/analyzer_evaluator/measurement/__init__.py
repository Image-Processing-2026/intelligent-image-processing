"""Lõi đo cục bộ (Module 1, Analysis v2 P1).

Đo phép đo classical trên ảnh độ phân giải gốc, theo tile, có độ tin cậy. Không dùng generative AI
(ADR-001). Hợp đồng: docs/analysis_v2/contracts.md §C1.
"""

from typing import Tuple

import numpy as np

from .schemas import Estimate, MeasurementReport, MeasurementResult, RegionStats
from .tiles import MAP_KEYS, aggregate

__all__ = [
    "Estimate",
    "MeasurementReport",
    "MeasurementResult",
    "RegionStats",
    "MAP_KEYS",
    "aggregate",
    "measure_image",
]


def _to_rgb_uint8(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Chuẩn hóa về RGB uint8 (H, W, 3). Trả thêm cờ ảnh xám.

    Ảnh xám (H, W) hoặc (H, W, 1) được lặp thành 3 kênh để các module đo chạy đồng nhất.

    Raises:
        ValueError: sai số chiều, không đủ kênh, hoặc không phải uint8.
    """
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    grayscale = arr.ndim == 2
    if grayscale:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected (H, W), (H, W, 1) or (H, W, 3) image, got {image.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("image must not be empty")
    if arr.dtype != np.uint8:
        raise ValueError(f"image must be uint8, got {arr.dtype}")
    return np.ascontiguousarray(arr), grayscale


def measure_image(image: np.ndarray, *, tile_size: int = 64) -> MeasurementResult:
    """Đo toàn ảnh: trả báo cáo tóm tắt và bản đồ tile.

    Args:
        image: ảnh RGB uint8 (H, W, 3); ảnh xám (H, W) hoặc (H, W, 1) cũng được.
        tile_size: cạnh tile (pixel) cho bản đồ.

    Raises:
        ValueError: ảnh sai định dạng hoặc tile_size < 1.
    """
    if tile_size < 1:
        raise ValueError(f"tile_size must be >= 1, got {tile_size}")
    rgb, _grayscale = _to_rgb_uint8(image)
    height, width = rgb.shape[:2]
    report = MeasurementReport(width=width, height=height, tile_size=tile_size)
    return MeasurementResult(report=report, maps={}, tile_size=tile_size)
