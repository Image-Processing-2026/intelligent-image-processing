"""
Tạo mặt nạ dựa trên hình học và không gian (Spatial & Geometric Mask Generators).
"""

from typing import Tuple

import numpy as np

from .mask_utils import create_soft_mask


def _validate_integer_vector(value: object, *, name: str, size: int) -> tuple[int, ...]:
    """Validate one public API vector without coercing its scalar values."""
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.size != size:
            raise ValueError(f"{name} must be a 1D vector with exactly {size} elements")
        values = value.tolist()
    elif isinstance(value, (tuple, list)):
        if len(value) != size:
            raise ValueError(f"{name} must contain exactly {size} elements")
        values = value
    else:
        raise TypeError(f"{name} must be a tuple, list, or 1D NumPy array")

    validated: list[int] = []
    for item in values:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
            raise TypeError(f"{name} elements must be Python or NumPy integers")
        # Chỉ chuyển sau khi đã xác nhận kiểu để không làm mất dấu bool/float.
        validated.append(int(item))
    return tuple(validated)


def _validate_feather_radius(feather_radius: object) -> int:
    """Validate the historical kernel-size parameter used by ``create_soft_mask``."""
    if isinstance(feather_radius, (bool, np.bool_)) or not isinstance(
        feather_radius, (int, np.integer)
    ):
        raise TypeError("feather_radius must be a non-negative integer")
    if feather_radius < 0:
        raise ValueError("feather_radius must be non-negative")
    return int(feather_radius)


def create_bbox_mask(
    image_shape: Tuple[int, int] | list[int] | np.ndarray,
    bbox: Tuple[int, int, int, int] | list[int] | np.ndarray,
    feather_radius: int = 15,
) -> np.ndarray:
    """Tạo soft mask toàn ảnh từ bounding box theo khoảng nửa kín.

    Args:
        image_shape: Tuple/list/ndarray 1D đúng hai số nguyên ``(H, W)``.
            Hai kích thước phải dương; không truyền trực tiếp ``image.shape``.
        bbox: Tuple/list/ndarray 1D đúng bốn số nguyên
            ``(xmin, ymin, xmax, ymax)``. Biên phải và biên dưới không bao gồm.
            Tọa độ ngoài ảnh được phép và sẽ được clip sau khi kiểm tra thứ tự.
        feather_radius: Số nguyên không âm, được chuyển tiếp theo quy ước của
            :func:`create_soft_mask` (giá trị này là kích thước kernel lịch sử).

    Returns:
        Mảng mới ``float32`` có shape ``(H, W)`` và giá trị trong ``[0, 1]``.

    Raises:
        TypeError: Nếu container hoặc một scalar có kiểu không hợp lệ.
        ValueError: Nếu số chiều/số phần tử sai, kích thước không dương, bbox
            đảo chiều hoặc ``feather_radius`` âm.

    Bbox rỗng hoặc nằm hoàn toàn ngoài ảnh trả về mask toàn số 0. Bbox phủ
    toàn ảnh trả về mask toàn số 1, kể cả khi có yêu cầu feathering. Bbox đảo
    chiều luôn là lỗi, kể cả khi clipping có thể làm mất dấu hiệu đó.
    """

    h, w = _validate_integer_vector(image_shape, name="image_shape", size=2)
    if h <= 0 or w <= 0:
        raise ValueError("image_shape dimensions must be positive")
    xmin, ymin, xmax, ymax = _validate_integer_vector(bbox, name="bbox", size=4)
    radius = _validate_feather_radius(feather_radius)

    if xmin > xmax or ymin > ymax:
        raise ValueError("bbox coordinates must be ordered as xmin <= xmax and ymin <= ymax")

    # Clip trước khi raster hóa; slicing dùng đúng quy ước [start, end).
    x0 = max(0, min(w, xmin))
    x1 = max(0, min(w, xmax))
    y0 = max(0, min(h, ymin))
    y1 = max(0, min(h, ymax))

    binary_mask = np.zeros((h, w), dtype=np.uint8)
    if x0 < x1 and y0 < y1:
        binary_mask[y0:y1, x0:x1] = 255
    return create_soft_mask(binary_mask, feather_radius=radius)


def create_quadrant_mask(
    image_shape: Tuple[int, int], quadrant: str, feather_radius: int = 25
) -> np.ndarray:
    """
    Tạo mặt nạ không gian theo góc phân tư (top, bottom, left, right, center).
    """
    h, w = image_shape[:2]
    binary_mask = np.zeros((h, w), dtype=np.uint8)

    if quadrant == "top":
        binary_mask[0 : h // 2, :] = 255
    elif quadrant == "bottom":
        binary_mask[h // 2 : h, :] = 255
    elif quadrant == "left":
        binary_mask[:, 0 : w // 2] = 255
    elif quadrant == "right":
        binary_mask[:, w // 2 : w] = 255
    elif quadrant == "center":
        binary_mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 255
    else:
        binary_mask[:, :] = 255

    return create_soft_mask(binary_mask, feather_radius=feather_radius)
