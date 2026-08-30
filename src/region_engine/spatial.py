"""
Tạo mặt nạ dựa trên hình học và không gian (Spatial & Geometric Mask Generators).
"""

from typing import Tuple
import numpy as np
from .mask_utils import create_soft_mask


def create_bbox_mask(
    image_shape: Tuple[int, int],
    bbox: Tuple[int, int, int, int],
    feather_radius: int = 15
) -> np.ndarray:
    """
    Tạo mặt nạ mềm từ hộp giới hạn (Bounding Box: xmin, ymin, xmax, ymax).

    Args:
        image_shape: Kích thước ảnh (H, W).
        bbox: Tọa độ (xmin, ymin, xmax, ymax).
        feather_radius: Độ mịn viền.

    Returns:
        np.ndarray: Soft mask float32 [0.0, 1.0].
    """
    h, w = image_shape[:2]
    binary_mask = np.zeros((h, w), dtype=np.uint8)
    xmin, ymin, xmax, ymax = bbox
    xmin = max(0, min(w, xmin))
    xmax = max(0, min(w, xmax))
    ymin = max(0, min(h, ymin))
    ymax = max(0, min(h, ymax))

    binary_mask[ymin:ymax, xmin:xmax] = 255
    return create_soft_mask(binary_mask, feather_radius=feather_radius)


def create_quadrant_mask(
    image_shape: Tuple[int, int],
    quadrant: str,
    feather_radius: int = 25
) -> np.ndarray:
    """
    Tạo mặt nạ không gian theo góc phân tư (top, bottom, left, right, center).
    """
    h, w = image_shape[:2]
    binary_mask = np.zeros((h, w), dtype=np.uint8)

    if quadrant == "top":
        binary_mask[0:h // 2, :] = 255
    elif quadrant == "bottom":
        binary_mask[h // 2:h, :] = 255
    elif quadrant == "left":
        binary_mask[:, 0:w // 2] = 255
    elif quadrant == "right":
        binary_mask[:, w // 2:w] = 255
    elif quadrant == "center":
        binary_mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 255
    else:
        binary_mask[:, :] = 255

    return create_soft_mask(binary_mask, feather_radius=feather_radius)
