"""
Hiệu chỉnh độ sáng và độ tương phản (Brightness & Contrast Enhancements).
Bao gồm: Gamma Correction, Cân bằng lược đồ xám (Histogram Equalization), và CLAHE.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from .base import apply_region_op


def _raw_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Biến đổi gamma: I_out = 255 * (I_in / 255) ^ (1 / gamma)."""
    if abs(gamma - 1.0) < 1e-3:
        return image
    # Tạo bảng ánh xạ LUT (Lookup Table) để tăng tốc độ tính toán
    inv_gamma = 1.0 / max(0.01, gamma)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)


def apply_gamma(
    image: np.ndarray, mask: Optional[np.ndarray] = None, gamma: float = 1.0
) -> np.ndarray:
    """
    Hiệu chỉnh độ sáng phi tuyến tính theo vùng bằng hàm Gamma.
    gamma > 1.0 làm sáng ảnh tối; gamma < 1.0 làm tối ảnh chói.
    """
    return apply_region_op(image, mask, _raw_gamma, gamma=gamma)


def _raw_clahe(
    image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Cân bằng lược đồ độ sáng cục bộ thích ứng có giới hạn độ tương phản (CLAHE) trong không gian LAB."""
    # Chuyển đổi sang không gian màu LAB để chỉ xử lý kênh độ sáng L
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe_op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe_op.apply(l_ch)

    lab_merged = cv2.merge((l_enhanced, a_ch, b_ch))
    return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2RGB)


def apply_clahe(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Áp dụng CLAHE lên vùng ảnh chỉ định để cải thiện độ tương phản cục bộ.
    """
    return apply_region_op(
        image, mask, _raw_clahe, clip_limit=clip_limit, tile_grid_size=tile_grid_size
    )
