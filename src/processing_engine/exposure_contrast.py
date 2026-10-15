"""
Hiệu chỉnh độ sáng và độ tương phản (Brightness & Contrast Enhancements).
Bao gồm: Gamma Correction, Cân bằng lược đồ xám (Histogram Equalization), và CLAHE.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from .base import ProcessingInputError, apply_region_op, require_number


def _raw_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Biến đổi gamma: I_out = 255 * (I_in / 255) ^ (1 / gamma)."""
    if abs(gamma - 1.0) < 1e-3:
        return image
    # Tạo bảng ánh xạ LUT (Lookup Table) 256 phần tử: tính lũy thừa 1 lần cho mỗi mức xám
    # thay vì cho từng điểm ảnh. Giữ phép cắt cụt (astype) vì test tích hợp của Module 2 khóa LUT này.
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)


def apply_gamma(
    image: np.ndarray, mask: Optional[np.ndarray] = None, gamma: float = 1.0
) -> np.ndarray:
    """
    Hiệu chỉnh độ sáng phi tuyến tính theo vùng bằng hàm Gamma.
    gamma > 1.0 làm sáng ảnh tối; gamma < 1.0 làm tối ảnh chói.

    Raises:
        ProcessingInputError: gamma không phải số > 0, hoặc ảnh/mặt nạ vi phạm hợp đồng.
    """
    gamma = require_number(gamma, "gamma", "apply_gamma", gt=0.0)
    return apply_region_op(image, mask, _raw_gamma, gamma=gamma)


def _raw_clahe(
    image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Cân bằng lược đồ độ sáng cục bộ thích ứng có giới hạn độ tương phản (CLAHE) trong không gian LAB."""
    clahe_op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Ảnh grayscale (1 kênh): áp CLAHE trực tiếp lên kênh sáng duy nhất
    if image.ndim == 2:
        return clahe_op.apply(image)

    # Ảnh màu (3 kênh): chuyển sang LAB để chỉ xử lý kênh độ sáng L, tránh lệch màu
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_enhanced = clahe_op.apply(l_ch)
    lab_merged = cv2.merge((l_enhanced, a_ch, b_ch))
    return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2RGB)


def _validate_tile_grid(tile_grid_size: object) -> Tuple[int, int]:
    """Lưới tile phải là 2 số nguyên dương; chấp nhận list (VD: từ JSON) và chuyển thành tuple."""
    if (
        not isinstance(tile_grid_size, (tuple, list))
        or len(tile_grid_size) != 2
        or not all(
            isinstance(v, (int, np.integer)) and not isinstance(v, bool) and v > 0
            for v in tile_grid_size
        )
    ):
        raise ProcessingInputError(
            f"apply_clahe: tile_grid_size must be two positive integers, got {tile_grid_size!r}"
        )
    return (int(tile_grid_size[0]), int(tile_grid_size[1]))


def apply_clahe(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Áp dụng CLAHE lên vùng ảnh chỉ định để cải thiện độ tương phản cục bộ.

    Raises:
        ProcessingInputError: clip_limit không phải số > 0, tile_grid_size không phải
            2 số nguyên dương, hoặc ảnh/mặt nạ vi phạm hợp đồng.
    """
    clip_limit = require_number(clip_limit, "clip_limit", "apply_clahe", gt=0.0)
    tile_grid_size = _validate_tile_grid(tile_grid_size)
    return apply_region_op(
        image, mask, _raw_clahe, clip_limit=clip_limit, tile_grid_size=tile_grid_size
    )
