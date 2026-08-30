"""
Khung cơ bản và bộ hòa trộn vùng xử lý (Base Region Operation Wrapper).
Đảm bảo mọi thao tác xử lý ảnh đều tự động hỗ trợ mặt nạ vùng (soft-mask).
"""

from typing import Any, Callable, Optional
import numpy as np
from src.region_engine.mask_utils import blend_regions


def apply_region_op(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    op_func: Callable[..., np.ndarray],
    **kwargs: Any
) -> np.ndarray:
    """
    Hàm bọc (wrapper) tiêu chuẩn áp dụng thuật toán xử lý ảnh lên vùng chỉ định.

    Args:
        image: Ảnh gốc đầu vào (RGB, uint8).
        mask: Mặt nạ mềm (float32 [0.0, 1.0]), None nếu áp dụng cho toàn bộ ảnh.
        op_func: Hàm xử lý ảnh thuần túy nhận vào `image` và các tham số `kwargs`.
        kwargs: Các siêu tham số của thuật toán xử lý ảnh.

    Returns:
        np.ndarray: Ảnh sau khi đã xử lý và hòa trộn mượt mà.
    """
    # 1. Áp dụng thuật toán xử lý ảnh lên toàn bộ khung hình
    processed = op_func(image, **kwargs)

    # 2. Nếu không có mặt nạ, trả về kết quả áp dụng toàn cục
    if mask is None:
        return processed

    # 3. Nếu có mặt nạ, hòa trộn kết quả bằng Alpha Compositing
    return blend_regions(image, processed, mask)
