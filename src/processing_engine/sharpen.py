"""
Tăng cường độ sắc nét (Image Sharpening).
Bao gồm: Mặt nạ làm nét (Unsharp Masking) và Toán tử vi phân bậc hai Laplacian.
"""

import math
from typing import Literal, Optional

import cv2
import numpy as np

from .base import apply_region_op, require_choice, require_number

SHARPEN_METHODS = ("unsharp_mask", "laplacian")

# σ của bộ làm mờ trong unsharp masking (pixel)
UNSHARP_BLUR_SIGMA = 2.0
# Bán kính ảnh hưởng để cắt theo mask (xem base.apply_region_op). Unsharp: kernel Gauss
# bán kính ≈ 3σ, lấy 4σ + 1 cho dư. Laplacian: None — filter2D số thực của OpenCV làm tròn
# khác nhau theo vị trí điểm ảnh (lệch 1 mức xám), nên luôn xử lý toàn ảnh.
_CROP_HALO = {"unsharp_mask": math.ceil(4 * UNSHARP_BLUR_SIGMA) + 1, "laplacian": None}


def _raw_sharpen(
    image: np.ndarray,
    method: Literal["unsharp_mask", "laplacian"] = "unsharp_mask",
    amount: float = 1.0,
) -> np.ndarray:
    """Thuật toán làm nét thuần túy."""
    if method == "unsharp_mask":
        # Unsharp masking: I_sharp = I + amount * (I - I_blurred)
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=UNSHARP_BLUR_SIGMA)
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    else:  # method == "laplacian" (đã được kiểm tra ở apply_sharpen)
        # I_sharp = I - amount * ∇²I: kernel = đơn vị + amount * (-Laplacian 3x3).
        # Tổng hệ số bằng 1 nên vùng phẳng giữ nguyên độ sáng; chỉ cạnh được tăng tương phản.
        kernel = np.array(
            [[0, -amount, 0], [-amount, 1 + 4 * amount, -amount], [0, -amount, 0]],
            dtype=np.float32,
        )
        sharpened = cv2.filter2D(image.astype(np.float32), -1, kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_sharpen(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    method: Literal["unsharp_mask", "laplacian"] = "unsharp_mask",
    amount: float = 1.0,
) -> np.ndarray:
    """
    Tăng cường độ sắc nét cho vùng chỉ định thông qua mặt nạ mềm.

    Raises:
        ProcessingInputError: method không thuộc SHARPEN_METHODS, amount âm hoặc không phải số,
            hoặc ảnh/mặt nạ vi phạm hợp đồng.
    """
    require_choice(method, SHARPEN_METHODS, "method", "apply_sharpen")
    amount = require_number(amount, "amount", "apply_sharpen", ge=0.0)
    return apply_region_op(
        image, mask, _raw_sharpen, crop_halo=_CROP_HALO[method], method=method, amount=amount
    )
