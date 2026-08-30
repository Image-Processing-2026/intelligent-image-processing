"""
Tăng cường độ sắc nét (Image Sharpening).
Bao gồm: Mặt nạ làm nét (Unsharp Masking) và Toán tử vi phân bậc hai Laplacian.
"""

from typing import Literal, Optional
import cv2
import numpy as np
from .base import apply_region_op


def _raw_sharpen(
    image: np.ndarray,
    method: Literal["unsharp_mask", "laplacian"] = "unsharp_mask",
    amount: float = 1.0
) -> np.ndarray:
    """Thuật toán làm nét thuần túy."""
    if method == "unsharp_mask":
        # Unsharp masking: I_sharp = I + amount * (I - I_blurred)
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=2.0)
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    elif method == "laplacian":
        # Kernel vi phân Laplacian 3x3
        kernel = np.array([[0, -1, 0],
                           [-1, 4 + amount, -1],
                           [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(image, -1, kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    return image


def apply_sharpen(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    method: Literal["unsharp_mask", "laplacian"] = "unsharp_mask",
    amount: float = 1.0
) -> np.ndarray:
    """
    Tăng cường độ sắc nét cho vùng chỉ định thông qua mặt nạ mềm.
    """
    return apply_region_op(image, mask, _raw_sharpen, method=method, amount=amount)
