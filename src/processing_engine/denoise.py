"""
Các thuật toán khử nhiễu kinh điển (Classical Image Denoising Algorithms).
Bao gồm: Lọc Gauss, Lọc Trung vị (Median), Lọc Song phương (Bilateral), Non-Local Means (NLM).
"""

from typing import Literal, Optional

import cv2
import numpy as np

from .base import apply_region_op, require_choice, require_number

DENOISE_METHODS = ("gaussian", "median", "bilateral", "nlm")


def _raw_denoise(
    image: np.ndarray,
    method: Literal["gaussian", "median", "bilateral", "nlm"] = "bilateral",
    strength: float = 1.0,
) -> np.ndarray:
    """Thuật toán khử nhiễu thuần túy trên toàn ảnh."""
    if method == "gaussian":
        ksize = int(max(3, int(strength * 5)))
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=strength * 1.5)

    elif method == "median":
        ksize = int(max(3, int(strength * 3)))
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        return cv2.medianBlur(image, ksize)

    elif method == "bilateral":
        # Bộ lọc song phương: bảo toàn cạnh biên trong khi khử nhiễu bề mặt
        d = int(max(5, int(strength * 5)))
        sigma_color = float(strength * 50.0)
        sigma_space = float(strength * 50.0)
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    else:  # method == "nlm" (đã được kiểm tra ở apply_denoise)
        # Non-Local Means: khử nhiễu bằng cách tìm kiếm mẫu lặp lại trong ảnh
        h_lum = float(strength * 10.0)
        if image.ndim == 2:
            return cv2.fastNlMeansDenoising(image, None, h_lum, 7, 21)
        return cv2.fastNlMeansDenoisingColored(image, None, h_lum, h_lum, 7, 21)


def apply_denoise(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    method: Literal["gaussian", "median", "bilateral", "nlm"] = "bilateral",
    strength: float = 1.0,
) -> np.ndarray:
    """
    Khử nhiễu cục bộ theo mặt nạ mềm hoặc toàn cục.

    Raises:
        ProcessingInputError: method không thuộc DENOISE_METHODS, strength không phải số > 0,
            hoặc ảnh/mặt nạ vi phạm hợp đồng.
    """
    require_choice(method, DENOISE_METHODS, "method", "apply_denoise")
    strength = require_number(strength, "strength", "apply_denoise", gt=0.0)
    return apply_region_op(image, mask, _raw_denoise, method=method, strength=strength)
