"""
Hiệu chỉnh màu sắc và cân bằng trắng (Color Correction & White Balance).
"""

from typing import Optional

import cv2
import numpy as np

from .base import apply_region_op


def _raw_color_balance(
    image: np.ndarray, saturation_scale: float = 1.0, temperature_shift: float = 0.0
) -> np.ndarray:
    """Hiệu chỉnh màu sắc và cân bằng nhiệt độ màu."""
    # 1. Điều chỉnh độ bão hòa màu trong không gian HSV
    if abs(saturation_scale - 1.0) > 1e-3:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # 2. Điều chỉnh nhiệt độ màu (Temperature: Warm/Cool)
    if abs(temperature_shift) > 1e-3:
        img_f = image.astype(np.float32)
        # Tăng kênh R và giảm kênh B nếu shift > 0 (ấm hơn), ngược lại nếu shift < 0 (lạnh hơn)
        img_f[:, :, 0] = np.clip(img_f[:, :, 0] + temperature_shift * 20.0, 0, 255)
        img_f[:, :, 2] = np.clip(img_f[:, :, 2] - temperature_shift * 20.0, 0, 255)
        image = img_f.astype(np.uint8)

    return image


def apply_color_balance(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    saturation_scale: float = 1.0,
    temperature_shift: float = 0.0,
) -> np.ndarray:
    """
    Cân chỉnh màu sắc cho vùng ảnh xác định theo mặt nạ mềm.
    """
    return apply_region_op(
        image,
        mask,
        _raw_color_balance,
        saturation_scale=saturation_scale,
        temperature_shift=temperature_shift,
    )
