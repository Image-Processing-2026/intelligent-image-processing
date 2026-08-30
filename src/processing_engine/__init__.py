"""
Module Classical Image Processing Engine.
Cung cấp các thuật toán xử lý ảnh kinh điển theo từng vùng thông qua mặt nạ mềm.
"""

from .base import apply_region_op
from .color import apply_color_balance
from .denoise import apply_denoise
from .exposure_contrast import apply_clahe, apply_gamma
from .sharpen import apply_sharpen

__all__ = [
    "apply_region_op",
    "apply_denoise",
    "apply_gamma",
    "apply_clahe",
    "apply_sharpen",
    "apply_color_balance",
]
