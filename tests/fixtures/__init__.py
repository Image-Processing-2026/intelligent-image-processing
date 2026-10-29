"""
Module fixtures cho các bài kiểm thử unit và integration của hệ thống.
Cung cấp các generator tạo ảnh giả lập chuẩn hóa cho các tình huống kiểm thử xử lý ảnh.
"""

from .synthetic_images import (
    create_black_image,
    create_blurred_image,
    create_checkerboard_image,
    create_color_cast_image,
    create_flat_image,
    create_noisy_image,
    create_overexposed_image,
    create_salt_pepper_image,
    create_underexposed_image,
    create_white_image,
)

__all__ = [
    "create_flat_image",
    "create_black_image",
    "create_white_image",
    "create_checkerboard_image",
    "create_noisy_image",
    "create_salt_pepper_image",
    "create_blurred_image",
    "create_color_cast_image",
    "create_underexposed_image",
    "create_overexposed_image",
]
