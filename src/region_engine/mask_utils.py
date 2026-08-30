"""
Tiện ích xử lý mặt nạ và hòa trộn biên mềm (Soft-Mask & Alpha Blending Utilities).
Đảm bảo kết quả xử lý cục bộ không bị lộ viền cắt (seam artifacts).
"""

import cv2
import numpy as np


def create_soft_mask(binary_mask: np.ndarray, feather_radius: int = 15) -> np.ndarray:
    """
    Chuyển đổi mặt nạ nhị phân (0 hoặc 1/255) thành mặt nạ mềm (soft mask) float32 [0.0, 1.0].
    Sử dụng bộ lọc Gauss để làm mờ dần đường biên (edge feathering).

    Args:
        binary_mask: Mặt nạ nhị phân shape (H, W) uint8 hoặc bool.
        feather_radius: Bán kính làm mờ biên (phải là số lẻ).

    Returns:
        np.ndarray: Mặt nạ mềm kiểu float32, khoảng giá trị [0.0, 1.0].
    """
    # Chuẩn hóa về float32 [0.0, 1.0]
    mask_f32 = binary_mask.astype(np.float32)
    if mask_f32.max() > 1.0:
        mask_f32 /= 255.0

    if feather_radius <= 0:
        return mask_f32

    # Đảm bảo kernel size là số lẻ
    ksize = feather_radius if feather_radius % 2 == 1 else feather_radius + 1
    # Áp dụng Gaussian Blur để làm mềm biên
    soft_mask = cv2.GaussianBlur(mask_f32, (ksize, ksize), sigmaX=ksize / 3.0)

    return np.clip(soft_mask, 0.0, 1.0)


def blend_regions(
    original_image: np.ndarray, processed_image: np.ndarray, soft_mask: np.ndarray
) -> np.ndarray:
    """
    Hòa trộn ảnh gốc và ảnh đã xử lý thông qua mặt nạ mềm (Alpha Compositing).

    Công thức: I_out = soft_mask * I_processed + (1.0 - soft_mask) * I_original

    Args:
        original_image: Ảnh gốc ban đầu (RGB, uint8).
        processed_image: Ảnh sau khi áp dụng thuật toán xử lý ảnh (RGB, uint8).
        soft_mask: Mặt nạ mềm float32 shape (H, W) hoặc (H, W, 1) trong khoảng [0.0, 1.0].

    Returns:
        np.ndarray: Ảnh đã hòa trộn hoàn chỉnh (RGB, uint8).
    """
    if soft_mask is None:
        return processed_image

    # Đảm bảo mặt nạ có 3 kênh màu nếu ảnh là RGB
    if len(original_image.shape) == 3 and (len(soft_mask.shape) == 2 or soft_mask.shape[2] == 1):
        if len(soft_mask.shape) == 2:
            soft_mask = np.expand_dims(soft_mask, axis=-1)
        soft_mask_3ch = np.repeat(soft_mask, original_image.shape[2], axis=-1)
    else:
        soft_mask_3ch = soft_mask

    orig_f = original_image.astype(np.float32)
    proc_f = processed_image.astype(np.float32)

    # Tính toán hòa trộn điểm ảnh
    blended = soft_mask_3ch * proc_f + (1.0 - soft_mask_3ch) * orig_f
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)
