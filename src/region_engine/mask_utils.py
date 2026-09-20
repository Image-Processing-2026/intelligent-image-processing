"""
Tiện ích xử lý mặt nạ và hòa trộn biên mềm (Soft-Mask & Alpha Blending Utilities).
Đảm bảo kết quả xử lý cục bộ không bị lộ viền cắt (seam artifacts).
"""

import cv2
import numpy as np


def create_soft_mask(binary_mask: np.ndarray, feather_radius: int = 15) -> np.ndarray:
    """Create a reproducible ``float32`` Gaussian-feathered binary mask.

    ``feather_radius`` is retained for API compatibility, but it denotes the
    Gaussian kernel size: odd values are used as-is and even values are
    rounded up to the next odd value.

    Args:
        binary_mask: A non-empty 2D ``bool`` or ``uint8`` array. ``uint8``
            values must be entirely from ``{0, 1}`` or entirely from
            ``{0, 255}``.
        feather_radius: A non-negative Python or NumPy integer.

    Returns:
        A new ``float32`` array with values in ``[0.0, 1.0]``.

    Raises:
        TypeError: If the array, dtype, or feather parameter has an invalid
            type.
        ValueError: If the mask shape/values or feather parameter is invalid.
    """
    if not isinstance(binary_mask, np.ndarray):
        raise TypeError("binary_mask must be a NumPy array")
    if binary_mask.ndim != 2:
        raise ValueError("binary_mask must be a non-empty 2D array")
    if binary_mask.shape[0] == 0 or binary_mask.shape[1] == 0:
        raise ValueError("binary_mask must be a non-empty 2D array")
    if binary_mask.dtype not in (np.dtype(np.bool_), np.dtype(np.uint8)):
        raise TypeError("binary_mask dtype must be bool or uint8")

    if isinstance(feather_radius, (bool, np.bool_)) or not isinstance(
        feather_radius, (int, np.integer)
    ):
        raise TypeError("feather_radius must be a non-negative integer")
    if feather_radius < 0:
        raise ValueError("feather_radius must be non-negative")

    if binary_mask.dtype == np.dtype(np.bool_):
        mask_f32 = binary_mask.astype(np.float32, copy=True)
    else:
        values = np.unique(binary_mask)
        if not np.all(np.isin(values, (0, 1))) and not np.all(np.isin(values, (0, 255))):
            raise ValueError("uint8 binary_mask values must be all in {0, 1} or {0, 255}")
        divisor = 255.0 if np.any(values == 255) else 1.0
        mask_f32 = binary_mask.astype(np.float32, copy=True) / divisor

    # Avoid introducing float32 rounding into mathematically constant masks.
    if not np.any(mask_f32) or np.all(mask_f32 == 1.0):
        return mask_f32
    if feather_radius in (0, 1):
        return mask_f32

    # OpenCV requires an odd, positive kernel size. The public parameter keeps
    # its historical name, but its value controls the kernel size by contract.
    kernel_size = int(feather_radius)
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = kernel_size / 3.0
    soft_mask = cv2.GaussianBlur(
        mask_f32,
        (kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )

    return np.clip(soft_mask, 0.0, 1.0).astype(np.float32, copy=False)


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
