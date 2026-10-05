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


def _validate_image(image: object, name: str) -> np.ndarray:
    """Validate one inter-module RGB image and return it unchanged."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if image.dtype != np.dtype(np.uint8):
        raise TypeError(f"{name} dtype must be uint8")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must have shape (H, W, 3)")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty")
    return image


def blend_regions(
    original_image: np.ndarray,
    processed_image: np.ndarray,
    soft_mask: np.ndarray | None,
) -> np.ndarray:
    """Composite a processed RGB image over its original using a spatial mask.

    The operation is source-over compositing with an opaque original background:

    ``output = mask * processed + (1 - mask) * original``

    Images must be non-empty RGB ``uint8`` arrays with identical shapes.  The
    mask must be a ``float32`` array of shape ``(H, W)`` or ``(H, W, 1)`` and
    every value must be finite and in ``[0, 1]``.  A mask of ``None`` means
    that the complete processed image is selected, but a copy is returned so
    callers never receive an input alias.

    Values are calculated in ``float32``, clipped to the byte range, and
    truncated when converted to ``uint8``.  No resizing, clipping, implicit
    dtype conversion, channel reordering, or mask blurring is performed.

    Raises:
        TypeError: If an image/mask is not an ndarray or has an unsupported
            dtype.
        ValueError: If an image/mask has an invalid shape, mismatched spatial
            dimensions, or the mask contains a non-finite/out-of-range value.
    """
    original = _validate_image(original_image, "original_image")
    processed = _validate_image(processed_image, "processed_image")
    if original.shape != processed.shape:
        raise ValueError("original_image and processed_image must have the same shape")

    if soft_mask is None:
        return processed.copy()

    if not isinstance(soft_mask, np.ndarray):
        raise TypeError("soft_mask must be a NumPy array or None")
    if soft_mask.dtype != np.dtype(np.float32):
        raise TypeError("soft_mask dtype must be float32")
    if soft_mask.ndim == 2:
        if soft_mask.shape != original.shape[:2]:
            raise ValueError("soft_mask spatial shape must match the input images")
        mask = soft_mask[..., None]
    elif soft_mask.ndim == 3:
        if soft_mask.shape != (*original.shape[:2], 1):
            raise ValueError("soft_mask must have shape (H, W) or (H, W, 1)")
        mask = soft_mask
    else:
        raise ValueError("soft_mask must have shape (H, W) or (H, W, 1)")

    if not np.isfinite(soft_mask).all():
        raise ValueError("soft_mask must contain only finite values")
    if np.any(soft_mask < 0.0) or np.any(soft_mask > 1.0):
        raise ValueError("soft_mask values must be in the range [0, 1]")

    original_f = original.astype(np.float32)
    processed_f = processed.astype(np.float32)
    blended = mask * processed_f + (1.0 - mask) * original_f

    # These are semantic identity points, not approximate numerical results.
    # Restoring them also protects byte-preservation from float32 cancellation.
    np.copyto(blended, original_f, where=mask == 0.0)
    np.copyto(blended, processed_f, where=mask == 1.0)
    np.copyto(blended, original_f, where=original == processed)

    return np.clip(blended, 0.0, 255.0).astype(np.uint8)
