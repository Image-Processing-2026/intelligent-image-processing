"""
Khung cơ bản và bộ hòa trộn vùng xử lý (Base Region Operation Wrapper).
Đảm bảo mọi thao tác xử lý ảnh đều tự động hỗ trợ mặt nạ vùng (soft-mask)
và kiểm tra hợp đồng dữ liệu đầu vào trước khi chạy thuật toán.
"""

import logging
import math
from typing import Any, Callable, Iterable, Optional

import numpy as np

from src.region_engine.mask_utils import blend_regions

logger = logging.getLogger(__name__)


class ProcessingInputError(ValueError):
    """Lỗi đầu vào của Module 3: ảnh, mặt nạ hoặc tham số vi phạm hợp đồng giao tiếp."""


def validate_image(image: Any, op_name: str) -> np.ndarray:
    """
    Kiểm tra ảnh đầu vào theo hợp đồng liên module (docs/interfaces.md).

    Chấp nhận: uint8 có shape (H, W, 3) RGB, (H, W) hoặc (H, W, 1) grayscale.
    Từ chối rõ ràng (không tự chuyển đổi): ảnh float, RGBA 4 kênh, ảnh rỗng.
    Việc chuẩn hóa ảnh (bỏ alpha, đổi kiểu) là trách nhiệm của tầng API/Agent.
    """
    if not isinstance(image, np.ndarray):
        raise ProcessingInputError(
            f"{op_name}: image must be a NumPy array, got {type(image).__name__}"
        )
    if image.dtype != np.uint8:
        raise ProcessingInputError(
            f"{op_name}: image dtype must be uint8 in [0, 255], got {image.dtype}; "
            "convert float images with (img * 255).round().astype(np.uint8) before calling"
        )
    if not (image.ndim == 2 or (image.ndim == 3 and image.shape[2] in (1, 3))):
        raise ProcessingInputError(
            f"{op_name}: image must have shape (H, W, 3), (H, W, 1) or (H, W), got {image.shape}; "
            "RGBA images must drop the alpha channel before processing"
        )
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ProcessingInputError(f"{op_name}: image must be non-empty, got {image.shape}")
    return image


def validate_mask(mask: Any, image_shape: tuple, op_name: str) -> Optional[np.ndarray]:
    """
    Kiểm tra mặt nạ mềm: float32, shape (H, W) hoặc (H, W, 1) khớp ảnh, giá trị hữu hạn trong [0, 1].

    Kiểm tra sớm (fail-fast) trước khi chạy bộ lọc tốn thời gian như NLM,
    thay vì chỉ phát hiện lỗi ở bước blend_regions sau cùng.
    """
    if mask is None:
        return None
    if not isinstance(mask, np.ndarray):
        raise ProcessingInputError(f"{op_name}: mask must be a NumPy array or None")
    if mask.dtype != np.float32:
        raise ProcessingInputError(f"{op_name}: mask dtype must be float32, got {mask.dtype}")
    spatial = tuple(image_shape[:2])
    if not (mask.shape == spatial or mask.shape == (*spatial, 1)):
        raise ProcessingInputError(
            f"{op_name}: mask shape {mask.shape} does not match image spatial shape {spatial}"
        )
    if not np.isfinite(mask).all():
        raise ProcessingInputError(f"{op_name}: mask must contain only finite values")
    if mask.min() < 0.0 or mask.max() > 1.0:
        raise ProcessingInputError(f"{op_name}: mask values must be in [0, 1]")
    return mask


def require_choice(value: Any, allowed: Iterable[str], name: str, op_name: str) -> str:
    """Kiểm tra tham số dạng liệt kê (enum); không âm thầm trả về ảnh gốc khi sai."""
    allowed = tuple(allowed)
    if value not in allowed:
        raise ProcessingInputError(f"{op_name}: {name} must be one of {allowed}, got {value!r}")
    return value


def require_number(
    value: Any,
    name: str,
    op_name: str,
    *,
    gt: float | None = None,
    ge: float | None = None,
    le: float | None = None,
) -> float:
    """
    Kiểm tra tham số số thực hữu hạn trong miền xác định toán học của thuật toán.

    Lưu ý: đây là miền hợp lệ về toán học (VD: gamma > 0), KHÔNG phải khoảng an toàn
    của Agent — việc kẹp tham số về khoảng an toàn do planner (Module 4) đảm nhiệm.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ProcessingInputError(f"{op_name}: {name} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ProcessingInputError(f"{op_name}: {name} must be finite, got {value}")
    if gt is not None and not value > gt:
        raise ProcessingInputError(f"{op_name}: {name} must be > {gt}, got {value}")
    if ge is not None and not value >= ge:
        raise ProcessingInputError(f"{op_name}: {name} must be >= {ge}, got {value}")
    if le is not None and not value <= le:
        raise ProcessingInputError(f"{op_name}: {name} must be <= {le}, got {value}")
    return value


def apply_region_op(
    image: np.ndarray, mask: Optional[np.ndarray], op_func: Callable[..., np.ndarray], **kwargs: Any
) -> np.ndarray:
    """
    Hàm bọc (wrapper) tiêu chuẩn áp dụng thuật toán xử lý ảnh lên vùng chỉ định.

    Quy trình:
        1. Kiểm tra ảnh và mặt nạ theo hợp đồng (raise ProcessingInputError nếu sai).
        2. Áp dụng thuật toán lên TOÀN BỘ ảnh — bộ lọc tích chập cần điểm ảnh lân cận,
           cắt vùng trước khi lọc sẽ gây artifact ở biên.
        3. Hòa trộn alpha: I_out = M · I_processed + (1 − M) · I_original.

    Args:
        image: Ảnh gốc đầu vào (uint8; RGB (H, W, 3) hoặc xám (H, W) / (H, W, 1)).
        mask: Mặt nạ mềm (float32 [0.0, 1.0]), None nếu áp dụng cho toàn bộ ảnh.
        op_func: Hàm xử lý ảnh thuần túy nhận vào `image` và các tham số `kwargs`.
        kwargs: Các siêu tham số của thuật toán xử lý ảnh.

    Returns:
        np.ndarray: Ảnh uint8 cùng shape với đầu vào, đã xử lý và hòa trộn mượt mà.

    Raises:
        ProcessingInputError: Ảnh hoặc mặt nạ vi phạm hợp đồng dữ liệu.
    """
    op_name = getattr(op_func, "__name__", "operation").lstrip("_").replace("raw_", "apply_")
    validate_image(image, op_name)
    validate_mask(mask, image.shape, op_name)

    # Ảnh xám (H, W, 1): OpenCV thường trả về (H, W) nên xử lý trên mảng 2D rồi khôi phục trục kênh
    restore_channel_axis = image.ndim == 3 and image.shape[2] == 1
    work = image[..., 0] if restore_channel_axis else image

    logger.debug(
        "Running %s on image %s (mask=%s) with params %s",
        op_name,
        image.shape,
        "none" if mask is None else "soft",
        kwargs,
    )

    # 1. Áp dụng thuật toán xử lý ảnh lên toàn bộ khung hình
    processed = op_func(work, **kwargs)
    if processed.shape != work.shape or processed.dtype != np.uint8:
        raise RuntimeError(
            f"{op_name} produced {processed.shape}/{processed.dtype}, expected {work.shape}/uint8"
        )

    # 2. Nếu có mặt nạ, hòa trộn kết quả bằng Alpha Compositing; nếu không, dùng kết quả toàn cục
    result = processed if mask is None else blend_regions(work, processed, mask)

    return result[..., None] if restore_channel_axis else result
