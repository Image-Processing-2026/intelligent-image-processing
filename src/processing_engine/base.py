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


def _mask_crop_box(mask: np.ndarray, halo: int) -> Optional[tuple[int, int, int, int]]:
    """
    Hộp bao (y0, y1, x0, x1) của vùng mask > 0, nới rộng thêm `halo` pixel mỗi phía.

    Trả về None nếu mask rỗng (toàn 0) — khi đó kết quả hòa trộn chính là ảnh gốc.
    """
    support = mask > 0.0
    rows = np.flatnonzero(support.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(support.any(axis=0))
    h, w = mask.shape
    return (
        max(0, int(rows[0]) - halo),
        min(h, int(rows[-1]) + 1 + halo),
        max(0, int(cols[0]) - halo),
        min(w, int(cols[-1]) + 1 + halo),
    )


def apply_region_op(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    op_func: Callable[..., np.ndarray],
    *,
    crop_halo: Optional[int] = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    Hàm bọc (wrapper) tiêu chuẩn áp dụng thuật toán xử lý ảnh lên vùng chỉ định.

    Quy trình:
        1. Kiểm tra ảnh và mặt nạ theo hợp đồng (raise ProcessingInputError nếu sai).
        2. Áp dụng thuật toán lên toàn bộ ảnh, hoặc chỉ lên hộp bao của mask khi
           `crop_halo` được khai báo (xem dưới).
        3. Hòa trộn alpha: I_out = M · I_processed + (1 − M) · I_original.

    Cắt theo mask (crop-to-mask): ngoài vùng M > 0, kết quả hòa trộn luôn là ảnh gốc, nên chỉ
    cần chạy thuật toán trên hộp bao của M > 0. Bộ lọc lân cận cần thêm `crop_halo` pixel quanh
    hộp (bán kính ảnh hưởng của kernel) để điểm ảnh trong vùng thấy đủ lân cận như khi chạy toàn
    ảnh; artifact biên của ảnh cắt chỉ nằm trong dải halo, nơi M = 0 nên bị loại khi hòa trộn.
    Chỉ khai báo `crop_halo` cho thuật toán đã kiểm chứng cho kết quả GIỐNG HỆT TỪNG BYTE với
    chạy toàn ảnh (tests/unit/test_crop_to_mask.py). Để None (mặc định) khi thuật toán dùng thống
    kê toàn ảnh (CLAHE, white balance) hoặc khi OpenCV làm tròn phụ thuộc vị trí điểm ảnh
    (đường SIMD số thực của bilateral, filter2D, HSV2RGB) — khi đó chạy toàn ảnh như cũ.

    Args:
        image: Ảnh gốc đầu vào (uint8; RGB (H, W, 3) hoặc xám (H, W) / (H, W, 1)).
        mask: Mặt nạ mềm (float32 [0.0, 1.0]), None nếu áp dụng cho toàn bộ ảnh.
        op_func: Hàm xử lý ảnh thuần túy nhận vào `image` và các tham số `kwargs`.
        crop_halo: Bán kính ảnh hưởng (pixel) của thuật toán nếu cho phép cắt theo mask;
            0 cho phép toán từng điểm ảnh; None để luôn xử lý toàn ảnh.
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
        {k: f"<array {v.shape}>" if isinstance(v, np.ndarray) else v for k, v in kwargs.items()},
    )

    if mask is not None and crop_halo is not None:
        mask_2d = mask[..., 0] if mask.ndim == 3 else mask
        box = _mask_crop_box(mask_2d, crop_halo)
        if box is None:
            # Mask toàn 0: hòa trộn trả về ảnh gốc, không cần chạy thuật toán
            result = work.copy()
        elif box != (0, work.shape[0], 0, work.shape[1]):
            y0, y1, x0, x1 = box
            logger.debug("%s cropped to mask box y[%d:%d] x[%d:%d]", op_name, y0, y1, x0, x1)
            region = np.ascontiguousarray(work[y0:y1, x0:x1])
            processed = _run_op(op_func, region, op_name, kwargs)
            result = work.copy()
            result[y0:y1, x0:x1] = blend_regions(
                region, processed, np.ascontiguousarray(mask_2d[y0:y1, x0:x1])
            )
        else:
            result = blend_regions(work, _run_op(op_func, work, op_name, kwargs), mask)
        return result[..., None] if restore_channel_axis else result

    # 1. Áp dụng thuật toán xử lý ảnh lên toàn bộ khung hình
    processed = _run_op(op_func, work, op_name, kwargs)

    # 2. Nếu có mặt nạ, hòa trộn kết quả bằng Alpha Compositing; nếu không, dùng kết quả toàn cục
    result = processed if mask is None else blend_regions(work, processed, mask)

    return result[..., None] if restore_channel_axis else result


def _run_op(
    op_func: Callable[..., np.ndarray], work: np.ndarray, op_name: str, kwargs: dict
) -> np.ndarray:
    """Chạy thuật toán thuần túy và kiểm tra nó giữ nguyên shape/dtype."""
    processed = op_func(work, **kwargs)
    if processed.shape != work.shape or processed.dtype != np.uint8:
        raise RuntimeError(
            f"{op_name} produced {processed.shape}/{processed.dtype}, expected {work.shape}/uint8"
        )
    return processed
