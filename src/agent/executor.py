"""
Bộ điều phối và thực thi công cụ (Tool Dispatcher & Executor).
Kết nối kế hoạch điều trị từ Agent với Module 2 (Region Engine) và Module 3 (Processing Engine).
"""

import logging
from typing import Any, Optional

import numpy as np

from src.processing_engine.color import apply_color_balance
from src.processing_engine.denoise import apply_denoise
from src.processing_engine.exposure_contrast import apply_clahe, apply_gamma
from src.processing_engine.sharpen import apply_sharpen
from src.region_engine.detector import segment_by_prompt
from src.region_engine.face_detector import detect_faces

from .state import TreatmentPlan

logger = logging.getLogger(__name__)


def _validate_mask(mask: Any, image_shape: tuple) -> bool:
    """
    Kiểm tra mask có hợp lệ để sử dụng hay không.
    - mask is not None và là np.ndarray
    - mask.ndim >= 2 và kích thước spatial khớp với image
    - mask.dtype == np.float32
    - Không chứa NaN/Inf và không phải toàn bộ là 0
    - Giá trị nằm trong khoảng [0.0, 1.0]
    """
    if mask is None:
        return False
    if not isinstance(mask, np.ndarray):
        return False
    if mask.ndim < 2 or mask.shape[:2] != image_shape[:2]:
        return False
    if mask.dtype != np.float32:
        return False
    if np.all(mask == 0):
        return False
    if np.any(np.isnan(mask)) or np.any(np.isinf(mask)):
        return False
    if np.any(mask < 0.0) or np.any(mask > 1.0):
        return False
    return True


def execute_plan(image: np.ndarray, plan: TreatmentPlan) -> np.ndarray:
    """
    Thực thi tuần tự các hành động trong kế hoạch điều trị trên ảnh.
    Mỗi action được bọc riêng trong try...except — một action lỗi không làm sập cả pipeline.

    Args:
        image: Ảnh đầu vào ở vòng lặp hiện tại (RGB, uint8).
        plan: Kế hoạch điều trị đã được chuẩn hóa.

    Returns:
        np.ndarray: Ảnh sau khi đã áp dụng toàn bộ các thao tác (RGB, uint8).
    """
    current_img = np.clip(image, 0, 255).astype(np.uint8)

    if not plan or not plan.actions:
        return current_img

    for action in plan.actions:
        try:
            # 1. Tạo mặt nạ vùng thông qua Module 2
            mask: Optional[np.ndarray] = None
            target = action.target_prompt.lower().strip()

            if target in ["face", "khuôn mặt"]:
                face_masks = detect_faces(current_img)
                if face_masks:
                    mask = face_masks[0]
                else:
                    logger.warning(
                        "No face detected for action '%s' on region '%s'. Skipping action.",
                        action.operation,
                        action.region_id,
                    )
                    continue
            elif target not in ["full", "all", "toàn bộ", "full_image"]:
                mask = segment_by_prompt(current_img, action.target_prompt)

            # Validate mask
            if mask is not None and not _validate_mask(mask, current_img.shape):
                logger.warning(
                    "Invalid mask for action '%s' on region '%s'. Fallback to full-image processing (mask=None).",
                    action.operation,
                    action.region_id,
                )
                mask = None

            # 2. Áp dụng thao tác xử lý ảnh tương ứng từ Module 3
            op = action.operation.lower().strip()
            params = action.parameters or {}

            if op == "denoise":
                current_img = apply_denoise(
                    current_img,
                    mask=mask,
                    method=params.get("method", "bilateral"),
                    strength=float(params.get("strength", 1.0)),
                )
            elif op == "gamma_correct":
                current_img = apply_gamma(
                    current_img,
                    mask=mask,
                    gamma=float(params.get("gamma", 1.2)),
                )
            elif op == "clahe":
                current_img = apply_clahe(
                    current_img,
                    mask=mask,
                    clip_limit=float(params.get("clip_limit", 2.0)),
                )
            elif op == "sharpen":
                current_img = apply_sharpen(
                    current_img,
                    mask=mask,
                    method=params.get("method", "unsharp_mask"),
                    amount=float(params.get("amount", 1.0)),
                )
            elif op == "color_correct":
                current_img = apply_color_balance(
                    current_img,
                    mask=mask,
                    saturation_scale=float(params.get("saturation_scale", 1.0)),
                    temperature_shift=float(params.get("temperature_shift", 0.0)),
                )
            else:
                logger.warning("Unsupported operation '%s'. Skipping action.", op)
                continue

            # 3. Đảm bảo output luôn là np.uint8 [0, 255]
            current_img = np.clip(current_img, 0, 255).astype(np.uint8)

        except Exception as e:
            logger.warning(
                "Error executing action '%s' on region '%s': %s. Skipping this action.",
                action.operation,
                action.region_id,
                str(e),
                exc_info=True,
            )
            # Giữ nguyên current_img, tiếp tục action tiếp theo

    return current_img
