"""
Bộ điều phối và thực thi công cụ (Tool Dispatcher & Executor).
Kết nối kế hoạch điều trị từ Agent với Module 2 (Region Engine) và Module 3 (Processing Engine).
"""

import numpy as np

from src.processing_engine.color import apply_color_balance
from src.processing_engine.denoise import apply_denoise
from src.processing_engine.exposure_contrast import apply_clahe, apply_gamma
from src.processing_engine.sharpen import apply_sharpen
from src.region_engine.detector import segment_by_prompt
from src.region_engine.face_detector import detect_faces

from .state import TreatmentPlan


def execute_plan(image: np.ndarray, plan: TreatmentPlan) -> np.ndarray:
    """
    Thực thi tuần tự các hành động trong kế hoạch điều trị trên ảnh.

    Args:
        image: Ảnh đầu vào ở vòng lặp hiện tại (RGB, uint8).
        plan: Kế hoạch điều trị đã được chuẩn hóa.

    Returns:
        np.ndarray: Ảnh sau khi đã áp dụng toàn bộ các thao tác.
    """
    current_img = image.copy()

    for action in plan.actions:
        # 1. Tạo mặt nạ vùng thông qua Module 2
        mask = None
        target = action.target_prompt.lower().strip()

        if target in ["face", "khuôn mặt"]:
            face_masks = detect_faces(current_img)
            if face_masks:
                mask = face_masks[0]
        elif target not in ["full", "all", "toàn bộ", "full_image"]:
            mask = segment_by_prompt(current_img, action.target_prompt)

        # 2. Áp dụng thao tác xử lý ảnh tương ứng từ Module 3
        op = action.operation.lower().strip()
        params = action.parameters

        if op == "denoise":
            current_img = apply_denoise(
                current_img,
                mask=mask,
                method=params.get("method", "bilateral"),
                strength=float(params.get("strength", 1.0)),
            )
        elif op == "gamma_correct":
            current_img = apply_gamma(current_img, mask=mask, gamma=float(params.get("gamma", 1.2)))
        elif op == "clahe":
            current_img = apply_clahe(
                current_img, mask=mask, clip_limit=float(params.get("clip_limit", 2.0))
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

    return current_img
