"""
Bộ điều phối và thực thi công cụ (Tool Dispatcher & Executor).
Kết nối kế hoạch điều trị từ Agent với Module 2 (Region Engine) và Module 3 (Processing Engine).
"""

import inspect

import numpy as np

from src.processing_engine.color import apply_color_balance
from src.processing_engine.denoise import apply_denoise
from src.processing_engine.exposure_contrast import apply_clahe, apply_gamma
from src.processing_engine.sharpen import apply_sharpen
from src.region_engine.controller import RegionRequest
from src.region_engine.controller import resolve_region as resolve_module_region
from src.region_engine.detector import segment_by_prompt
from src.region_engine.face_detector import detect_faces

from .state import TreatmentPlan

_FACE_TARGETS = frozenset(("face", "faces", "khuôn mặt", "khuôn mặt người"))
_FULL_TARGETS = frozenset(("full", "all", "toàn", "toàn bộ", "full_image"))
_SPATIAL_TARGETS = frozenset(("top", "bottom", "left", "right", "center", "giữa"))


def _request_for_action(action) -> RegionRequest:
    target = action.target_prompt.strip().casefold()
    kind = action.region_type
    if kind == "semantic" and target in _FACE_TARGETS:
        kind = "face"
    elif kind == "semantic" and target in _FULL_TARGETS:
        kind = "full"
    elif kind == "semantic" and target in _SPATIAL_TARGETS:
        kind = "spatial"

    return RegionRequest(
        kind=kind,
        bbox=action.bbox,
        quadrant=action.quadrant or (action.target_prompt if kind == "spatial" else None),
        prompt=action.target_prompt if kind == "semantic" else None,
        binary_mask=action.binary_mask if kind == "binary_mask" else None,
        feather_radius=action.feather_radius,
        expand_ratio=action.expand_ratio,
        merge_policy=action.merge_policy,
    )


def _resolve_action_region(image: np.ndarray, action):
    """Resolve an action through Module 2 while keeping test seams injectable."""
    request = _request_for_action(action)
    face_resolver = detect_faces
    semantic_resolver = segment_by_prompt
    if request.kind == "face":
        parameters = inspect.signature(face_resolver).parameters
        if "feather_radius" not in parameters and not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            def face_resolver_without_options(current_image, **_kwargs):
                return detect_faces(current_image)

            face_resolver = face_resolver_without_options
    if request.kind == "semantic":
        parameters = inspect.signature(semantic_resolver).parameters
        if "feather_radius" not in parameters and not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            def semantic_resolver_without_options(current_image, prompt, **_kwargs):
                return segment_by_prompt(current_image, prompt)

            semantic_resolver = semantic_resolver_without_options
    return resolve_module_region(
        image,
        request,
        _face_resolver=face_resolver,
        _semantic_resolver=semantic_resolver,
    )


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
        # 1. Tạo mặt nạ vùng thông qua Module 2 controller.
        region = _resolve_action_region(current_img, action)
        if region.is_empty:
            # Empty là kết quả hợp lệ của detector/segmenter; không truyền None
            # để Module 3 xử lý toàn ảnh.
            continue
        # Giữ tối ưu tương thích legacy cho full: Module 3 hiểu None là full
        # image. Controller vẫn luôn trả về mask ones cho callers trực tiếp.
        mask = None if region.metadata.get("kind") == "full" else region.mask

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
