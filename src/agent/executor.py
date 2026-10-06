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
from src.region_engine.controller import (
    InvalidRegionRequestError,
    RegionBackendUnavailableError,
    RegionInferenceError,
    RegionRequest,
)
from src.region_engine.controller import resolve_region as resolve_module_region
from src.region_engine.face_detector import detect_faces

from .regions import FACE_TARGETS, FULL_TARGETS, SPATIAL_TARGETS
from .state import RegionOperation, TreatmentPlan

logger = logging.getLogger(__name__)


# Heuristic dự phòng (quyết định D1): chỉ dùng khi backend ngữ nghĩa không khả dụng
# hoặc suy luận lỗi. Từ khóa không khớp → bỏ qua action, KHÔNG xử lý toàn ảnh.
_HEURISTIC_QUADRANTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("sky", "trời"), "top"),
    (("ground", "floor", "đất"), "bottom"),
    (("center", "centre", "giữa"), "center"),
)


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


def _infer_region_kind(action: RegionOperation) -> str:
    """
    Xác định loại vùng cho Module 2 (quyết định D2).
    region_type=None → suy ra từ target_prompt; 'semantic' với từ khóa đặc biệt
    (face/full/quadrant) được chuyển sang bộ phân giải rẻ hơn tương ứng.
    """
    target = action.target_prompt.strip().casefold()
    kind = action.region_type
    if kind is None or kind == "semantic":
        if target in FACE_TARGETS:
            return "face"
        if target in FULL_TARGETS:
            return "full"
        if target in SPATIAL_TARGETS:
            return "spatial"
        return "semantic"
    return kind


def _request_for_action(action: RegionOperation, kind: str) -> RegionRequest:
    """Chuyển một action của kế hoạch thành RegionRequest chuẩn của Module 2."""
    return RegionRequest(
        kind=kind,
        bbox=action.bbox,
        quadrant=action.quadrant or (action.target_prompt if kind == "spatial" else None),
        prompt=action.target_prompt if kind == "semantic" else None,
        binary_mask=action.binary_mask if kind == "binary_mask" else None,
        feather_radius=action.feather_radius,
        expand_ratio=action.expand_ratio,
        merge_policy=action.merge_policy,
        face_mode=action.face_mode,
        num_faces=action.num_faces,
        instance_selection=action.instance_selection,
        instance_index=action.instance_index,
        box_threshold=action.box_threshold,
        text_threshold=action.text_threshold,
        nms_iou_threshold=action.nms_iou_threshold,
    )


def _heuristic_quadrant(prompt: str) -> Optional[str]:
    """Ánh xạ từ khóa không gian quen thuộc sang góc phần tư; None nếu không khớp."""
    lowered = prompt.casefold()
    for keywords, quadrant in _HEURISTIC_QUADRANTS:
        if any(keyword in lowered for keyword in keywords):
            return quadrant
    return None


def _as_rgb_for_detection(image: np.ndarray) -> np.ndarray:
    """Module 2 chỉ nhận RGB (H, W, 3); ảnh xám được nhân bản kênh chỉ để dò vùng."""
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    return image


def _resolve_action_region(
    image: np.ndarray, action: RegionOperation
) -> tuple[bool, Optional[np.ndarray], Optional[str]]:
    """
    Tạo mặt nạ vùng cho một action thông qua Module 2 controller.

    Returns:
        (proceed, mask, backend): proceed=False → bỏ qua action; mask=None → xử lý toàn ảnh;
        backend là bộ phân giải đã tạo mask (ví dụ 'heuristic' theo D1).
    """
    kind = _infer_region_kind(action)
    if kind == "full":
        return True, None, "geometry"

    detection_image = _as_rgb_for_detection(image)
    request = _request_for_action(action, kind)
    try:
        # Resolver khuôn mặt truyền qua namespace của executor để test có thể patch.
        # Không truyền resolver ngữ nghĩa: controller tự gọi resolve_prompt_instances
        # để instance_selection/instance_index có hiệu lực (test patch hàm đó trong controller).
        region = resolve_module_region(detection_image, request, _face_resolver=detect_faces)
    except (RegionBackendUnavailableError, RegionInferenceError) as exc:
        # Chế độ khuôn mặt tinh chỉnh (oval cần model Face Landmarker) không khả dụng
        # → hạ về bbox thay vì bỏ cả action (cùng tinh thần quyết định D1)
        if kind == "face" and action.face_mode != "bbox":
            logger.warning(
                "Face mode '%s' unavailable for region '%s' (%s). Falling back to 'bbox'.",
                action.face_mode,
                action.region_id,
                exc,
            )
            return _resolve_action_region(image, action.model_copy(update={"face_mode": "bbox"}))
        quadrant = _heuristic_quadrant(action.target_prompt) if kind == "semantic" else None
        if quadrant is None:
            logger.warning(
                "Region '%s' (%s) could not be resolved (%s). Skipping action '%s'.",
                action.region_id,
                kind,
                exc,
                action.operation,
            )
            return False, None, None
        logger.warning(
            "Semantic backend failed for region '%s' (%s). Using heuristic '%s' quadrant.",
            action.region_id,
            exc,
            quadrant,
        )
        region = resolve_module_region(
            detection_image,
            RegionRequest(kind="spatial", quadrant=quadrant, feather_radius=action.feather_radius),
        )
        # Ghi nhận nguồn gốc mask là heuristic (quyết định D1), không phải backend ngữ nghĩa.
        region.metadata["backend"] = "heuristic"
    except InvalidRegionRequestError as exc:
        logger.warning(
            "Invalid region request for '%s': %s. Skipping action '%s'.",
            action.region_id,
            exc,
            action.operation,
        )
        return False, None, None

    if region.is_empty or not _validate_mask(region.mask, image.shape):
        logger.warning(
            "No usable mask for region '%s' (%s). Skipping action '%s'.",
            action.region_id,
            kind,
            action.operation,
        )
        return False, None, None

    backend = str(region.metadata.get("backend", "unknown"))
    logger.info("Region '%s' (%s) resolved by backend '%s'.", action.region_id, kind, backend)
    # Module 3 hiểu mask=None là toàn ảnh; giữ tối ưu này cho vùng 'full'.
    if region.metadata.get("kind") == "full":
        return True, None, backend
    return True, region.mask, backend


def _resolve_action_mask(
    image: np.ndarray, action: RegionOperation
) -> tuple[bool, Optional[np.ndarray]]:
    """(proceed, mask) của _resolve_action_region, dùng khi thực thi kế hoạch."""
    proceed, mask, _ = _resolve_action_region(image, action)
    return proceed, mask


def resolve_region_mask(
    image: np.ndarray, target_prompt: str, feather_radius: int = 15
) -> tuple[bool, Optional[np.ndarray], Optional[str]]:
    """
    Tạo mặt nạ cho một vùng theo tên (dùng để đo số liệu theo vùng ở giai đoạn Perceive).
    Đi cùng đường phân giải với execute_plan (face/full/quadrant/semantic, heuristic D1).

    Returns:
        (found, mask, backend): found=False → không xác định được vùng; mask=None → toàn ảnh.
    """
    probe = RegionOperation(
        region_id=target_prompt,
        target_prompt=target_prompt,
        feather_radius=feather_radius,
        # Hai trường bắt buộc của RegionOperation; chỉ xuất hiện trong log, không thực thi
        detected_issue="region_measurement",
        operation="measure",
    )
    return _resolve_action_region(image, probe)


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
            proceed, mask = _resolve_action_mask(current_img, action)
            if not proceed:
                continue

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
