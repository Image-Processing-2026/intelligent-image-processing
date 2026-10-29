"""
Bộ điều phối và thực thi công cụ (Tool Dispatcher & Executor).
Kết nối kế hoạch điều trị từ Agent với Module 2 (Region Engine) và Module 3 (Processing Engine).
"""

import hashlib
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence

import cv2
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
from src.region_engine.segmentation_backend import (
    SegmentationInferenceError,
    SegmentationUnavailableError,
)
from src.region_engine.subject import capabilities as subject_capabilities
from src.region_engine.subject import resolve_subject_mask, segment_box

from .regions import FACE_TARGETS, FULL_TARGETS, SPATIAL_TARGETS, canonical_region
from .state import ExecutedAction, RegionOperation, TreatmentPlan

logger = logging.getLogger(__name__)


# Heuristic dự phòng (quyết định D1): chỉ dùng khi backend ngữ nghĩa không khả dụng
# hoặc suy luận lỗi. Từ khóa không khớp → bỏ qua action, KHÔNG xử lý toàn ảnh.
_HEURISTIC_QUADRANTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("sky", "trời"), "top"),
    (("ground", "floor", "đất"), "bottom"),
    (("center", "centre", "giữa"), "center"),
)


# Thao tác chỉnh phơi sáng: trên vùng mặt mặc định dùng mask oval bám đường viền (R-04)
EXPOSURE_OPERATIONS: frozenset[str] = frozenset({"gamma_correct", "clahe"})

# Tạm thời, chưa hiệu chỉnh (BL-10): hệ số cường độ khi mask mặt phải rơi về bbox
FALLBACK_STRENGTH_SCALE = 0.5
# Tạm thời, chưa hiệu chỉnh (BL-10): ngưỡng QA mask mặt so với hộp dò tìm
FACE_QA_BOX_MARGIN = 0.05  # nới hộp dò tìm theo tỉ lệ cạnh khi xét "nằm trong hộp"
FACE_QA_MIN_INSIDE = 0.9  # tỉ lệ khối lượng mask phải nằm trong hộp (đã nới)
FACE_QA_MIN_COVERAGE = 0.1  # diện tích mask cứng / diện tích hộp, thấp hơn là quá nhỏ
FACE_QA_MAX_COVERAGE = 1.3  # cao hơn là tràn ra ngoài hộp
FACE_QA_MAX_COMPONENTS_PER_FACE = 2  # số thành phần liên thông tối đa cho mỗi khuôn mặt
FACE_QA_MIN_COMPONENT_SHARE = 0.05  # thành phần nhỏ hơn tỉ lệ này của mask bị coi là nhiễu

# Mask đã thực thi, tra theo mask_hash để overlay vẽ đúng mask đã dùng (R-04)
_EXECUTED_MASK_MAX_ENTRIES = 32
_EXECUTED_MASKS: "OrderedDict[str, np.ndarray]" = OrderedDict()
_EXECUTED_MASKS_LOCK = threading.Lock()


# bbox Gemini (ymin, xmin, ymax, xmax) 0..1000 theo tên subject, do Perceive trả về
SubjectBoxes = Mapping[str, Sequence[int]]


def image_cache_key(image: np.ndarray) -> str:
    """Khóa cache mask theo nội dung ảnh gốc của lượt chạy (blake2b trên shape, dtype, bytes)."""
    array = np.ascontiguousarray(image)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(f"{array.shape}|{array.dtype}".encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def subject_cache_key(base: str, shape: Sequence[int]) -> str:
    """
    Khóa cache của region_engine.resolve_subject_mask cho một kích thước ảnh: cùng ảnh gốc
    nhưng ảnh preview (phiên bản phong cách) có kích thước khác thì không được trúng cache.
    """
    return f"{base}:{int(shape[0])}x{int(shape[1])}"


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


@dataclass
class _Resolution:
    """Kết quả giải vùng của một action cùng dấu vết để ghi ExecutedAction."""

    proceed: bool
    mask: Optional[np.ndarray] = None  # None + proceed → xử lý toàn ảnh
    backend: str = ""
    requested_mode: Optional[str] = None
    actual_mode: Optional[str] = None
    fallback_reason: Optional[str] = None
    feather_actual: Optional[int] = None
    strength_scale: float = 1.0
    skip_reason: Optional[str] = None


def mask_fingerprint(mask: Optional[np.ndarray]) -> tuple[Optional[str], float]:
    """
    (sha256, tỉ lệ diện tích) của mask float32 làm tròn 3 chữ số; mask None (toàn ảnh) → (None, 1.0).
    Làm tròn để hash không đổi theo sai số số học nhỏ.
    """
    if mask is None:
        return None, 1.0
    plane = mask[..., 0] if mask.ndim == 3 else mask
    rounded = np.ascontiguousarray(np.round(plane, 3).astype(np.float32))
    digest = hashlib.sha256()
    digest.update(f"{rounded.shape}".encode())
    digest.update(rounded.tobytes())
    return digest.hexdigest(), float(rounded.mean())


def _remember_mask(mask: np.ndarray, mask_hash: str) -> None:
    """Lưu mask đã thực thi (LRU) để tra lại theo mask_hash."""
    with _EXECUTED_MASKS_LOCK:
        _EXECUTED_MASKS[mask_hash] = mask
        _EXECUTED_MASKS.move_to_end(mask_hash)
        while len(_EXECUTED_MASKS) > _EXECUTED_MASK_MAX_ENTRIES:
            _EXECUTED_MASKS.popitem(last=False)


def executed_mask(mask_hash: Optional[str]) -> Optional[np.ndarray]:
    """Mask đã thực thi theo mask_hash; None nếu không có hoặc đã bị đẩy khỏi cache."""
    if mask_hash is None:
        return None
    with _EXECUTED_MASKS_LOCK:
        mask = _EXECUTED_MASKS.get(mask_hash)
        if mask is not None:
            _EXECUTED_MASKS.move_to_end(mask_hash)
        return mask


def _instance_boxes(region: Any) -> list[tuple[int, int, int, int]]:
    """Hộp dò tìm (x0, y0, x1, y1, nửa mở) của từng khuôn mặt, suy từ mask bbox-mode (>= 0.5)."""
    masks = list(region.instance_masks) or [region.mask]
    boxes: list[tuple[int, int, int, int]] = []
    for mask in masks:
        plane = mask[..., 0] if mask.ndim == 3 else mask
        ys, xs = np.nonzero(plane >= 0.5)
        if len(xs) == 0:
            continue
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    return boxes


def face_mask_qa(
    mask: np.ndarray, boxes: Sequence[tuple[int, int, int, int]], image_shape: tuple
) -> Optional[str]:
    """
    QA mask mặt tinh chỉnh so với hộp dò tìm: không rỗng, phần lớn nằm trong hộp, không tràn
    hay quá nhỏ so với hộp, số thành phần liên thông hợp lý. Trả None nếu đạt, ngược lại là lý do.
    """
    plane = mask[..., 0] if mask.ndim == 3 else mask
    total = float(plane.sum())
    if total <= 1e-6 or not boxes:
        return "qa_empty"
    height, width = image_shape[:2]
    inside_zone = np.zeros((height, width), dtype=bool)
    box_area = 0
    for x0, y0, x1, y1 in boxes:
        pad_x = int(round((x1 - x0) * FACE_QA_BOX_MARGIN))
        pad_y = int(round((y1 - y0) * FACE_QA_BOX_MARGIN))
        inside_zone[max(0, y0 - pad_y) : y1 + pad_y, max(0, x0 - pad_x) : x1 + pad_x] = True
        box_area += (x1 - x0) * (y1 - y0)
    inside = float(plane[inside_zone].sum()) / total
    if inside < FACE_QA_MIN_INSIDE:
        return f"qa_outside_box({inside:.2f})"
    hard = plane >= 0.5
    coverage = float(hard.sum()) / max(1, box_area)
    if coverage < FACE_QA_MIN_COVERAGE:
        return f"qa_too_small({coverage:.2f})"
    if coverage > FACE_QA_MAX_COVERAGE:
        return f"qa_too_large({coverage:.2f})"
    count, _, stats, _ = cv2.connectedComponentsWithStats(hard.astype(np.uint8), connectivity=8)
    hard_area = max(1, int(hard.sum()))
    components = sum(
        1
        for index in range(1, count)
        if stats[index, cv2.CC_STAT_AREA] >= FACE_QA_MIN_COMPONENT_SHARE * hard_area
    )
    if components > FACE_QA_MAX_COMPONENTS_PER_FACE * len(boxes):
        return f"qa_fragmented({components})"
    return None


def _sam_face_mask(
    image: np.ndarray, boxes: Sequence[tuple[int, int, int, int]], feather: int
) -> np.ndarray:
    """MobileSAM trong ROI từng khuôn mặt (hộp dò tìm), hợp các mask lại."""
    union: Optional[np.ndarray] = None
    for x0, y0, x1, y1 in boxes:
        box = (float(x0), float(y0), float(x1), float(y1))
        part = segment_box(image, box, feather_radius=feather)
        union = part if union is None else np.maximum(union, part)
    if union is None:
        raise SegmentationInferenceError("no face box for SAM")
    return union.astype(np.float32)


def _skip(action: RegionOperation, reason: str, requested: Optional[str] = None) -> _Resolution:
    return _Resolution(
        proceed=False,
        requested_mode=requested,
        feather_actual=action.feather_radius,
        skip_reason=reason,
    )


def _resolve_face_region(image: np.ndarray, action: RegionOperation) -> _Resolution:
    """
    Mask mặt cho một action (R-04). Hộp dò tìm (detection_box) lấy từ bộ dò mặt bbox, chỉ để
    tìm và QA; mask chỉnh (edit_mask) là SAM-trong-ROI (nếu có model, với thao tác phơi sáng),
    rồi oval, mỗi cái phải qua QA. Mọi cái trượt hoặc lỗi → bbox với cường độ giảm
    FALLBACK_STRENGTH_SCALE, kèm fallback_reason.
    """
    operation = action.operation.lower().strip()
    requested = action.face_mode
    # Thao tác phơi sáng: không chỉ định face_mode hoặc VLM ghi 'bbox' (giá trị mặc định trong
    # schema/prompt, không phải lựa chọn có chủ ý) → oval/SAM; hộp chữ nhật lộ viền khi nâng sáng.
    # bbox chỉ còn là đường lùi ở cường độ giảm FALLBACK_STRENGTH_SCALE.
    if operation in EXPOSURE_OPERATIONS and (
        "face_mode" not in action.model_fields_set or requested == "bbox"
    ):
        requested = "oval"

    detection_image = _as_rgb_for_detection(image)
    try:
        bbox_region = resolve_module_region(
            detection_image,
            _request_for_action(action.model_copy(update={"face_mode": "bbox"}), "face"),
            _face_resolver=detect_faces,
        )
    except (RegionBackendUnavailableError, RegionInferenceError, InvalidRegionRequestError) as exc:
        logger.warning(
            "Face region '%s' could not be resolved (%s). Skipping action '%s'.",
            action.region_id,
            exc,
            action.operation,
        )
        return _skip(action, f"face_unresolved: {exc}", requested)
    if bbox_region.is_empty or not _validate_mask(bbox_region.mask, image.shape):
        logger.warning("No face found for region '%s'. Skipping action.", action.region_id)
        return _skip(action, "no_face", requested)

    bbox_backend = str(bbox_region.metadata.get("backend", "mediapipe"))
    if requested == "bbox":
        return _Resolution(
            True, bbox_region.mask, bbox_backend, requested, "bbox", None, action.feather_radius
        )

    boxes = _instance_boxes(bbox_region)
    reasons: list[str] = []
    feather = action.feather_radius

    if subject_capabilities()["semantic_models"] and (
        requested == "sam_refined" or operation in EXPOSURE_OPERATIONS
    ):
        try:
            sam_mask = _sam_face_mask(detection_image, boxes, feather)
            qa = (
                face_mask_qa(sam_mask, boxes, image.shape)
                if _validate_mask(sam_mask, image.shape)
                else "qa_invalid_mask"
            )
            if qa is None:
                return _Resolution(
                    True, sam_mask, "face_sam_roi", requested, "sam_refined", None, feather
                )
            reasons.append(f"sam_refined: {qa}")
        except (SegmentationUnavailableError, SegmentationInferenceError, ValueError) as exc:
            reasons.append(f"sam_refined: {exc}")

    try:
        oval_region = resolve_module_region(
            detection_image,
            _request_for_action(action.model_copy(update={"face_mode": "oval"}), "face"),
            _face_resolver=detect_faces,
        )
        if oval_region.is_empty or not _validate_mask(oval_region.mask, image.shape):
            reasons.append("oval: no usable mask")
        else:
            qa = face_mask_qa(oval_region.mask, boxes, image.shape)
            if qa is None:
                return _Resolution(
                    True,
                    oval_region.mask,
                    str(oval_region.metadata.get("backend", "mediapipe_face_landmarker")),
                    requested,
                    "oval",
                    "; ".join(reasons) or None,
                    int(oval_region.metadata.get("feather_radius", feather)),
                )
            reasons.append(f"oval: {qa}")
    except (RegionBackendUnavailableError, RegionInferenceError, InvalidRegionRequestError) as exc:
        reasons.append(f"oval: {exc}")

    reason = "; ".join(reasons)
    logger.warning(
        "Face mode '%s' unavailable for region '%s' (%s). Falling back to 'bbox' at strength %.2f.",
        requested,
        action.region_id,
        reason,
        FALLBACK_STRENGTH_SCALE,
    )
    return _Resolution(
        True,
        bbox_region.mask,
        bbox_backend,
        requested,
        "bbox",
        reason,
        action.feather_radius,
        strength_scale=FALLBACK_STRENGTH_SCALE,
    )


def _resolve_action_region(image: np.ndarray, action: RegionOperation) -> _Resolution:
    """
    Tạo mặt nạ vùng cho một action thông qua Module 2 controller.

    Returns:
        _Resolution: proceed=False → bỏ qua action; mask=None → xử lý toàn ảnh; backend là bộ
        phân giải đã tạo mask (ví dụ 'heuristic' theo D1).
    """
    kind = _infer_region_kind(action)
    if kind == "full":
        return _Resolution(True, None, "geometry")
    if kind == "face":
        return _resolve_face_region(image, action)

    detection_image = _as_rgb_for_detection(image)
    request = _request_for_action(action, kind)
    try:
        # Không truyền resolver ngữ nghĩa: controller tự gọi resolve_prompt_instances
        # để instance_selection/instance_index có hiệu lực (test patch hàm đó trong controller).
        region = resolve_module_region(detection_image, request, _face_resolver=detect_faces)
    except (RegionBackendUnavailableError, RegionInferenceError) as exc:
        quadrant = _heuristic_quadrant(action.target_prompt) if kind == "semantic" else None
        if quadrant is None:
            logger.warning(
                "Region '%s' (%s) could not be resolved (%s). Skipping action '%s'.",
                action.region_id,
                kind,
                exc,
                action.operation,
            )
            return _skip(action, f"region_unresolved: {exc}")
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
        region.metadata["fallback_reason"] = f"semantic_unavailable: {exc}"
    except InvalidRegionRequestError as exc:
        logger.warning(
            "Invalid region request for '%s': %s. Skipping action '%s'.",
            action.region_id,
            exc,
            action.operation,
        )
        return _skip(action, f"invalid_region_request: {exc}")

    if region.is_empty or not _validate_mask(region.mask, image.shape):
        logger.warning(
            "No usable mask for region '%s' (%s). Skipping action '%s'.",
            action.region_id,
            kind,
            action.operation,
        )
        return _skip(action, "no_usable_mask")

    backend = str(region.metadata.get("backend", "unknown"))
    logger.info("Region '%s' (%s) resolved by backend '%s'.", action.region_id, kind, backend)
    return _Resolution(
        True,
        # Module 3 hiểu mask=None là toàn ảnh; giữ tối ưu này cho vùng 'full'.
        None if region.metadata.get("kind") == "full" else region.mask,
        backend,
        fallback_reason=region.metadata.get("fallback_reason"),
        feather_actual=int(region.metadata.get("feather_radius", action.feather_radius)),
    )


def _subject_box(
    action: RegionOperation, subject_boxes: Optional[SubjectBoxes]
) -> Optional[tuple[int, int, int, int]]:
    """bbox Perceive của vùng ngữ nghĩa trong action (khớp theo tên chuẩn hóa); None nếu không có."""
    if not subject_boxes or _infer_region_kind(action) != "semantic":
        return None
    box = subject_boxes.get(canonical_region(action.target_prompt))
    if box is None or len(box) != 4:
        return None
    ymin, xmin, ymax, xmax = (int(value) for value in box)
    return ymin, xmin, ymax, xmax


def _resolve_subject_region(
    image: np.ndarray,
    action: RegionOperation,
    box_2d: tuple[int, int, int, int],
    cache_key: Optional[str],
) -> _Resolution:
    """
    Vùng ngữ nghĩa Perceive đã định vị bằng bbox: dùng chung resolver với Perceive
    (region_engine.resolve_subject_mask, C2), nên mask xử lý trùng mask đã đo và trúng cache.
    Feather của action được truyền xuống resolver (R-03) và nằm trong cache key.
    Resolver rơi về 'full' (không định vị được) → bỏ qua action (D1, không xử lý toàn ảnh).
    """
    key = subject_cache_key(cache_key, image.shape) if cache_key else None
    target = canonical_region(action.target_prompt)
    subject = resolve_subject_mask(
        _as_rgb_for_detection(image),
        target,
        box_2d,
        cache_key=key,
        feather_radius=action.feather_radius,
    )
    if subject.source == "full" or not _validate_mask(subject.mask, image.shape):
        logger.warning(
            "Subject '%s' could not be located (source %s). Skipping action '%s'.",
            action.region_id,
            subject.source,
            action.operation,
        )
        return _skip(action, f"subject_not_located: {subject.source}")
    logger.info("Region '%s' resolved by subject resolver '%s'.", action.region_id, subject.source)
    return _Resolution(True, subject.mask, subject.source, feather_actual=action.feather_radius)


def _resolve_action_mask(
    image: np.ndarray,
    action: RegionOperation,
    subject_boxes: Optional[SubjectBoxes] = None,
    cache_key: Optional[str] = None,
) -> _Resolution:
    """
    Giải vùng cho một action khi thực thi kế hoạch. Vùng ngữ nghĩa có bbox từ Perceive
    → resolver chung; còn lại (face/full/quadrant, hoặc không có bbox) → đường cũ qua controller.
    """
    box_2d = _subject_box(action, subject_boxes)
    if box_2d is not None:
        return _resolve_subject_region(image, action, box_2d, cache_key)
    return _resolve_action_region(image, action)


def resolve_region_mask(
    image: np.ndarray,
    target_prompt: str,
    feather_radius: int = 15,
    box_2d: Optional[Sequence[int]] = None,
    cache_key: Optional[str] = None,
) -> tuple[bool, Optional[np.ndarray], Optional[str]]:
    """
    Tạo mặt nạ cho một vùng theo tên, cùng đường phân giải với execute_plan: vùng ngữ nghĩa có
    box_2d (bbox Gemini 0..1000) → resolver chung resolve_subject_mask; không có → face/full/
    quadrant/semantic qua controller (heuristic D1).

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
    boxes = {canonical_region(target_prompt): box_2d} if box_2d is not None else None
    box = _subject_box(probe, boxes)
    resolution = (
        _resolve_subject_region(image, probe, box, cache_key)
        if box is not None
        else _resolve_action_region(image, probe)
    )
    if not resolution.proceed:
        return False, None, None
    return True, resolution.mask, resolution.backend


def _blend_toward_input(before: np.ndarray, after: np.ndarray, scale: float) -> np.ndarray:
    """Giảm cường độ một thao tác: kéo kết quả về ảnh đầu vào (trung tính) theo hệ số scale."""
    start = before.astype(np.float32)
    mixed = start + scale * (after.astype(np.float32) - start)
    return np.clip(np.round(mixed), 0, 255).astype(np.uint8)


def _record(
    report: Optional[List[ExecutedAction]],
    action: RegionOperation,
    resolution: Optional[_Resolution],
    *,
    skipped: bool = False,
    skip_reason: Optional[str] = None,
) -> None:
    """Ghi một ExecutedAction (yêu cầu so với thực tế) vào report nếu có."""
    if report is None:
        return
    mask_hash, area = (None, None)
    if resolution is not None and (resolution.proceed or resolution.mask is not None):
        mask_hash, area = mask_fingerprint(resolution.mask)
        if resolution.mask is not None and mask_hash is not None:
            _remember_mask(resolution.mask, mask_hash)
    report.append(
        ExecutedAction(
            region_id=action.region_id,
            operation=action.operation,
            parameters=dict(action.parameters or {}),
            requested_mode=resolution.requested_mode if resolution else None,
            actual_mode=resolution.actual_mode if resolution else None,
            backend=resolution.backend if resolution else "",
            fallback_reason=resolution.fallback_reason if resolution else None,
            feather_requested=action.feather_radius,
            feather_actual=resolution.feather_actual if resolution else None,
            strength_scale=resolution.strength_scale if resolution else 1.0,
            mask_hash=mask_hash,
            mask_area=area,
            skipped=skipped,
            skip_reason=skip_reason,
        )
    )


def execute_plan(
    image: np.ndarray,
    plan: TreatmentPlan,
    subject_boxes: Optional[SubjectBoxes] = None,
    cache_key: Optional[str] = None,
    report: Optional[List[ExecutedAction]] = None,
) -> np.ndarray:
    """
    Thực thi tuần tự các hành động trong kế hoạch điều trị trên ảnh.
    Mỗi action được bọc riêng trong try...except — một action lỗi không làm sập cả pipeline.

    Args:
        image: Ảnh đầu vào ở vòng lặp hiện tại (RGB, uint8).
        plan: Kế hoạch điều trị đã được chuẩn hóa.
        subject_boxes: bbox Perceive theo tên subject (DiagnosisReport.subject_boxes); vùng
            ngữ nghĩa có bbox dùng resolver chung với Perceive.
        cache_key: image_cache_key của ảnh gốc của lượt chạy, để dùng lại mask đã phân giải.
        report: danh sách để executor ghi ExecutedAction cho từng action (kể cả action bị bỏ
            qua): mode yêu cầu so với thực tế, backend, feather, mask_hash; None thì bỏ qua.

    Returns:
        np.ndarray: Ảnh sau khi đã áp dụng toàn bộ các thao tác (RGB, uint8).
    """
    current_img = np.clip(image, 0, 255).astype(np.uint8)

    if not plan or not plan.actions:
        return current_img

    for action in plan.actions:
        resolution: Optional[_Resolution] = None
        try:
            # 1. Tạo mặt nạ vùng thông qua Module 2
            resolution = _resolve_action_mask(current_img, action, subject_boxes, cache_key)
            if not resolution.proceed:
                _record(
                    report, action, resolution, skipped=True, skip_reason=resolution.skip_reason
                )
                continue
            mask = resolution.mask
            before = current_img

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
                _record(
                    report, action, resolution, skipped=True, skip_reason="unsupported_operation"
                )
                continue

            # 3. Đảm bảo output luôn là np.uint8 [0, 255]
            current_img = np.clip(current_img, 0, 255).astype(np.uint8)
            # 4. Mask rơi về bbox (fallback) thì không giữ nguyên cường độ (R-04)
            if resolution.strength_scale < 1.0:
                current_img = _blend_toward_input(before, current_img, resolution.strength_scale)
            _record(report, action, resolution)

        except Exception as e:
            logger.warning(
                "Error executing action '%s' on region '%s': %s. Skipping this action.",
                action.operation,
                action.region_id,
                str(e),
                exc_info=True,
            )
            _record(report, action, resolution, skipped=True, skip_reason=f"error: {e}")
            # Giữ nguyên current_img, tiếp tục action tiếp theo

    return current_img
