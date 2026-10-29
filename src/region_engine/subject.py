"""Phân giải mặt nạ chủ thể theo chuỗi ưu tiên (Analysis v2, hợp đồng C2).

Thứ tự: khuôn mặt (MediaPipe) → có model: DINO khớp bbox Gemini → MobileSAM;
không khớp → bbox Gemini → MobileSAM → không model: bbox Gemini → GrabCut;
không bbox → từ khóa hình học → toàn ảnh. Mọi mask đều đi qua
``create_soft_mask`` (ADR-003).
"""

from __future__ import annotations

import logging
import math
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import cv2
import numpy as np

from .controller import _FACE_NAMES, _dependency_available, _resolved_path
from .detector import (
    _FULL_COMMANDS,
    _QUADRANT_COMMANDS,
    PromptSegmentationConfig,
    _detect_with_config,
    _mask_from_backend,
    _prepare_detections,
    _semantic_prompt,
    normalize_prompt,
)
from .face_detector import (
    DEFAULT_MODEL_PATH as DEFAULT_FACE_MODEL_PATH,
)
from .face_detector import (
    MODEL_PATH_ENV as FACE_MODEL_ENV,
)
from .face_detector import (
    FaceDetectionError,
    FaceDetectorUnavailableError,
    detect_faces,
)
from .mask_utils import create_soft_mask
from .segmentation_backend import (
    _SEGMENTATION_LOCK,
    DEFAULT_DINO_MODEL_PATH,
    DEFAULT_MOBILE_SAM_CHECKPOINT,
    DINO_MODEL_ENV,
    MOBILE_SAM_CHECKPOINT_ENV,
    SegmentationBackend,
    SegmentationInferenceError,
    SegmentationUnavailableError,
    _ensure_encoder_locked,
    _get_backend_locked,
)
from .spatial import create_quadrant_mask

_LOGGER = logging.getLogger(__name__)

BoxXYXY = Tuple[float, float, float, float]
"""Pixel ``(xmin, ymin, xmax, ymax)``, như ``spatial.create_bbox_mask``."""
Box2D = Tuple[int, int, int, int]
"""Chuẩn Gemini ``(ymin, xmin, ymax, xmax)`` trong khoảng 0..1000."""
MaskSource = Literal[
    "face_mediapipe", "dino_sam", "gemini_sam", "gemini_grabcut", "quadrant", "full"
]

DINO_MATCH_IOU = 0.3
"""Ngưỡng IoU tối thiểu giữa detection DINO và bbox Gemini để coi là khớp."""
DINO_MAX_BOX_FRACTION = 0.9
"""Detection DINO phủ từ ngưỡng này trở lên của khung ảnh bị coi là trượt."""
UNANCHORED_DINO_CONFIDENCE = 0.5
"""Trần confidence khi DINO chạy không có bbox Gemini để đối chiếu."""
GRABCUT_MAX_SIDE = 1024
DEFAULT_FEATHER_RADIUS = 15
_CONFIDENCE_FACE = 0.9
_CONFIDENCE_GEMINI_SAM = 0.6
_CONFIDENCE_GEMINI_GRABCUT = 0.4
_CONFIDENCE_QUADRANT = 0.3
_CONFIDENCE_FULL = 0.1
_SUBJECT_CACHE_MAX_ENTRIES = 16

_QUADRANT_KEYWORDS: dict[str, str] = {
    "sky": "top",
    "trời": "top",
    "bầu trời": "top",
    "ground": "bottom",
    "đất": "bottom",
    "mặt đất": "bottom",
}


@dataclass(frozen=True)
class SubjectMask:
    """Mặt nạ float32 [0, 1] đã feather cùng nguồn gốc và độ tin cậy."""

    mask: np.ndarray
    source: MaskSource
    confidence: float
    box: Optional[BoxXYXY]


_SubjectCacheKey = tuple[str, str, Optional[Box2D], Optional[int]]
_SUBJECT_CACHE: OrderedDict[_SubjectCacheKey, SubjectMask] = OrderedDict()
_SUBJECT_CACHE_LOCK = threading.Lock()
_UNAVAILABLE_LOGGED = False


def capabilities() -> dict[str, bool]:
    """Báo ``face`` và ``semantic_models`` mà không nạp model nặng."""
    face_path = _resolved_path(FACE_MODEL_ENV, DEFAULT_FACE_MODEL_PATH)
    dino_path = _resolved_path(DINO_MODEL_ENV, DEFAULT_DINO_MODEL_PATH)
    sam_path = _resolved_path(MOBILE_SAM_CHECKPOINT_ENV, DEFAULT_MOBILE_SAM_CHECKPOINT)
    face_ready = face_path.is_file() and _dependency_available("mediapipe")
    semantic_ready = (
        dino_path.is_dir()
        and sam_path.is_file()
        and all(_dependency_available(name) for name in ("torch", "transformers", "mobile_sam"))
    )
    return {"face": face_ready, "semantic_models": semantic_ready}


def runtime_capabilities() -> dict[str, object]:
    """Môi trường phân vùng đang chạy: interpreter, khả năng chủ thể, có ``mobile_sam`` hay không.

    Không nạp model nặng; dùng để ghi lúc khởi động, trả ở ``/system`` và trong export phiên.
    """
    return {
        "interpreter": sys.executable,
        "mobile_sam_installed": _dependency_available("mobile_sam"),
        **capabilities(),
    }


def box2d_to_xyxy(box_2d: Box2D, image_shape: Tuple[int, int]) -> BoxXYXY:
    """Đổi bbox Gemini ``(ymin, xmin, ymax, xmax)`` 0..1000 sang pixel XYXY.

    Raises:
        ValueError: Nếu bbox không có đúng 4 số hoặc suy biến sau khi kẹp vào 0..1000.
    """
    if len(box_2d) != 4:
        raise ValueError("box_2d must contain four values")
    ymin, xmin, ymax, xmax = (min(max(int(value), 0), 1000) for value in box_2d)
    if ymax <= ymin or xmax <= xmin:
        raise ValueError("box_2d must satisfy ymax > ymin and xmax > xmin")
    height, width = image_shape
    return (
        xmin * width / 1000.0,
        ymin * height / 1000.0,
        xmax * width / 1000.0,
        ymax * height / 1000.0,
    )


def segment_box(image: np.ndarray, box: BoxXYXY, *, feather_radius: int = 15) -> np.ndarray:
    """Phân đoạn MobileSAM trong bbox pixel XYXY, trả về mặt nạ float32 đã feather.

    Encoder chỉ chạy lại khi ảnh đổi. Raise ``SegmentationUnavailableError`` nếu thiếu model.
    """
    rgb = _as_rgb(image)
    xyxy = _validate_box(box)
    with _SEGMENTATION_LOCK:
        backend = _get_backend_locked()
        return _predict_soft_locked(backend, rgb, xyxy, feather_radius)


def refine_bbox_mask(
    image: np.ndarray,
    box: BoxXYXY,
    *,
    iterations: int = 5,
    max_side: int = GRABCUT_MAX_SIDE,
    feather_radius: int = 15,
) -> np.ndarray:
    """GrabCut khởi tạo từ bbox trên ảnh thu về ``max_side``, rồi upscale và feather.

    Không cần model. Vùng ngoài bbox luôn bằng 0. Bbox suy biến cho mặt nạ rỗng.
    """
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if isinstance(max_side, bool) or not isinstance(max_side, int) or max_side < 1:
        raise ValueError("max_side must be a positive integer")
    rgb = _as_rgb(image)
    height, width = rgb.shape[:2]
    clipped = _clip_box(box, (height, width))
    if clipped is None:
        return np.zeros((height, width), dtype=np.float32)
    x0, y0, x1, y1 = clipped

    scale = min(1.0, max_side / max(height, width))
    small_h = max(1, round(height * scale))
    small_w = max(1, round(width * scale))
    small = (
        rgb if scale == 1.0 else cv2.resize(rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
    )
    sx = small_w / width
    sy = small_h / height
    sx0 = int(math.floor(x0 * sx))
    sy0 = int(math.floor(y0 * sy))
    sx1 = min(small_w, max(sx0 + 1, int(math.ceil(x1 * sx))))
    sy1 = min(small_h, max(sy0 + 1, int(math.ceil(y1 * sy))))
    rect = (sx0, sy0, sx1 - sx0, sy1 - sy0)

    grabcut_mask = np.zeros((small_h, small_w), dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        np.ascontiguousarray(small),
        grabcut_mask,
        rect,
        background_model,
        foreground_model,
        iterations,
        cv2.GC_INIT_WITH_RECT,
    )
    foreground = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    if scale != 1.0:
        foreground = cv2.resize(foreground, (width, height), interpolation=cv2.INTER_LINEAR)
    hard = foreground >= 128
    inside_box = np.zeros((height, width), dtype=bool)
    inside_box[y0:y1, x0:x1] = True
    return create_soft_mask(hard & inside_box, feather_radius=feather_radius)


def resolve_subject_mask(
    image: np.ndarray,
    name: str,
    box_2d: Optional[Box2D] = None,
    *,
    use_semantic: Optional[bool] = None,
    cache_key: Optional[str] = None,
    feather_radius: Optional[int] = None,
) -> SubjectMask:
    """Phân giải mặt nạ cho một tên vùng theo chuỗi ưu tiên ở đầu module.

    ``use_semantic=None`` lấy theo ``capabilities()``. ``feather_radius=None`` dùng
    ``DEFAULT_FEATHER_RADIUS``. Khi ``cache_key`` khác ``None``, kết quả được cache theo
    ``(cache_key, name, box_2d, feather)`` và trả lại cùng object: người gọi không được sửa
    mặt nạ tại chỗ; hai feather khác nhau không dùng chung mục cache. ``cache_key`` do người
    gọi chịu trách nhiệm (thường là hash của ảnh gốc).
    """
    rgb = _as_rgb(image)
    feather = DEFAULT_FEATHER_RADIUS if feather_radius is None else max(0, int(feather_radius))
    # Feather mặc định dùng chung mục cache với người gọi không khai báo feather (Perceive)
    feather_key = None if feather == DEFAULT_FEATHER_RADIUS else feather
    cache_id = None if cache_key is None else (cache_key, name, _box_key(box_2d), feather_key)
    if cache_id is not None:
        cached = _cache_get(cache_id)
        if cached is not None:
            return cached
    result = _resolve(rgb, name, box_2d, use_semantic, feather)
    if cache_id is not None:
        _cache_put(cache_id, result)
    return result


def clear_subject_mask_cache() -> None:
    """Xóa cache mặt nạ chủ thể của process."""
    with _SUBJECT_CACHE_LOCK:
        _SUBJECT_CACHE.clear()


def _resolve(
    rgb: np.ndarray,
    name: str,
    box_2d: Optional[Box2D],
    use_semantic: Optional[bool],
    feather: int = DEFAULT_FEATHER_RADIUS,
) -> SubjectMask:
    shape = rgb.shape[:2]
    normalized = normalize_prompt(name)
    if normalized in _FULL_COMMANDS:
        return _full_subject(shape, feather)
    if normalized in _QUADRANT_COMMANDS:
        return _quadrant_subject(shape, _QUADRANT_COMMANDS[normalized], feather)

    if normalized in _FACE_NAMES:
        face = _face_subject(rgb)
        if face is not None:
            return face

    gemini_box = _gemini_box(box_2d, shape)
    semantic_ready = capabilities()["semantic_models"] if use_semantic is None else use_semantic
    if semantic_ready:
        try:
            semantic = _semantic_subject(rgb, normalized, gemini_box, feather)
        except SegmentationUnavailableError as exc:
            _log_semantic_unavailable_once(exc)
        except SegmentationInferenceError as exc:
            _LOGGER.warning("semantic subject segmentation failed for %r: %s", name, exc)
        else:
            if semantic is not None:
                return semantic

    if gemini_box is not None:
        grabcut = _grabcut_subject(rgb, gemini_box, feather)
        if grabcut is not None:
            return grabcut

    region = _QUADRANT_KEYWORDS.get(normalized)
    if region is not None:
        return _quadrant_subject(shape, region, feather)
    return _full_subject(shape, feather)


def _semantic_subject(
    rgb: np.ndarray,
    prompt: str,
    gemini_box: Optional[BoxXYXY],
    feather: int = DEFAULT_FEATHER_RADIUS,
) -> Optional[SubjectMask]:
    """DINO theo tên vùng rồi MobileSAM. Trả ``None`` khi không có mặt nạ không rỗng."""
    shape = rgb.shape[:2]
    config = PromptSegmentationConfig()
    with _SEGMENTATION_LOCK:
        backend = _get_backend_locked()
        raw = _detect_with_config(backend, rgb, _semantic_prompt(prompt), config)
        detections = [
            detection
            for detection in _prepare_detections(raw, shape, config)
            if _box_fraction(detection.box, shape) < DINO_MAX_BOX_FRACTION
        ]

        if gemini_box is None:
            if not detections:
                return None
            best = max(detections, key=lambda detection: detection.score)
            mask = _predict_soft_locked(backend, rgb, best.box, feather)
            if not mask.any():
                return None
            confidence = min(best.score, UNANCHORED_DINO_CONFIDENCE)
            return SubjectMask(mask, "dino_sam", confidence, best.box)

        matched = [
            detection
            for detection in detections
            if _box_iou(detection.box, gemini_box) >= DINO_MATCH_IOU
        ]
        if matched:
            best = max(matched, key=lambda detection: detection.score)
            mask = _predict_soft_locked(backend, rgb, best.box, feather)
            if mask.any():
                return SubjectMask(mask, "dino_sam", best.score, best.box)
        mask = _predict_soft_locked(backend, rgb, gemini_box, feather)
        if not mask.any():
            return None
        return SubjectMask(mask, "gemini_sam", _CONFIDENCE_GEMINI_SAM, gemini_box)


def _face_subject(rgb: np.ndarray) -> Optional[SubjectMask]:
    shape = rgb.shape[:2]
    try:
        masks = detect_faces(rgb)
    except (FaceDetectorUnavailableError, FaceDetectionError) as exc:
        _LOGGER.info("face subject mask unavailable: %s", exc)
        return None
    if not masks:
        return SubjectMask(np.zeros(shape, dtype=np.float32), "face_mediapipe", 0.0, None)
    union = np.clip(np.max(np.stack(masks), axis=0), 0.0, 1.0).astype(np.float32)
    return SubjectMask(union, "face_mediapipe", _CONFIDENCE_FACE, None)


def _grabcut_subject(
    rgb: np.ndarray, gemini_box: BoxXYXY, feather: int = DEFAULT_FEATHER_RADIUS
) -> Optional[SubjectMask]:
    try:
        mask = refine_bbox_mask(rgb, gemini_box, feather_radius=feather)
    except cv2.error as exc:
        _LOGGER.warning("GrabCut subject refinement failed: %s", exc)
        return None
    if not mask.any():
        return None
    return SubjectMask(mask, "gemini_grabcut", _CONFIDENCE_GEMINI_GRABCUT, gemini_box)


def _quadrant_subject(
    shape: Tuple[int, int], region: str, feather: int = DEFAULT_FEATHER_RADIUS
) -> SubjectMask:
    hard = create_quadrant_mask(shape, region, 0) > 0
    mask = create_soft_mask(hard, feather_radius=feather)
    return SubjectMask(mask, "quadrant", _CONFIDENCE_QUADRANT, None)


def _full_subject(shape: Tuple[int, int], feather: int = DEFAULT_FEATHER_RADIUS) -> SubjectMask:
    mask = create_soft_mask(np.ones(shape, dtype=bool), feather_radius=feather)
    return SubjectMask(mask, "full", _CONFIDENCE_FULL, None)


def _predict_soft_locked(
    backend: SegmentationBackend, rgb: np.ndarray, box: BoxXYXY, feather_radius: int
) -> np.ndarray:
    """Chạy encoder (nếu ảnh mới) rồi decoder cho một bbox; caller giữ ``_SEGMENTATION_LOCK``."""
    _ensure_encoder_locked(backend, rgb)
    hard = _mask_from_backend(backend.predict(box), rgb.shape[:2], 0)
    return create_soft_mask(hard, feather_radius=feather_radius)


def _as_rgb(image: object) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.dtype != np.dtype(np.uint8):
        raise TypeError("image dtype must be uint8")
    if image.ndim == 2:
        image = np.repeat(image[..., np.newaxis], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif not (image.ndim == 3 and image.shape[2] == 3):
        raise ValueError("image must have shape (H, W), (H, W, 1) or (H, W, 3)")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must be non-empty")
    return np.ascontiguousarray(image)


def _validate_box(box: object) -> BoxXYXY:
    values = tuple(float(value) for value in box)  # type: ignore[attr-defined]
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("box must contain four finite values")
    xmin, ymin, xmax, ymax = values
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("box must satisfy xmax > xmin and ymax > ymin")
    return values


def _clip_box(box: BoxXYXY, shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    height, width = shape
    x0 = max(0, math.floor(box[0]))
    y0 = max(0, math.floor(box[1]))
    x1 = min(width, math.ceil(box[2]))
    y1 = min(height, math.ceil(box[3]))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _gemini_box(box_2d: Optional[Box2D], shape: Tuple[int, int]) -> Optional[BoxXYXY]:
    if box_2d is None:
        return None
    try:
        return box2d_to_xyxy(box_2d, shape)
    except ValueError as exc:
        _LOGGER.info("ignoring degenerate subject box %r: %s", box_2d, exc)
        return None


def _box_key(box_2d: Optional[Box2D]) -> Optional[Box2D]:
    if box_2d is None:
        return None
    return tuple(int(value) for value in box_2d)  # type: ignore[return-value]


def _box_iou(first: BoxXYXY, second: BoxXYXY) -> float:
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _box_fraction(box: BoxXYXY, shape: Tuple[int, int]) -> float:
    height, width = shape
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    return area / float(height * width)


def _cache_get(key: _SubjectCacheKey) -> Optional[SubjectMask]:
    with _SUBJECT_CACHE_LOCK:
        cached = _SUBJECT_CACHE.get(key)
        if cached is not None:
            _SUBJECT_CACHE.move_to_end(key)
        return cached


def _cache_put(key: _SubjectCacheKey, value: SubjectMask) -> None:
    with _SUBJECT_CACHE_LOCK:
        _SUBJECT_CACHE[key] = value
        _SUBJECT_CACHE.move_to_end(key)
        while len(_SUBJECT_CACHE) > _SUBJECT_CACHE_MAX_ENTRIES:
            _SUBJECT_CACHE.popitem(last=False)


def _log_semantic_unavailable_once(exc: BaseException) -> None:
    global _UNAVAILABLE_LOGGED
    if not _UNAVAILABLE_LOGGED:
        _UNAVAILABLE_LOGGED = True
        _LOGGER.info("semantic subject models unavailable, using fallback chain: %s", exc)


__all__ = [
    "Box2D",
    "BoxXYXY",
    "DINO_MATCH_IOU",
    "DINO_MAX_BOX_FRACTION",
    "MaskSource",
    "SubjectMask",
    "box2d_to_xyxy",
    "capabilities",
    "clear_subject_mask_cache",
    "refine_bbox_mask",
    "resolve_subject_mask",
    "segment_box",
]
