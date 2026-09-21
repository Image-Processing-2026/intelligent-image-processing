"""MediaPipe Tasks face detection and full-frame soft-mask synthesis.

The detector is intentionally lazy: importing this module does not import
MediaPipe or require a model file. Runtime failures are reported explicitly so
an unavailable detector cannot silently turn a face-only operation into a
full-image operation.
"""

from __future__ import annotations

import atexit
import math
import os
import threading
import warnings
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Callable, Mapping, Protocol

import numpy as np

from .spatial import _validate_feather_radius, create_bbox_mask

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "mediapipe" / "face_detection_full_range.tflite"
MODEL_PATH_ENV = "REGION_FACE_MODEL_PATH"
MIN_DETECTION_CONFIDENCE = 0.5
MIN_SUPPRESSION_THRESHOLD = 0.3


class FaceDetectorUnavailableError(RuntimeError):
    """Raised when MediaPipe, the checkpoint, or backend initialization is unavailable."""


class FaceDetectionError(RuntimeError):
    """Raised when inference returns malformed data or fails during execution."""


@dataclass(frozen=True)
class FaceDetectionRecord:
    """Pixel-space face detection record returned by the backend adapter."""

    x: float
    y: float
    width: float
    height: float
    score: float


class _DetectorBackend(Protocol):
    def detect(self, image: np.ndarray) -> object:
        """Run one IMAGE-mode inference and return backend detections."""

    def close(self) -> None:
        """Release backend resources."""


class _MediaPipeTasksBackend:
    """Small adapter around the MediaPipe Tasks FaceDetector API."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise FaceDetectorUnavailableError(
                f"Face detector model was not found at {model_path}. "
                f"Set {MODEL_PATH_ENV} to a verified .tflite checkpoint."
            )

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except Exception as exc:  # pragma: no cover - depends on optional runtime
            raise FaceDetectorUnavailableError(
                "MediaPipe Tasks is unavailable; install a compatible mediapipe wheel"
            ) from exc

        try:
            options = vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(model_path),
                    delegate=mp_python.BaseOptions.Delegate.CPU,
                ),
                running_mode=vision.RunningMode.IMAGE,
                min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_suppression_threshold=MIN_SUPPRESSION_THRESHOLD,
            )
            self._mp = mp
            self._detector = vision.FaceDetector.create_from_options(options)
        except Exception as exc:  # pragma: no cover - depends on checkpoint/runtime
            raise FaceDetectorUnavailableError(
                f"Could not initialize MediaPipe FaceDetector with {model_path}"
            ) from exc

    def detect(self, image: np.ndarray) -> list[FaceDetectionRecord]:
        try:
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=image,
            )
            result = self._detector.detect(mp_image)
            detections = getattr(result, "detections", None)
            if detections is None:
                raise FaceDetectionError("MediaPipe result did not contain detections")
            records: list[FaceDetectionRecord] = []
            for index, detection in enumerate(detections):
                bbox = getattr(detection, "bounding_box", None)
                categories = getattr(detection, "categories", None)
                if bbox is None or not categories:
                    raise FaceDetectionError(
                        f"MediaPipe detection {index} has no bounding box or score"
                    )
                score = getattr(categories[0], "score", None)
                records.append(
                    FaceDetectionRecord(
                        x=getattr(bbox, "origin_x"),
                        y=getattr(bbox, "origin_y"),
                        width=getattr(bbox, "width"),
                        height=getattr(bbox, "height"),
                        score=score,
                    )
                )
            return records
        except FaceDetectionError:
            raise
        except Exception as exc:
            raise FaceDetectionError("MediaPipe face detection inference failed") from exc

    def close(self) -> None:
        try:
            self._detector.close()
        except Exception as exc:  # pragma: no cover - depends on optional runtime
            raise FaceDetectorUnavailableError("Could not close MediaPipe FaceDetector") from exc


def _create_default_backend(model_path: Path) -> _DetectorBackend:
    return _MediaPipeTasksBackend(model_path)


_DETECTOR_FACTORY: Callable[[Path], _DetectorBackend] = _create_default_backend
_DETECTOR_CACHE: _DetectorBackend | None = None
_DETECTOR_CACHE_KEY: tuple[str, int] | None = None
_DETECTOR_LOCK = threading.RLock()


def _resolve_model_path() -> Path:
    configured = os.environ.get(MODEL_PATH_ENV, "").strip()
    path = Path(configured) if configured else DEFAULT_MODEL_PATH
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.expanduser().resolve()


def _get_backend_locked() -> _DetectorBackend:
    """Return a cached backend; caller must hold ``_DETECTOR_LOCK``."""
    global _DETECTOR_CACHE, _DETECTOR_CACHE_KEY
    model_path = _resolve_model_path()
    cache_key = (str(model_path), id(_DETECTOR_FACTORY))
    if _DETECTOR_CACHE is not None and _DETECTOR_CACHE_KEY == cache_key:
        return _DETECTOR_CACHE

    if _DETECTOR_CACHE is not None:
        _DETECTOR_CACHE.close()
        _DETECTOR_CACHE = None
        _DETECTOR_CACHE_KEY = None

    try:
        backend = _DETECTOR_FACTORY(model_path)
    except FaceDetectorUnavailableError:
        raise
    except Exception as exc:
        raise FaceDetectorUnavailableError(
            f"Could not initialize face detector backend for {model_path}"
        ) from exc
    if not callable(getattr(backend, "detect", None)) or not callable(
        getattr(backend, "close", None)
    ):
        raise FaceDetectorUnavailableError(
            "Face detector backend must provide callable detect() and close() methods"
        )
    _DETECTOR_CACHE = backend
    _DETECTOR_CACHE_KEY = cache_key
    return backend


def close_face_detector() -> None:
    """Close and clear the process-local detector cache."""
    global _DETECTOR_CACHE, _DETECTOR_CACHE_KEY
    with _DETECTOR_LOCK:
        backend = _DETECTOR_CACHE
        _DETECTOR_CACHE = None
        _DETECTOR_CACHE_KEY = None
        if backend is not None:
            backend.close()


def reset_face_detector() -> None:
    """Alias used by tests and application lifecycle hooks."""
    close_face_detector()


def _close_at_exit() -> None:
    try:
        close_face_detector()
    except Exception:
        pass


atexit.register(_close_at_exit)


def _validate_image(image: object) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.dtype != np.dtype(np.uint8):
        raise TypeError("image dtype must be uint8")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3) in RGB channel order")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must be non-empty")
    return image


def _validate_expand_ratio(expand_ratio: object) -> float:
    if isinstance(expand_ratio, (bool, np.bool_)) or not isinstance(expand_ratio, Real):
        raise TypeError("expand_ratio must be a real number in [0, 1]")
    ratio = float(expand_ratio)
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError("expand_ratio must be finite and in [0, 1]")
    return ratio


def _numeric_field(value: object, *, field: str, index: int) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise FaceDetectionError(f"detection {index} field {field} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FaceDetectionError(f"detection {index} field {field} is not finite")
    return result


def _record_from_raw(raw: object, index: int) -> FaceDetectionRecord:
    if isinstance(raw, FaceDetectionRecord):
        values = [raw.x, raw.y, raw.width, raw.height, raw.score]
    elif isinstance(raw, Mapping):
        try:
            values = [raw[name] for name in ("x", "y", "width", "height", "score")]
        except KeyError as exc:
            raise FaceDetectionError(f"detection {index} is missing {exc.args[0]}") from exc
    else:
        try:
            values = [getattr(raw, name) for name in ("x", "y", "width", "height", "score")]
        except AttributeError as exc:
            raise FaceDetectionError(f"detection {index} has an invalid record shape") from exc

    return FaceDetectionRecord(
        x=_numeric_field(values[0], field="x", index=index),
        y=_numeric_field(values[1], field="y", index=index),
        width=_numeric_field(values[2], field="width", index=index),
        height=_numeric_field(values[3], field="height", index=index),
        score=_numeric_field(values[4], field="score", index=index),
    )


def _records_to_masks(
    raw_records: object,
    image_shape: tuple[int, int],
    feather_radius: int,
    expand_ratio: float,
) -> list[np.ndarray]:
    if raw_records is None or isinstance(raw_records, (str, bytes)):
        raise FaceDetectionError("detector backend returned a non-sequence result")
    try:
        records = list(raw_records)
    except TypeError as exc:
        raise FaceDetectionError("detector backend returned a non-iterable result") from exc

    height, width = image_shape
    candidates: list[tuple[tuple[float, float, float, float, float, int], FaceDetectionRecord]] = []
    for index, raw in enumerate(records):
        record = _record_from_raw(raw, index)
        if not 0.0 <= record.score <= 1.0:
            raise FaceDetectionError(f"detection {index} score must be in [0, 1]")
        if record.width <= 0.0 or record.height <= 0.0:
            raise FaceDetectionError(f"detection {index} width and height must be positive")
        raw_key = (
            record.y,
            record.x,
            record.y + record.height,
            record.x + record.width,
            -record.score,
            index,
        )
        candidates.append((raw_key, record))

    masks: list[np.ndarray] = []
    for _, record in sorted(candidates, key=lambda item: item[0]):
        xmin = math.floor(record.x - expand_ratio * record.width)
        ymin = math.floor(record.y - expand_ratio * record.height)
        xmax = math.ceil(record.x + record.width + expand_ratio * record.width)
        ymax = math.ceil(record.y + record.height + expand_ratio * record.height)

        if xmax <= 0 or ymax <= 0 or xmin >= width or ymin >= height:
            warnings.warn(
                "Ignoring a valid face detection that is completely outside the image",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        try:
            mask = create_bbox_mask(
                image_shape,
                (xmin, ymin, xmax, ymax),
                feather_radius=feather_radius,
            )
        except Exception as exc:
            raise FaceDetectionError(
                f"Could not convert detection bbox {(xmin, ymin, xmax, ymax)} to a mask"
            ) from exc
        if mask.dtype != np.dtype(np.float32) or mask.shape != image_shape:
            raise FaceDetectionError("bbox mask backend returned an invalid mask contract")
        masks.append(mask)
    return masks


def detect_faces(
    image: np.ndarray,
    feather_radius: int = 20,
    *,
    expand_ratio: float = 0.15,
) -> list[np.ndarray]:
    """Detect faces and return one feathered full-frame mask per face.

    Args:
        image: RGB ``uint8`` NumPy array with shape ``(H, W, 3)``. BGR/RGBA
            arrays are rejected rather than guessed or reordered.
        feather_radius: Non-negative integer passed to ``create_bbox_mask``.
        expand_ratio: Finite real value in ``[0, 1]`` expanded on each side
            of the detector bbox. The default ``0.15`` increases each
            dimension by approximately 30% before clipping.

    Returns:
        A list of independent ``float32`` masks in stable raw-bbox order.
        A successful inference with no qualifying detections returns ``[]``.

    Raises:
        TypeError/ValueError: If public inputs violate the contract.
        FaceDetectorUnavailableError: If MediaPipe or the configured model is
            unavailable or cannot be initialized.
        FaceDetectionError: If inference or backend output is malformed.

    The MediaPipe Tasks backend is initialized lazily from
    ``REGION_FACE_MODEL_PATH`` or the repository default
    ``models/mediapipe/face_detection_full_range.tflite``. No model is
    downloaded implicitly. Backend initialization and inference are serialized
    because the detector instance is cached per process and is not assumed to
    be thread-safe.
    """
    validated_image = _validate_image(image)
    radius = _validate_feather_radius(feather_radius)
    ratio = _validate_expand_ratio(expand_ratio)
    contiguous_image = np.ascontiguousarray(validated_image)

    with _DETECTOR_LOCK:
        backend = _get_backend_locked()
        try:
            raw_records = backend.detect(contiguous_image)
        except (FaceDetectorUnavailableError, FaceDetectionError):
            raise
        except Exception as exc:
            raise FaceDetectionError("Face detector backend inference failed") from exc
        return _records_to_masks(
            raw_records,
            validated_image.shape[:2],
            radius,
            ratio,
        )


__all__ = [
    "FaceDetectionError",
    "FaceDetectionRecord",
    "FaceDetectorUnavailableError",
    "close_face_detector",
    "detect_faces",
    "reset_face_detector",
]
