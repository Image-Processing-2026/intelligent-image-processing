"""Lazy MediaPipe Face Landmarker adapter for face-oval masks.

The face detector in :mod:`face_detector` intentionally remains a bbox-only
baseline.  This module is a separate optional backend: it turns the official
MediaPipe face-oval landmark ring into one hard full-frame mask and one
contour per face.  It does not claim to be skin segmentation.
"""

from __future__ import annotations

import atexit
import math
import os
import threading
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Callable, Mapping, Protocol

import numpy as np

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "mediapipe" / "face_landmarker.task"
)
MODEL_PATH_ENV = "REGION_FACE_LANDMARKER_MODEL_PATH"
DEFAULT_NUM_FACES = 4

# The ordered cycle formed by MediaPipe FACEMESH_FACE_OVAL.  Keeping the ring
# explicit avoids the incorrect and common shortcut of connecting all 468
# landmarks in index order.
FACE_OVAL_INDICES = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109,
)


class FaceLandmarkerUnavailableError(RuntimeError):
    """Raised when the optional Tasks runtime or .task asset is unavailable."""


class FaceLandmarkerError(RuntimeError):
    """Raised when landmark inference returns malformed data."""


@dataclass(frozen=True, slots=True)
class FaceOval:
    """A hard mask and its closed pixel-space face-oval contour."""

    mask: np.ndarray
    contour: np.ndarray


class _LandmarkerBackend(Protocol):
    def detect(self, image: np.ndarray) -> object:
        """Return one landmark sequence per detected face."""

    def close(self) -> None:
        """Release backend resources."""


class _MediaPipeTasksLandmarkerBackend:
    def __init__(self, model_path: Path, num_faces: int) -> None:
        if not model_path.is_file():
            raise FaceLandmarkerUnavailableError(
                f"Face Landmarker model was not found at {model_path}. "
                f"Set {MODEL_PATH_ENV} to a verified .task checkpoint."
            )
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except Exception as exc:  # pragma: no cover - optional runtime
            raise FaceLandmarkerUnavailableError(
                "MediaPipe Tasks FaceLandmarker is unavailable; install a compatible mediapipe wheel"
            ) from exc
        try:
            options = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(model_path),
                    delegate=mp_python.BaseOptions.Delegate.CPU,
                ),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=num_faces,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._mp = mp
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:  # pragma: no cover - depends on checkpoint/runtime
            raise FaceLandmarkerUnavailableError(
                f"Could not initialize MediaPipe FaceLandmarker with {model_path}"
            ) from exc

    def detect(self, image: np.ndarray) -> object:
        try:
            result = self._landmarker.detect(
                self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=image)
            )
            faces = getattr(result, "face_landmarks", None)
            if faces is None:
                raise FaceLandmarkerError("MediaPipe result did not contain face_landmarks")
            return faces
        except FaceLandmarkerError:
            raise
        except Exception as exc:  # pragma: no cover - optional runtime
            raise FaceLandmarkerError("MediaPipe FaceLandmarker inference failed") from exc

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception as exc:  # pragma: no cover - optional runtime
            raise FaceLandmarkerUnavailableError("Could not close MediaPipe FaceLandmarker") from exc


def _create_default_backend(model_path: Path, num_faces: int) -> _LandmarkerBackend:
    return _MediaPipeTasksLandmarkerBackend(model_path, num_faces)


_LANDMARKER_FACTORY: Callable[[Path, int], _LandmarkerBackend] = _create_default_backend
_LANDMARKER_CACHE: _LandmarkerBackend | None = None
_LANDMARKER_CACHE_KEY: tuple[str, int, int] | None = None
_LANDMARKER_LOCK = threading.RLock()


def _resolve_model_path() -> Path:
    configured = os.environ.get(MODEL_PATH_ENV, "").strip()
    path = Path(configured) if configured else DEFAULT_MODEL_PATH
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.expanduser().resolve()


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


def _validate_num_faces(num_faces: object) -> int:
    if isinstance(num_faces, (bool, np.bool_)) or not isinstance(num_faces, (int, np.integer)):
        raise TypeError("num_faces must be a positive integer")
    value = int(num_faces)
    if value <= 0:
        raise ValueError("num_faces must be a positive integer")
    return value


def _get_backend_locked(num_faces: int) -> _LandmarkerBackend:
    global _LANDMARKER_CACHE, _LANDMARKER_CACHE_KEY
    model_path = _resolve_model_path()
    cache_key = (str(model_path), num_faces, id(_LANDMARKER_FACTORY))
    if _LANDMARKER_CACHE is not None and _LANDMARKER_CACHE_KEY == cache_key:
        return _LANDMARKER_CACHE
    if _LANDMARKER_CACHE is not None:
        _LANDMARKER_CACHE.close()
        _LANDMARKER_CACHE = None
        _LANDMARKER_CACHE_KEY = None
    try:
        backend = _LANDMARKER_FACTORY(model_path, num_faces)
    except FaceLandmarkerUnavailableError:
        raise
    except Exception as exc:
        raise FaceLandmarkerUnavailableError(
            f"Could not initialize face landmarker backend for {model_path}"
        ) from exc
    if not callable(getattr(backend, "detect", None)) or not callable(getattr(backend, "close", None)):
        raise FaceLandmarkerUnavailableError(
            "Face Landmarker backend must provide callable detect() and close() methods"
        )
    _LANDMARKER_CACHE = backend
    _LANDMARKER_CACHE_KEY = cache_key
    return backend


def close_face_landmarker() -> None:
    """Close and clear the process-local landmarker cache."""
    global _LANDMARKER_CACHE, _LANDMARKER_CACHE_KEY
    with _LANDMARKER_LOCK:
        backend = _LANDMARKER_CACHE
        _LANDMARKER_CACHE = None
        _LANDMARKER_CACHE_KEY = None
        if backend is not None:
            backend.close()


def reset_face_landmarker() -> None:
    """Alias useful for tests and application lifecycle hooks."""
    close_face_landmarker()


def _close_at_exit() -> None:
    try:
        close_face_landmarker()
    except Exception:
        pass


atexit.register(_close_at_exit)


def _coordinate(value: object, *, field: str, face_index: int, point_index: int) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        try:
            value = value.item()  # type: ignore[union-attr]
        except AttributeError as exc:
            raise FaceLandmarkerError(
                f"face {face_index} landmark {point_index} {field} is not numeric"
            ) from exc
    try:
        coordinate = float(value)
    except (TypeError, ValueError) as exc:
        raise FaceLandmarkerError(
            f"face {face_index} landmark {point_index} {field} is not numeric"
        ) from exc
    if not math.isfinite(coordinate) or not 0.0 <= coordinate <= 1.0:
        raise FaceLandmarkerError(
            f"face {face_index} landmark {point_index} {field} must be finite and in [0, 1]"
        )
    return coordinate


def _point_from_raw(raw: object, face_index: int, point_index: int) -> tuple[float, float]:
    if isinstance(raw, Mapping):
        try:
            raw_x, raw_y = raw["x"], raw["y"]
        except KeyError as exc:
            raise FaceLandmarkerError(
                f"face {face_index} landmark {point_index} is missing {exc.args[0]}"
            ) from exc
    else:
        try:
            raw_x, raw_y = getattr(raw, "x"), getattr(raw, "y")
        except AttributeError as exc:
            raise FaceLandmarkerError(
                f"face {face_index} landmark {point_index} has an invalid record shape"
            ) from exc
    return (
        _coordinate(raw_x, field="x", face_index=face_index, point_index=point_index),
        _coordinate(raw_y, field="y", face_index=face_index, point_index=point_index),
    )


def _contour_from_landmarks(raw: object, image_shape: tuple[int, int], face_index: int) -> np.ndarray:
    if raw is None or isinstance(raw, (str, bytes)):
        raise FaceLandmarkerError(f"face {face_index} landmarks are not a sequence")
    try:
        landmarks = list(raw)
    except TypeError as exc:
        raise FaceLandmarkerError(f"face {face_index} landmarks are not iterable") from exc
    highest_index = max(FACE_OVAL_INDICES)
    if len(landmarks) <= highest_index:
        raise FaceLandmarkerError(
            f"face {face_index} has {len(landmarks)} landmarks; face oval needs index {highest_index}"
        )
    height, width = image_shape
    points = [_point_from_raw(landmarks[index], face_index, index) for index in FACE_OVAL_INDICES]
    # x and y use the original image dimensions; no crop or resize transform
    # is introduced by this full-frame IMAGE-mode adapter.
    contour = np.asarray(
        [(round(x * (width - 1)), round(y * (height - 1))) for x, y in points],
        dtype=np.float32,
    )
    if len(np.unique(contour, axis=0)) < 3:
        raise FaceLandmarkerError(f"face {face_index} oval is degenerate")
    return contour


def _rasterize_contour(contour: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    # Pillow is already a core dependency and avoids importing OpenCV simply to
    # fill a polygon.  Import it lazily with the optional runtime.
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:  # pragma: no cover - Pillow is a project dependency
        raise FaceLandmarkerUnavailableError("Pillow is required to rasterize face contours") from exc
    height, width = image_shape
    canvas = Image.new("L", (width, height), 0)
    ImageDraw.Draw(canvas).polygon([tuple(point) for point in contour.tolist()], fill=1)
    return np.asarray(canvas, dtype=np.uint8).astype(bool)


def _ovals_from_raw(raw_faces: object, image_shape: tuple[int, int]) -> list[FaceOval]:
    if raw_faces is None or isinstance(raw_faces, (str, bytes)):
        raise FaceLandmarkerError("Face Landmarker backend returned a non-sequence")
    try:
        faces = list(raw_faces)
    except TypeError as exc:
        raise FaceLandmarkerError("Face Landmarker backend returned a non-iterable") from exc
    ovals: list[FaceOval] = []
    for index, face in enumerate(faces):
        contour = _contour_from_landmarks(face, image_shape, index)
        ovals.append(FaceOval(_rasterize_contour(contour, image_shape), contour))
    return ovals


def detect_face_ovals(image: np.ndarray, *, num_faces: int = DEFAULT_NUM_FACES) -> list[FaceOval]:
    """Return hard masks and contours for the face-oval landmark ring.

    The model is initialized only when this function is called.  Missing
    assets or dependencies raise :class:`FaceLandmarkerUnavailableError`; an
    empty successful result is returned as ``[]``.  Calls are serialized to
    prevent cached MediaPipe task state from crossing requests.
    """
    validated_image = _validate_image(image)
    count = _validate_num_faces(num_faces)
    contiguous_image = np.ascontiguousarray(validated_image)
    with _LANDMARKER_LOCK:
        backend = _get_backend_locked(count)
        try:
            raw_faces = backend.detect(contiguous_image)
        except (FaceLandmarkerUnavailableError, FaceLandmarkerError):
            raise
        except Exception as exc:
            raise FaceLandmarkerError("Face Landmarker backend inference failed") from exc
        return _ovals_from_raw(raw_faces, validated_image.shape[:2])


__all__ = [
    "DEFAULT_MODEL_PATH",
    "DEFAULT_NUM_FACES",
    "FACE_OVAL_INDICES",
    "MODEL_PATH_ENV",
    "FaceLandmarkerError",
    "FaceLandmarkerUnavailableError",
    "FaceOval",
    "close_face_landmarker",
    "detect_face_ovals",
    "reset_face_landmarker",
]
