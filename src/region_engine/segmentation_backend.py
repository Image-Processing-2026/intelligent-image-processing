"""Optional GroundingDINO + MobileSAM backend for prompt segmentation.

The module deliberately keeps Torch, Transformers and MobileSAM imports lazy.
Importing the region engine therefore remains safe for the classical mask
APIs, while missing weights/dependencies are reported only for semantic
inference.
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

DEFAULT_DINO_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "segmentation" / "grounding-dino-tiny"
)
DEFAULT_MOBILE_SAM_CHECKPOINT = (
    Path(__file__).resolve().parents[2] / "models" / "segmentation" / "mobile_sam.pt"
)
DINO_MODEL_ENV = "REGION_DINO_MODEL_PATH"
MOBILE_SAM_CHECKPOINT_ENV = "REGION_MOBILE_SAM_CHECKPOINT"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.8


class SegmentationUnavailableError(RuntimeError):
    """Raised when optional dependencies, model files, or backend init fail."""


class SegmentationInferenceError(RuntimeError):
    """Raised when model inference or an adapter output is malformed."""


@dataclass(frozen=True)
class GroundingDetection:
    """One GroundingDINO detection in clipped image-pixel XYXY coordinates."""

    box: tuple[float, float, float, float]
    score: float
    phrase: str


class SegmentationBackend(Protocol):
    def detect(self, image: np.ndarray, prompt: str) -> object:
        """Return raw or normalized bbox detections for one image."""

    def set_image(self, image: np.ndarray) -> None:
        """Set the image embedding used by subsequent box predictions."""

    def predict(self, box: tuple[float, float, float, float]) -> object:
        """Return one binary mask for one image-pixel XYXY box."""

    def reset_image(self) -> None:
        """Clear predictor image state."""

    def close(self) -> None:
        """Release model resources."""


def _resolve_path(environment_name: str, default: Path) -> Path:
    configured = os.environ.get(environment_name, "").strip()
    path = Path(configured) if configured else default
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.expanduser().resolve()


def _as_numpy(value: object) -> np.ndarray:
    detached = value.detach() if callable(getattr(value, "detach", None)) else value
    cpu_value = detached.cpu() if callable(getattr(detached, "cpu", None)) else detached
    return np.asarray(cpu_value)


def _as_float(value: object, field: str, index: int) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        try:
            value = value.item()  # type: ignore[union-attr]
        except AttributeError as exc:
            raise SegmentationInferenceError(
                f"detection {index} field {field} is not numeric"
            ) from exc
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SegmentationInferenceError(f"detection {index} field {field} is not numeric") from exc
    if not math.isfinite(result):
        raise SegmentationInferenceError(f"detection {index} field {field} is not finite")
    return result


def _record_from_raw(raw: object, index: int) -> GroundingDetection:
    if isinstance(raw, GroundingDetection):
        values: tuple[object, object, object] = (raw.box, raw.score, raw.phrase)
    elif isinstance(raw, Mapping):
        try:
            values = (raw["box"], raw["score"], raw.get("phrase", ""))
        except KeyError as exc:
            raise SegmentationInferenceError(
                f"detection {index} is missing {exc.args[0]}"
            ) from exc
    else:
        try:
            values = (
                getattr(raw, "box"),
                getattr(raw, "score"),
                getattr(raw, "phrase", ""),
            )
        except AttributeError as exc:
            raise SegmentationInferenceError(
                f"detection {index} has an invalid record shape"
            ) from exc

    box_array = np.asarray(values[0])
    if box_array.ndim != 1 or box_array.size != 4:
        raise SegmentationInferenceError(f"detection {index} box must contain four values")
    box = tuple(_as_float(item, "box", index) for item in box_array.tolist())
    score = _as_float(values[1], "score", index)
    if not 0.0 <= score <= 1.0:
        raise SegmentationInferenceError(f"detection {index} score must be in [0, 1]")
    phrase = values[2]
    if not isinstance(phrase, str):
        raise SegmentationInferenceError(f"detection {index} phrase must be a string")
    return GroundingDetection(box=box, score=score, phrase=phrase)


class _GroundingDinoMobileSAMBackend:
    """CPU adapter using local Transformers GroundingDINO and MobileSAM files."""

    def __init__(self) -> None:
        dino_path = _resolve_path(DINO_MODEL_ENV, DEFAULT_DINO_MODEL_PATH)
        sam_path = _resolve_path(MOBILE_SAM_CHECKPOINT_ENV, DEFAULT_MOBILE_SAM_CHECKPOINT)
        if not dino_path.is_dir():
            raise SegmentationUnavailableError(
                f"GroundingDINO model directory was not found at {dino_path}. "
                f"Set {DINO_MODEL_ENV} to a verified local snapshot."
            )
        if not sam_path.is_file():
            raise SegmentationUnavailableError(
                f"MobileSAM checkpoint was not found at {sam_path}. "
                f"Set {MOBILE_SAM_CHECKPOINT_ENV} to a verified .pt checkpoint."
            )

        try:
            import torch
            from mobile_sam import SamPredictor, sam_model_registry
            from PIL import Image
            from transformers import (
                AutoConfig,
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )
        except Exception as exc:  # pragma: no cover - optional runtime
            raise SegmentationUnavailableError(
                "Semantic segmentation requires torch, transformers and mobile_sam"
            ) from exc

        try:
            self._torch = torch
            self._image_type = Image
            self._device = torch.device("cpu")
            self._processor = AutoProcessor.from_pretrained(
                str(dino_path), local_files_only=True
            )
            dino_config = AutoConfig.from_pretrained(str(dino_path), local_files_only=True)
            dino_config.disable_custom_kernels = True
            self._dino = AutoModelForZeroShotObjectDetection.from_pretrained(
                str(dino_path), config=dino_config, local_files_only=True
            )
            self._dino.to(self._device)
            self._dino.eval()

            mobile_sam = sam_model_registry["vit_t"](checkpoint=str(sam_path))
            mobile_sam.to(device=self._device)
            mobile_sam.eval()
            self._predictor = SamPredictor(mobile_sam)
        except Exception as exc:  # pragma: no cover - optional runtime
            raise SegmentationUnavailableError(
                f"Could not initialize local GroundingDINO/MobileSAM assets: {dino_path}, {sam_path}"
            ) from exc

    def detect(self, image: np.ndarray, prompt: str) -> list[GroundingDetection]:
        try:
            pil_image = self._image_type.fromarray(image, mode="RGB")
            inputs = self._processor(images=pil_image, text=[prompt], return_tensors="pt")
            inputs = {
                key: value.to(self._device) if callable(getattr(value, "to", None)) else value
                for key, value in inputs.items()
            }
            with self._torch.inference_mode():
                outputs = self._dino(**inputs)
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs.get("input_ids"),
                threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                target_sizes=[image.shape[:2]],
                text_labels=[[prompt]],
            )
            if not isinstance(results, list) or len(results) != 1:
                raise SegmentationInferenceError(
                    "GroundingDINO post-processing returned an invalid batch"
                )
            result = results[0]
            boxes = _as_numpy(result.get("boxes", []))
            scores = _as_numpy(result.get("scores", []))
            phrases = result.get("text_labels", result.get("labels", []))
            if boxes.size == 0:
                return []
            if boxes.ndim != 2 or boxes.shape[1] != 4 or scores.ndim != 1:
                raise SegmentationInferenceError("GroundingDINO output has invalid box/score shapes")
            if len(boxes) != len(scores) or len(phrases) != len(scores):
                raise SegmentationInferenceError("GroundingDINO output fields have different lengths")
            records: list[GroundingDetection] = []
            for index, (box, score, phrase) in enumerate(zip(boxes, scores, phrases)):
                if not isinstance(phrase, str):
                    phrase = str(phrase)
                records.append(
                    GroundingDetection(
                        box=tuple(_as_float(item, "box", index) for item in box.tolist()),
                        score=_as_float(score, "score", index),
                        phrase=phrase,
                    )
                )
            return records
        except SegmentationInferenceError:
            raise
        except Exception as exc:  # pragma: no cover - optional runtime
            raise SegmentationInferenceError("GroundingDINO inference failed") from exc

    def set_image(self, image: np.ndarray) -> None:
        try:
            self._predictor.set_image(image)
        except Exception as exc:  # pragma: no cover - optional runtime
            raise SegmentationInferenceError("MobileSAM set_image failed") from exc

    def predict(self, box: tuple[float, float, float, float]) -> np.ndarray:
        try:
            masks, _, _ = self._predictor.predict(
                box=np.asarray(box, dtype=np.float32),
                multimask_output=False,
            )
            masks_array = np.asarray(masks)
            if masks_array.ndim != 3 or masks_array.shape[0] != 1:
                raise SegmentationInferenceError("MobileSAM returned an invalid mask batch")
            return masks_array[0]
        except SegmentationInferenceError:
            raise
        except Exception as exc:  # pragma: no cover - optional runtime
            raise SegmentationInferenceError("MobileSAM prediction failed") from exc

    def reset_image(self) -> None:
        try:
            self._predictor.reset_image()
        except Exception as exc:  # pragma: no cover - optional runtime
            raise SegmentationInferenceError("MobileSAM reset_image failed") from exc

    def close(self) -> None:
        # MobileSAM and Torch modules do not expose a mandatory close method.
        self._predictor.reset_image()


def _create_default_backend() -> SegmentationBackend:
    return _GroundingDinoMobileSAMBackend()


_SEGMENTATION_FACTORY: Callable[[], SegmentationBackend] = _create_default_backend
_SEGMENTATION_CACHE: SegmentationBackend | None = None
_SEGMENTATION_CACHE_KEY: int | None = None
_SEGMENTATION_LOCK = threading.RLock()


def _get_backend_locked() -> SegmentationBackend:
    """Return the cached backend; caller must hold ``_SEGMENTATION_LOCK``."""
    global _SEGMENTATION_CACHE, _SEGMENTATION_CACHE_KEY
    cache_key = id(_SEGMENTATION_FACTORY)
    if _SEGMENTATION_CACHE is not None and _SEGMENTATION_CACHE_KEY == cache_key:
        return _SEGMENTATION_CACHE
    if _SEGMENTATION_CACHE is not None:
        _SEGMENTATION_CACHE.close()
        _SEGMENTATION_CACHE = None
        _SEGMENTATION_CACHE_KEY = None
    try:
        backend = _SEGMENTATION_FACTORY()
    except SegmentationUnavailableError:
        raise
    except Exception as exc:
        raise SegmentationUnavailableError("Could not initialize semantic segmentation backend") from exc
    for method in ("detect", "set_image", "predict", "reset_image", "close"):
        if not callable(getattr(backend, method, None)):
            raise SegmentationUnavailableError(
                f"semantic backend must provide callable {method}()"
            )
    _SEGMENTATION_CACHE = backend
    _SEGMENTATION_CACHE_KEY = cache_key
    return backend


def close_segmentation_backend() -> None:
    """Close and clear the process-local semantic backend cache."""
    global _SEGMENTATION_CACHE, _SEGMENTATION_CACHE_KEY
    with _SEGMENTATION_LOCK:
        backend = _SEGMENTATION_CACHE
        _SEGMENTATION_CACHE = None
        _SEGMENTATION_CACHE_KEY = None
        if backend is not None:
            backend.close()


def reset_segmentation_backend() -> None:
    """Alias for tests and application lifecycle hooks."""
    close_segmentation_backend()


def _close_at_exit() -> None:
    try:
        close_segmentation_backend()
    except Exception:
        pass


atexit.register(_close_at_exit)


__all__ = [
    "BOX_THRESHOLD",
    "DEFAULT_DINO_MODEL_PATH",
    "DEFAULT_MOBILE_SAM_CHECKPOINT",
    "DINO_MODEL_ENV",
    "GroundingDetection",
    "MOBILE_SAM_CHECKPOINT_ENV",
    "NMS_IOU_THRESHOLD",
    "SegmentationInferenceError",
    "SegmentationUnavailableError",
    "TEXT_THRESHOLD",
    "close_segmentation_backend",
    "reset_segmentation_backend",
]
