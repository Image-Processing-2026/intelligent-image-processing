"""Prompt-based semantic region segmentation.

Exact geometric commands stay dependency-free. Semantic prompts use the lazy
GroundingDINO + MobileSAM adapter and never fall back to a full-image or
quadrant mask when optional model assets are unavailable.
"""

from __future__ import annotations

import inspect
import math
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

import numpy as np

from .mask_utils import create_soft_mask
from .segmentation_backend import (
    _SEGMENTATION_LOCK,
    BOX_THRESHOLD,
    NMS_IOU_THRESHOLD,
    GroundingDetection,
    SegmentationInferenceError,
    SegmentationUnavailableError,
    _get_backend_locked,
)
from .spatial import _validate_feather_radius, create_quadrant_mask

MAX_PROMPT_TOKENS = 256
_FULL_COMMANDS = frozenset(("full", "all", "toàn", "toàn bộ", "full_image"))
_QUADRANT_COMMANDS = {
    "top": "top",
    "bottom": "bottom",
    "left": "left",
    "right": "right",
    "center": "center",
    "giữa": "center",
}
_PROMPT_ALIASES = {
    "người": "person",
    "bầu trời": "sky",
    "trời": "sky",
    "mèo": "cat",
    "chó": "dog",
}


@dataclass(frozen=True, slots=True)
class PromptSegmentationConfig:
    """Validated semantic thresholds, kept explicit in each result's metadata."""

    box_threshold: float = BOX_THRESHOLD
    text_threshold: float = 0.25
    nms_iou_threshold: float = NMS_IOU_THRESHOLD


@dataclass(frozen=True, slots=True)
class PromptSegmentationResult:
    """Hard per-instance masks plus provenance before union/feathering."""

    original_prompt: str
    model_prompt: str
    detections: tuple[GroundingDetection, ...]
    instance_masks: tuple[np.ndarray, ...]

    @property
    def union_hard_mask(self) -> np.ndarray:
        if not self.instance_masks:
            raise ValueError("an empty result has no implicit image shape")
        return np.logical_or.reduce(self.instance_masks)


def _validate_config(config: PromptSegmentationConfig | None) -> PromptSegmentationConfig:
    if config is None:
        return PromptSegmentationConfig()
    if not isinstance(config, PromptSegmentationConfig):
        raise TypeError("config must be a PromptSegmentationConfig")
    values = (
        ("box_threshold", config.box_threshold),
        ("text_threshold", config.text_threshold),
        ("nms_iou_threshold", config.nms_iou_threshold),
    )
    for name, raw_value in values:
        if isinstance(raw_value, (bool, np.bool_)) or not isinstance(raw_value, Real):
            raise TypeError(f"config.{name} must be a real number in [0, 1]")
        if not math.isfinite(float(raw_value)) or not 0.0 <= float(raw_value) <= 1.0:
            raise ValueError(f"config.{name} must be finite and in [0, 1]")
    return PromptSegmentationConfig(
        box_threshold=float(config.box_threshold),
        text_threshold=float(config.text_threshold),
        nms_iou_threshold=float(config.nms_iou_threshold),
    )


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


def normalize_prompt(text_prompt: object) -> str:
    """Normalize a prompt without translating or dropping meaningful words."""
    if not isinstance(text_prompt, str):
        raise TypeError("text_prompt must be a string")
    normalized = unicodedata.normalize("NFC", text_prompt)
    normalized = " ".join(normalized.split()).casefold()
    if not normalized:
        raise ValueError("text_prompt must not be empty after normalization")
    token_count = len(re.findall(r"\w+|[^\w\s]", normalized, flags=re.UNICODE))
    if token_count > MAX_PROMPT_TOKENS:
        raise ValueError(
            f"text_prompt exceeds the locked model limit of {MAX_PROMPT_TOKENS} tokens"
        )
    return normalized


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
        box, score, phrase = raw.box, raw.score, raw.phrase
    elif isinstance(raw, Mapping):
        try:
            box = raw["box"]
            score = raw["score"]
            phrase = raw.get("phrase", "")
        except KeyError as exc:
            raise SegmentationInferenceError(
                f"detection {index} is missing {exc.args[0]}"
            ) from exc
    else:
        try:
            box = getattr(raw, "box")
            score = getattr(raw, "score")
            phrase = getattr(raw, "phrase", "")
        except AttributeError as exc:
            raise SegmentationInferenceError(
                f"detection {index} has an invalid record shape"
            ) from exc
    box_array = np.asarray(box)
    if box_array.ndim != 1 or box_array.size != 4:
        raise SegmentationInferenceError(f"detection {index} box must contain four values")
    normalized_box = tuple(_as_float(value, "box", index) for value in box_array.tolist())
    normalized_score = _as_float(score, "score", index)
    if not 0.0 <= normalized_score <= 1.0:
        raise SegmentationInferenceError(f"detection {index} score must be in [0, 1]")
    if not isinstance(phrase, str):
        raise SegmentationInferenceError(f"detection {index} phrase must be a string")
    return GroundingDetection(normalized_box, normalized_score, phrase)


def _clip_detection(
    detection: GroundingDetection, image_shape: tuple[int, int], index: int
) -> GroundingDetection | None:
    height, width = image_shape
    xmin, ymin, xmax, ymax = detection.box
    if any(not math.isfinite(value) for value in detection.box):
        raise SegmentationInferenceError(f"detection {index} bbox is not finite")
    clipped = (
        min(float(width), max(0.0, xmin)),
        min(float(height), max(0.0, ymin)),
        min(float(width), max(0.0, xmax)),
        min(float(height), max(0.0, ymax)),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return GroundingDetection(clipped, detection.score, detection.phrase.strip())


def _iou(first: GroundingDetection, second: GroundingDetection) -> float:
    ax0, ay0, ax1, ay1 = first.box
    bx0, by0, bx1, by1 = second.box
    intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(
        0.0, min(ay1, by1) - max(ay0, by0)
    )
    first_area = (ax1 - ax0) * (ay1 - ay0)
    second_area = (bx1 - bx0) * (by1 - by0)
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _prepare_detections(
    raw: object,
    image_shape: tuple[int, int],
    config: PromptSegmentationConfig,
) -> list[GroundingDetection]:
    if raw is None or isinstance(raw, (str, bytes)):
        raise SegmentationInferenceError("GroundingDINO backend returned a non-sequence")
    try:
        records = list(raw)
    except TypeError as exc:
        raise SegmentationInferenceError("GroundingDINO backend returned a non-iterable") from exc

    candidates: list[GroundingDetection] = []
    for index, item in enumerate(records):
        detection = _record_from_raw(item, index)
        if detection.score < config.box_threshold:
            continue
        if not detection.phrase.strip():
            continue
        clipped = _clip_detection(detection, image_shape, index)
        if clipped is not None:
            candidates.append(clipped)

    kept: list[GroundingDetection] = []
    for detection in sorted(
        candidates,
        key=lambda item: (-item.score, item.box[1], item.box[0], item.box[3], item.box[2]),
    ):
        # Phrases can originate from a multi-phrase GroundingDINO prompt.  A
        # person box and a nearby bicycle box must not suppress each other.
        same_phrase = (
            previous
            for previous in kept
            if previous.phrase.casefold().strip() == detection.phrase.casefold().strip()
        )
        if all(_iou(detection, previous) <= config.nms_iou_threshold for previous in same_phrase):
            kept.append(detection)
    return kept


def _mask_from_backend(raw: object, image_shape: tuple[int, int], index: int) -> np.ndarray:
    value = raw[0] if isinstance(raw, (tuple, list)) and raw else raw
    if not isinstance(value, np.ndarray):
        raise SegmentationInferenceError(f"MobileSAM mask {index} is not a NumPy array")
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.shape != image_shape:
        raise SegmentationInferenceError(
            f"MobileSAM mask {index} has shape {value.shape}, expected {image_shape}"
        )
    if value.dtype == np.dtype(np.bool_):
        return value.astype(bool, copy=True)
    if value.dtype == np.dtype(np.uint8):
        unique = np.unique(value)
        if not np.all(np.isin(unique, (0, 1, 255))):
            raise SegmentationInferenceError(f"MobileSAM mask {index} contains non-binary uint8 values")
        return value != 0
    raise SegmentationInferenceError(
        f"MobileSAM mask {index} must be boolean or binary uint8, not {value.dtype}"
    )


def _semantic_prompt(normalized_prompt: str) -> str:
    prompt = _PROMPT_ALIASES.get(normalized_prompt, normalized_prompt)
    return prompt if prompt.endswith(".") else f"{prompt}."


def _detect_with_config(
    backend: object,
    image: np.ndarray,
    prompt: str,
    config: PromptSegmentationConfig,
) -> object:
    """Call newer backends with thresholds while retaining the old test seam."""
    detect = getattr(backend, "detect", None)
    if not callable(detect):
        raise SegmentationInferenceError("semantic backend must provide detect()")
    try:
        parameters = inspect.signature(detect).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    supports_keywords = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)
    names = {parameter.name for parameter in parameters}
    if supports_keywords or {"box_threshold", "text_threshold"} <= names:
        return detect(
            image,
            prompt,
            box_threshold=config.box_threshold,
            text_threshold=config.text_threshold,
        )
    return detect(image, prompt)


def resolve_prompt_instances(
    image: np.ndarray,
    text_prompt: str,
    *,
    config: PromptSegmentationConfig | None = None,
) -> PromptSegmentationResult:
    """Resolve one semantic prompt to hard masks before union/feathering.

    ``segment_by_prompt`` remains the compatible convenience wrapper.  New
    callers should use this function when they need stable instance selection,
    detection boxes, scores and phrases.
    """
    validated_image = _validate_image(image)
    normalized_prompt = normalize_prompt(text_prompt)
    validated_config = _validate_config(config)
    image_shape = validated_image.shape[:2]

    if normalized_prompt in _FULL_COMMANDS:
        return PromptSegmentationResult(
            normalized_prompt,
            normalized_prompt,
            (),
            (np.ones(image_shape, dtype=bool),),
        )
    if normalized_prompt in _QUADRANT_COMMANDS:
        hard = create_quadrant_mask(image_shape, _QUADRANT_COMMANDS[normalized_prompt], 0) > 0
        return PromptSegmentationResult(normalized_prompt, normalized_prompt, (), (hard,))

    semantic_prompt = _semantic_prompt(normalized_prompt)
    contiguous_image = np.ascontiguousarray(validated_image)
    if not contiguous_image.flags.writeable:
        contiguous_image = contiguous_image.copy()
    with _SEGMENTATION_LOCK:
        backend = _get_backend_locked()
        try:
            raw_detections = _detect_with_config(
                backend, contiguous_image, semantic_prompt, validated_config
            )
            detections = _prepare_detections(raw_detections, image_shape, validated_config)
            if not detections:
                return PromptSegmentationResult(normalized_prompt, semantic_prompt, (), ())

            instances: list[np.ndarray] = []
            primary_error: BaseException | None = None
            try:
                backend.set_image(contiguous_image)
                for index, detection in enumerate(detections):
                    instances.append(
                        _mask_from_backend(backend.predict(detection.box), image_shape, index)
                    )
            except BaseException as exc:
                primary_error = exc
                raise
            finally:
                try:
                    backend.reset_image()
                except Exception:
                    if primary_error is None:
                        raise
            return PromptSegmentationResult(
                normalized_prompt,
                semantic_prompt,
                tuple(detections),
                tuple(instances),
            )
        except (SegmentationUnavailableError, SegmentationInferenceError):
            raise
        except Exception as exc:
            raise SegmentationInferenceError("Prompt segmentation inference failed") from exc


def segment_by_prompt(
    image: np.ndarray,
    text_prompt: str,
    feather_radius: int = 15,
    *,
    config: PromptSegmentationConfig | None = None,
) -> np.ndarray:
    """Return one feathered union mask for a semantic or exact region prompt.

    Exact geometric commands are handled without optional model dependencies.
    Every other prompt is sent to the cached GroundingDINO + MobileSAM
    backend. A successful semantic inference with no retained object returns a
    zero mask; missing assets and inference failures raise explicit errors.
    """
    radius = _validate_feather_radius(feather_radius)
    result = resolve_prompt_instances(image, text_prompt, config=config)
    if not result.instance_masks:
        return np.zeros(_validate_image(image).shape[:2], dtype=np.float32)
    return create_soft_mask(result.union_hard_mask.astype(np.uint8), feather_radius=radius)


__all__ = [
    "MAX_PROMPT_TOKENS",
    "PromptSegmentationConfig",
    "PromptSegmentationResult",
    "normalize_prompt",
    "resolve_prompt_instances",
    "segment_by_prompt",
]
