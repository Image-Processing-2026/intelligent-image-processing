"""Controller/facade for resolving Module 2 regions.

The low-level mask functions remain useful on their own, while this module
provides one normalized request/result contract for callers outside the
region engine.  It deliberately does not blend images; Module 3 owns that
operation and receives the resolved mask exactly once.
"""

from __future__ import annotations

import importlib.util
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from .detector import segment_by_prompt
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
    DEFAULT_DINO_MODEL_PATH,
    DEFAULT_MOBILE_SAM_CHECKPOINT,
    DINO_MODEL_ENV,
    MOBILE_SAM_CHECKPOINT_ENV,
    SegmentationInferenceError,
    SegmentationUnavailableError,
)
from .spatial import create_bbox_mask, create_quadrant_mask

RegionKind = Literal["full", "bbox", "spatial", "face", "semantic", "binary_mask"]
RegionStatus = Literal["ok", "empty"]
MergePolicy = Literal["max"]

_VALID_KINDS = frozenset(("full", "bbox", "spatial", "face", "semantic", "binary_mask"))
_SPATIAL_NAMES = frozenset(("top", "bottom", "left", "right", "center", "giữa"))
_FACE_NAMES = frozenset(("face", "faces", "khuôn mặt", "khuôn mặt người"))
_FULL_NAMES = frozenset(("full", "all", "toàn", "toàn bộ", "full_image"))
_INFERENCE_VERIFIED = {"face": False, "semantic": False}


class RegionEngineError(RuntimeError):
    """Base class for controller-level failures."""


class InvalidRegionRequestError(RegionEngineError, ValueError):
    """Raised when a normalized region request violates its contract."""


class RegionBackendUnavailableError(RegionEngineError):
    """Raised when a requested AI backend or its assets are unavailable."""


class RegionInferenceError(RegionEngineError):
    """Raised when a requested AI backend fails during inference."""


@dataclass(frozen=True, slots=True)
class RegionRequest:
    """Normalized request accepted by :func:`resolve_region`.

    ``bbox`` uses half-open pixel coordinates ``(xmin, ymin, xmax, ymax)``.
    ``binary_mask`` is kept as an ndarray because it is an in-process API;
    serialized callers should provide it separately and construct the request
    after decoding the payload.
    """

    kind: RegionKind
    bbox: tuple[int, int, int, int] | None = None
    quadrant: str | None = None
    prompt: str | None = None
    binary_mask: np.ndarray | None = None
    feather_radius: int = 15
    expand_ratio: float = 0.15
    merge_policy: MergePolicy = "max"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RegionRequest":
        """Build a request from a runtime mapping or plan-like object."""
        payload = dict(value)
        raw_kind = payload.get("kind", payload.get("region_type"))
        target = payload.get("prompt", payload.get("text_prompt", payload.get("target_prompt")))
        if raw_kind is None:
            raw_kind = _infer_kind(target, payload)

        if raw_kind == "semantic" and isinstance(target, str):
            prompt = payload.get("prompt", payload.get("text_prompt", target))
        else:
            prompt = payload.get("prompt", payload.get("text_prompt"))

        raw_bbox = payload.get("bbox")
        bbox = None if raw_bbox is None else _coerce_int_vector(raw_bbox, 4, "bbox")
        binary_mask = payload.get("binary_mask", payload.get("mask"))
        return cls(
            kind=raw_kind,
            bbox=bbox,
            quadrant=payload.get("quadrant", target if raw_kind == "spatial" else None),
            prompt=prompt,
            binary_mask=binary_mask,
            feather_radius=payload.get("feather_radius", 15),
            expand_ratio=payload.get("expand_ratio", 0.15),
            merge_policy=payload.get("merge_policy", "max"),
        )


@dataclass(frozen=True, slots=True)
class RegionResult:
    """Resolved region mask and provenance metadata."""

    status: RegionStatus
    mask: np.ndarray
    instance_masks: tuple[np.ndarray, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return self.status == "empty"


def _infer_kind(target: object, payload: Mapping[str, Any]) -> str:
    if payload.get("bbox") is not None:
        return "bbox"
    if payload.get("binary_mask", payload.get("mask")) is not None:
        return "binary_mask"
    if not isinstance(target, str):
        raise InvalidRegionRequestError("request.kind is required when target_prompt is absent")
    normalized = " ".join(target.split()).casefold()
    if normalized in _FULL_NAMES:
        return "full"
    if normalized in _SPATIAL_NAMES:
        return "spatial"
    if normalized in _FACE_NAMES:
        return "face"
    return "semantic"


def _coerce_int_vector(value: object, size: int, name: str) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list, np.ndarray)):
        raise InvalidRegionRequestError(f"{name} must contain exactly {size} integers")
    values = value.tolist() if isinstance(value, np.ndarray) else list(value)
    if len(values) != size:
        raise InvalidRegionRequestError(f"{name} must contain exactly {size} integers")
    result: list[int] = []
    for item in values:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
            raise InvalidRegionRequestError(f"{name} must contain exactly {size} integers")
        result.append(int(item))
    return tuple(result)


def _validate_image(image: object) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise InvalidRegionRequestError("image must be a NumPy array")
    if image.dtype != np.dtype(np.uint8):
        raise InvalidRegionRequestError("image dtype must be uint8")
    if image.ndim != 3 or image.shape[2] != 3:
        raise InvalidRegionRequestError("image must have shape (H, W, 3) in RGB order")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise InvalidRegionRequestError("image must be non-empty")
    return image


def _validate_common(request: RegionRequest) -> tuple[int, float, str]:
    if not isinstance(request.kind, str) or request.kind not in _VALID_KINDS:
        raise InvalidRegionRequestError(f"unsupported region kind: {request.kind!r}")
    if request.merge_policy != "max":
        raise InvalidRegionRequestError("merge_policy currently supports only 'max'")
    if isinstance(request.feather_radius, (bool, np.bool_)) or not isinstance(
        request.feather_radius, (int, np.integer)
    ):
        raise InvalidRegionRequestError("feather_radius must be a non-negative integer")
    radius = int(request.feather_radius)
    if radius < 0:
        raise InvalidRegionRequestError("feather_radius must be non-negative")
    if isinstance(request.expand_ratio, (bool, np.bool_)) or not isinstance(
        request.expand_ratio, (int, float, np.integer, np.floating)
    ):
        raise InvalidRegionRequestError("expand_ratio must be a finite number in [0, 1]")
    ratio = float(request.expand_ratio)
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise InvalidRegionRequestError("expand_ratio must be a finite number in [0, 1]")
    return radius, ratio, request.kind


def _validate_mask(mask: object, shape: tuple[int, int], *, name: str) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        raise RegionInferenceError(f"{name} must be a NumPy array")
    if mask.dtype not in (np.dtype(np.bool_), np.dtype(np.uint8), np.dtype(np.float32)):
        raise RegionInferenceError(f"{name} has unsupported dtype {mask.dtype}")
    if mask.shape != shape:
        raise RegionInferenceError(f"{name} has shape {mask.shape}, expected {shape}")
    if mask.dtype == np.dtype(np.float32):
        if not np.isfinite(mask).all() or np.any(mask < 0.0) or np.any(mask > 1.0):
            raise RegionInferenceError(f"{name} must be finite and bounded in [0, 1]")
        return mask.astype(np.float32, copy=True)
    return mask.astype(bool, copy=True).astype(np.float32)


def _empty_result(shape: tuple[int, int], backend: str, metadata: dict[str, Any]) -> RegionResult:
    details = {"backend": backend, **metadata}
    return RegionResult("empty", np.zeros(shape, dtype=np.float32), (), details)


def _ok_result(
    mask: np.ndarray,
    backend: str,
    metadata: dict[str, Any],
    instance_masks: tuple[np.ndarray, ...] = (),
) -> RegionResult:
    validated = _validate_mask(mask, mask.shape, name="resolved mask")
    details = {"backend": backend, **metadata}
    status: RegionStatus = "empty" if not np.any(validated) else "ok"
    return RegionResult(status, validated, instance_masks, details)


def _resolve_request(request: RegionRequest | Mapping[str, Any]) -> RegionRequest:
    if isinstance(request, RegionRequest):
        return request
    if isinstance(request, Mapping):
        try:
            return RegionRequest.from_mapping(request)
        except RegionEngineError:
            raise
        except (TypeError, ValueError) as exc:
            raise InvalidRegionRequestError("request mapping contains invalid values") from exc
    raise InvalidRegionRequestError("request must be a RegionRequest or mapping")


def resolve_region(
    image: np.ndarray,
    request: RegionRequest | Mapping[str, Any],
    *,
    _face_resolver: Callable[..., list[np.ndarray]] | None = None,
    _semantic_resolver: Callable[..., np.ndarray] | None = None,
) -> RegionResult:
    """Resolve one normalized request into a concrete Module 2 mask.

    The private resolver hooks are intentionally available for integration
    tests and application adapters; production callers should use the default
    detector functions.  This function never calls ``blend_regions``.
    """
    validated_image = _validate_image(image)
    normalized = _resolve_request(request)
    radius, ratio, kind = _validate_common(normalized)
    shape = validated_image.shape[:2]

    try:
        if kind == "full":
            return _ok_result(
                np.ones(shape, dtype=np.float32),
                "geometry",
                {"kind": kind, "feather_radius": radius},
            )

        if kind == "bbox":
            if normalized.bbox is None:
                raise InvalidRegionRequestError("bbox is required for region kind 'bbox'")
            mask = create_bbox_mask(shape, normalized.bbox, feather_radius=radius)
            return _ok_result(
                mask,
                "geometry",
                {"kind": kind, "bbox": normalized.bbox, "feather_radius": radius},
            )

        if kind == "spatial":
            if not isinstance(normalized.quadrant, str):
                raise InvalidRegionRequestError("quadrant is required for region kind 'spatial'")
            mask = create_quadrant_mask(shape, normalized.quadrant, feather_radius=radius)
            return _ok_result(
                mask,
                "geometry",
                {"kind": kind, "quadrant": normalized.quadrant, "feather_radius": radius},
            )

        if kind == "binary_mask":
            if not isinstance(normalized.binary_mask, np.ndarray):
                raise InvalidRegionRequestError(
                    "binary_mask is required for region kind 'binary_mask'"
                )
            if normalized.binary_mask.shape != shape:
                raise InvalidRegionRequestError(
                    f"binary_mask has shape {normalized.binary_mask.shape}, expected {shape}"
                )
            if normalized.binary_mask.dtype == np.dtype(np.bool_):
                binary = normalized.binary_mask.copy()
            elif normalized.binary_mask.dtype == np.dtype(np.uint8):
                values = np.unique(normalized.binary_mask)
                if not np.all(np.isin(values, (0, 1, 255))):
                    raise InvalidRegionRequestError("binary_mask uint8 values must be binary")
                binary = normalized.binary_mask != 0
            else:
                raise InvalidRegionRequestError("binary_mask dtype must be bool or uint8")
            mask = create_soft_mask(binary, feather_radius=radius)
            return _ok_result(
                mask,
                "geometry",
                {"kind": kind, "feather_radius": radius},
            )

        if kind == "face":
            resolver = _face_resolver or detect_faces
            masks = resolver(validated_image, feather_radius=radius, expand_ratio=ratio)
            normalized_masks = tuple(
                _validate_mask(mask, shape, name=f"face mask {index}")
                for index, mask in enumerate(masks)
            )
            _INFERENCE_VERIFIED["face"] = True
            if not normalized_masks:
                return _empty_result(
                    shape,
                    "mediapipe",
                    {
                        "kind": kind,
                        "count": 0,
                        "feather_radius": radius,
                        "expand_ratio": ratio,
                        "merge_policy": normalized.merge_policy,
                    },
                )
            merged = np.maximum.reduce(normalized_masks)
            return _ok_result(
                merged,
                "mediapipe",
                {
                    "kind": kind,
                    "count": len(normalized_masks),
                    "feather_radius": radius,
                    "expand_ratio": ratio,
                    "merge_policy": normalized.merge_policy,
                },
                normalized_masks,
            )

        if kind == "semantic":
            if not isinstance(normalized.prompt, str) or not normalized.prompt.strip():
                raise InvalidRegionRequestError("prompt is required for region kind 'semantic'")
            resolver = _semantic_resolver or segment_by_prompt
            mask = _validate_mask(
                resolver(validated_image, normalized.prompt, feather_radius=radius),
                shape,
                name="semantic mask",
            )
            _INFERENCE_VERIFIED["semantic"] = True
            return _ok_result(
                mask,
                "groundingdino+mobilesam",
                {"kind": kind, "prompt": normalized.prompt, "feather_radius": radius},
            )
    except InvalidRegionRequestError:
        raise
    except (FaceDetectorUnavailableError, SegmentationUnavailableError) as exc:
        raise RegionBackendUnavailableError(str(exc)) from exc
    except (FaceDetectionError, SegmentationInferenceError) as exc:
        raise RegionInferenceError(str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise InvalidRegionRequestError("region request could not be resolved") from exc
    except RegionEngineError:
        raise

    raise InvalidRegionRequestError(f"unsupported region kind: {kind!r}")


def _resolved_path(environment_name: str, default: Path) -> Path:
    configured = os.environ.get(environment_name, "").strip()
    path = Path(configured) if configured else default
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.expanduser().resolve()


def _dependency_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def capabilities() -> dict[str, Any]:
    """Report static asset/dependency readiness without loading model weights."""
    face_path = _resolved_path(FACE_MODEL_ENV, DEFAULT_FACE_MODEL_PATH)
    dino_path = _resolved_path(DINO_MODEL_ENV, DEFAULT_DINO_MODEL_PATH)
    sam_path = _resolved_path(MOBILE_SAM_CHECKPOINT_ENV, DEFAULT_MOBILE_SAM_CHECKPOINT)
    face_dependencies = {"mediapipe": _dependency_available("mediapipe")}
    semantic_dependencies = {
        "torch": _dependency_available("torch"),
        "transformers": _dependency_available("transformers"),
        "mobile_sam": _dependency_available("mobile_sam"),
    }
    face_assets = {"model": face_path.is_file()}
    semantic_assets = {"grounding_dino": dino_path.is_dir(), "mobile_sam": sam_path.is_file()}
    return {
        "geometric": {
            "ready": True,
            "functions": ["full", "bbox", "spatial", "binary_mask"],
        },
        "face": {
            "ready": all(face_assets.values()) and all(face_dependencies.values()),
            "assets_present": face_assets,
            "dependencies_available": face_dependencies,
            "inference_verified": _INFERENCE_VERIFIED["face"],
            "model_path": str(face_path),
        },
        "semantic": {
            "ready": all(semantic_assets.values()) and all(semantic_dependencies.values()),
            "assets_present": semantic_assets,
            "dependencies_available": semantic_dependencies,
            "inference_verified": _INFERENCE_VERIFIED["semantic"],
            "dino_model_path": str(dino_path),
            "mobile_sam_checkpoint": str(sam_path),
        },
    }


__all__ = [
    "InvalidRegionRequestError",
    "RegionBackendUnavailableError",
    "RegionEngineError",
    "RegionInferenceError",
    "RegionRequest",
    "RegionResult",
    "capabilities",
    "resolve_region",
]
