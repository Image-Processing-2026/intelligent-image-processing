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
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from .detector import (
    PromptSegmentationConfig,
    resolve_prompt_instances,
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
from .face_landmarker import (
    DEFAULT_MODEL_PATH as DEFAULT_FACE_LANDMARKER_MODEL_PATH,
)
from .face_landmarker import (
    MODEL_PATH_ENV as FACE_LANDMARKER_MODEL_ENV,
)
from .face_landmarker import (
    FaceLandmarkerError,
    FaceLandmarkerUnavailableError,
    FaceOval,
    detect_face_ovals,
)
from .mask_utils import create_soft_mask
from .segmentation_backend import (
    DEFAULT_DINO_MODEL_PATH,
    DEFAULT_MOBILE_SAM_CHECKPOINT,
    DINO_MODEL_ENV,
    MOBILE_SAM_CHECKPOINT_ENV,
    NMS_IOU_THRESHOLD,
    TEXT_THRESHOLD,
    SegmentationInferenceError,
    SegmentationUnavailableError,
)
from .spatial import create_bbox_mask, create_quadrant_mask

RegionKind = Literal["full", "bbox", "spatial", "face", "semantic", "binary_mask"]
RegionStatus = Literal["ok", "empty"]
MergePolicy = Literal["max"]
FaceMode = Literal["bbox", "oval", "sam_refined"]
InstanceSelection = Literal["all", "largest", "index"]

_VALID_KINDS = frozenset(("full", "bbox", "spatial", "face", "semantic", "binary_mask"))
_SPATIAL_NAMES = frozenset(("top", "bottom", "left", "right", "center", "giữa"))
_FACE_NAMES = frozenset(("face", "faces", "khuôn mặt", "khuôn mặt người"))
_FULL_NAMES = frozenset(("full", "all", "toàn", "toàn bộ", "full_image"))
# Values are path/config keys set only by the production resolver.  Injected
# fake resolvers are useful test seams but cannot certify a real model.
_INFERENCE_VERIFIED: dict[str, tuple[str, ...] | None] = {
    "face_bbox": None,
    "face_oval": None,
    "semantic": None,
}


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
    face_mode: FaceMode = "bbox"
    num_faces: int = 4
    instance_selection: InstanceSelection = "all"
    instance_index: int | None = None
    box_threshold: float = 0.35
    text_threshold: float = TEXT_THRESHOLD
    nms_iou_threshold: float = NMS_IOU_THRESHOLD

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
            face_mode=payload.get("face_mode", "bbox"),
            num_faces=payload.get("num_faces", 4),
            instance_selection=payload.get("instance_selection", payload.get("selection", "all")),
            instance_index=payload.get("instance_index"),
            box_threshold=payload.get("box_threshold", 0.35),
            text_threshold=payload.get("text_threshold", TEXT_THRESHOLD),
            nms_iou_threshold=payload.get("nms_iou_threshold", NMS_IOU_THRESHOLD),
        )


@dataclass(frozen=True, slots=True)
class RegionResult:
    """Resolved region mask and provenance metadata."""

    status: RegionStatus
    mask: np.ndarray
    instance_masks: tuple[np.ndarray, ...] = ()
    contours: tuple[np.ndarray, ...] = ()
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


def _validate_face_options(request: RegionRequest) -> tuple[str, int, str, int | None]:
    if request.face_mode not in ("bbox", "oval", "sam_refined"):
        raise InvalidRegionRequestError("face_mode must be 'bbox', 'oval', or 'sam_refined'")
    if isinstance(request.num_faces, (bool, np.bool_)) or not isinstance(
        request.num_faces, (int, np.integer)
    ):
        raise InvalidRegionRequestError("num_faces must be a positive integer")
    num_faces = int(request.num_faces)
    if num_faces <= 0:
        raise InvalidRegionRequestError("num_faces must be a positive integer")
    selection, index = _validate_instance_selection(request)
    return request.face_mode, num_faces, selection, index


def _validate_instance_selection(request: RegionRequest) -> tuple[str, int | None]:
    if request.instance_selection not in ("all", "largest", "index"):
        raise InvalidRegionRequestError("instance_selection must be 'all', 'largest', or 'index'")
    index = request.instance_index
    if request.instance_selection == "index":
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise InvalidRegionRequestError("instance_index is required when instance_selection is 'index'")
        if int(index) < 0:
            raise InvalidRegionRequestError("instance_index must be non-negative")
        index = int(index)
    elif index is not None:
        raise InvalidRegionRequestError("instance_index is allowed only when instance_selection is 'index'")
    return request.instance_selection, index


def _semantic_config(request: RegionRequest) -> PromptSegmentationConfig:
    raw_values = {
        "box_threshold": request.box_threshold,
        "text_threshold": request.text_threshold,
        "nms_iou_threshold": request.nms_iou_threshold,
    }
    for name, raw_value in raw_values.items():
        if isinstance(raw_value, (bool, np.bool_)) or not isinstance(raw_value, (int, float, np.integer, np.floating)):
            raise InvalidRegionRequestError(f"{name} must be a finite number in [0, 1]")
        value = float(raw_value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise InvalidRegionRequestError(f"{name} must be a finite number in [0, 1]")
    return PromptSegmentationConfig(**{name: float(value) for name, value in raw_values.items()})


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
    return RegionResult("empty", np.zeros(shape, dtype=np.float32), (), (), details)


def _ok_result(
    mask: np.ndarray,
    backend: str,
    metadata: dict[str, Any],
    instance_masks: tuple[np.ndarray, ...] = (),
    contours: tuple[np.ndarray, ...] = (),
) -> RegionResult:
    validated = _validate_mask(mask, mask.shape, name="resolved mask")
    details = {"backend": backend, **metadata}
    status: RegionStatus = "empty" if not np.any(validated) else "ok"
    return RegionResult(status, validated, instance_masks, contours, details)


def _select_instances(
    masks: tuple[np.ndarray, ...], selection: str, index: int | None
) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
    if not masks:
        return (), ()
    if selection == "all":
        positions = tuple(range(len(masks)))
    elif selection == "largest":
        positions = (max(range(len(masks)), key=lambda position: float(masks[position].sum())),)
    else:
        assert index is not None
        if index >= len(masks):
            raise InvalidRegionRequestError(
                f"instance_index {index} is outside the {len(masks)} available instances"
            )
        positions = (index,)
    return tuple(masks[position] for position in positions), positions


def _mark_inference_verified(kind: str, *key: str) -> None:
    _INFERENCE_VERIFIED[kind] = tuple(key)


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
    _face_oval_resolver: Callable[..., list[FaceOval]] | None = None,
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
            face_mode, num_faces, selection, index = _validate_face_options(normalized)
            if face_mode == "sam_refined":
                raise RegionBackendUnavailableError(
                    "face_mode='sam_refined' is experimental and is not enabled in this build"
                )
            started = time.perf_counter()
            if face_mode == "bbox":
                resolver = _face_resolver or detect_faces
                masks = resolver(validated_image, feather_radius=radius, expand_ratio=ratio)
                normalized_masks = tuple(
                    _validate_mask(mask, shape, name=f"face mask {mask_index}")
                    for mask_index, mask in enumerate(masks)
                )
                selected_masks, positions = _select_instances(normalized_masks, selection, index)
                if _face_resolver is None:
                    _mark_inference_verified("face_bbox", str(_resolved_path(FACE_MODEL_ENV, DEFAULT_FACE_MODEL_PATH)))
                details = {
                    "kind": kind,
                    "face_mode": "bbox",
                    "count": len(normalized_masks),
                    "selected_indices": positions,
                    "instance_selection": selection,
                    "feather_radius": radius,
                    "expand_ratio": ratio,
                    "merge_policy": normalized.merge_policy,
                    "mask_kind": "soft",
                    "timing_ms": round((time.perf_counter() - started) * 1000, 3),
                }
                if not selected_masks:
                    return _empty_result(shape, "mediapipe", details)
                return _ok_result(
                    np.maximum.reduce(selected_masks),
                    "mediapipe",
                    details,
                    selected_masks,
                )

            resolver = _face_oval_resolver or detect_face_ovals
            ovals = resolver(validated_image, num_faces=num_faces)
            if ovals is None or isinstance(ovals, (str, bytes)):
                raise RegionInferenceError("face oval resolver returned a non-sequence")
            normalized_masks: list[np.ndarray] = []
            contours: list[np.ndarray] = []
            for oval_index, oval in enumerate(ovals):
                if not isinstance(oval, FaceOval):
                    raise RegionInferenceError(f"face oval {oval_index} has an invalid record type")
                hard_mask = _validate_mask(oval.mask, shape, name=f"face oval mask {oval_index}")
                if np.any((hard_mask != 0.0) & (hard_mask != 1.0)):
                    raise RegionInferenceError(f"face oval mask {oval_index} must be binary")
                contour = np.asarray(oval.contour, dtype=np.float32)
                if contour.ndim != 2 or contour.shape[1] != 2 or len(contour) < 3:
                    raise RegionInferenceError(f"face oval contour {oval_index} must have shape (N, 2), N >= 3")
                if not np.isfinite(contour).all():
                    raise RegionInferenceError(f"face oval contour {oval_index} must be finite")
                normalized_masks.append(hard_mask)
                contours.append(contour.copy())
            selected_masks, positions = _select_instances(tuple(normalized_masks), selection, index)
            selected_contours = tuple(contours[position] for position in positions)
            if _face_oval_resolver is None:
                _mark_inference_verified(
                    "face_oval",
                    str(_resolved_path(FACE_LANDMARKER_MODEL_ENV, DEFAULT_FACE_LANDMARKER_MODEL_PATH)),
                    str(num_faces),
                )
            details = {
                "kind": kind,
                "face_mode": "oval",
                "count": len(normalized_masks),
                "selected_indices": positions,
                "instance_selection": selection,
                "feather_radius": radius,
                "expand_ratio": ratio,
                "merge_policy": normalized.merge_policy,
                "mask_kind": "hard_instance_soft_union",
                "timing_ms": round((time.perf_counter() - started) * 1000, 3),
            }
            if not selected_masks:
                return _empty_result(shape, "mediapipe_face_landmarker", details)
            union = np.logical_or.reduce(tuple(mask > 0.0 for mask in selected_masks))
            soft_mask = create_soft_mask(union.astype(np.uint8), feather_radius=radius)
            return _ok_result(
                soft_mask,
                "mediapipe_face_landmarker",
                details,
                selected_masks,
                selected_contours,
            )

        if kind == "semantic":
            if not isinstance(normalized.prompt, str) or not normalized.prompt.strip():
                raise InvalidRegionRequestError("prompt is required for region kind 'semantic'")
            selection, index = _validate_instance_selection(normalized)
            config = _semantic_config(normalized)
            started = time.perf_counter()
            if _semantic_resolver is not None:
                mask = _validate_mask(
                    _semantic_resolver(validated_image, normalized.prompt, feather_radius=radius),
                    shape,
                    name="semantic mask",
                )
                return _ok_result(
                    mask,
                    "custom_semantic_resolver",
                    {
                        "kind": kind,
                        "requested_prompt": normalized.prompt,
                        "model_prompt": None,
                        "feather_radius": radius,
                        "config": {
                            "box_threshold": config.box_threshold,
                            "text_threshold": config.text_threshold,
                            "nms_iou_threshold": config.nms_iou_threshold,
                        },
                        "timing_ms": round((time.perf_counter() - started) * 1000, 3),
                    },
                )
            semantic = resolve_prompt_instances(
                validated_image,
                normalized.prompt,
                config=config,
            )
            selected_masks, positions = _select_instances(
                tuple(
                    _validate_mask(mask, shape, name=f"semantic instance mask {mask_index}")
                    for mask_index, mask in enumerate(semantic.instance_masks)
                ),
                selection,
                index,
            )
            semantic_backend = "geometry" if not semantic.detections else "groundingdino+mobilesam"
            if semantic_backend != "geometry":
                _mark_inference_verified(
                    "semantic",
                    str(_resolved_path(DINO_MODEL_ENV, DEFAULT_DINO_MODEL_PATH)),
                    str(_resolved_path(MOBILE_SAM_CHECKPOINT_ENV, DEFAULT_MOBILE_SAM_CHECKPOINT)),
                )
            details = {
                "kind": kind,
                "requested_prompt": semantic.original_prompt,
                "model_prompt": semantic.model_prompt,
                "count": len(semantic.instance_masks),
                "selected_indices": positions,
                "instance_selection": selection,
                "feather_radius": radius,
                "mask_kind": "hard_instance_soft_union",
                "config": {
                    "box_threshold": config.box_threshold,
                    "text_threshold": config.text_threshold,
                    "nms_iou_threshold": config.nms_iou_threshold,
                },
                "detections": [
                    {"bbox": detection.box, "score": detection.score, "phrase": detection.phrase}
                    for detection in semantic.detections
                ],
                "timing_ms": round((time.perf_counter() - started) * 1000, 3),
            }
            if not selected_masks:
                return _empty_result(shape, semantic_backend, details)
            union = np.logical_or.reduce(tuple(mask > 0.0 for mask in selected_masks))
            return _ok_result(
                create_soft_mask(union.astype(np.uint8), feather_radius=radius),
                semantic_backend,
                details,
                selected_masks,
            )
    except InvalidRegionRequestError:
        raise
    except (
        FaceDetectorUnavailableError,
        FaceLandmarkerUnavailableError,
        SegmentationUnavailableError,
    ) as exc:
        raise RegionBackendUnavailableError(str(exc)) from exc
    except (FaceDetectionError, FaceLandmarkerError, SegmentationInferenceError) as exc:
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
    face_landmarker_path = _resolved_path(
        FACE_LANDMARKER_MODEL_ENV, DEFAULT_FACE_LANDMARKER_MODEL_PATH
    )
    dino_path = _resolved_path(DINO_MODEL_ENV, DEFAULT_DINO_MODEL_PATH)
    sam_path = _resolved_path(MOBILE_SAM_CHECKPOINT_ENV, DEFAULT_MOBILE_SAM_CHECKPOINT)
    face_dependencies = {"mediapipe": _dependency_available("mediapipe")}
    semantic_dependencies = {
        "torch": _dependency_available("torch"),
        "transformers": _dependency_available("transformers"),
        "mobile_sam": _dependency_available("mobile_sam"),
    }
    face_assets = {"model": face_path.is_file()}
    face_oval_assets = {"model": face_landmarker_path.is_file()}
    semantic_assets = {"grounding_dino": dino_path.is_dir(), "mobile_sam": sam_path.is_file()}
    face_key = (str(face_path),)
    face_oval_key_prefix = (str(face_landmarker_path),)
    semantic_key = (str(dino_path), str(sam_path))
    face_verified = _INFERENCE_VERIFIED["face_bbox"] == face_key
    face_oval_verified = (
        _INFERENCE_VERIFIED["face_oval"] is not None
        and _INFERENCE_VERIFIED["face_oval"][:1] == face_oval_key_prefix
    )
    semantic_verified = _INFERENCE_VERIFIED["semantic"] == semantic_key
    return {
        "geometric": {
            "ready": True,
            "functions": ["full", "bbox", "spatial", "binary_mask"],
        },
        "face": {
            "ready": all(face_assets.values()) and all(face_dependencies.values()),
            "assets_present": face_assets,
            "asset_validation": {"hash_verified": False},
            "dependencies_importable": face_dependencies,
            "dependencies_available": face_dependencies,
            "model_load_verified": face_verified,
            "inference_verified": face_verified,
            "quality_passed": False,
            "model_path": str(face_path),
        },
        "face_oval": {
            "ready": all(face_oval_assets.values()) and all(face_dependencies.values()),
            "assets_present": face_oval_assets,
            "asset_validation": {"hash_verified": False},
            "dependencies_importable": face_dependencies,
            "dependencies_available": face_dependencies,
            "model_load_verified": face_oval_verified,
            "inference_verified": face_oval_verified,
            "quality_passed": False,
            "model_path": str(face_landmarker_path),
            "implemented_modes": ["oval"],
            "experimental_modes": ["sam_refined"],
        },
        "semantic": {
            "ready": all(semantic_assets.values()) and all(semantic_dependencies.values()),
            "assets_present": semantic_assets,
            "asset_validation": {"hash_verified": False, "snapshot_complete": False},
            "dependencies_importable": semantic_dependencies,
            "dependencies_available": semantic_dependencies,
            "model_load_verified": semantic_verified,
            "inference_verified": semantic_verified,
            "quality_passed": False,
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
