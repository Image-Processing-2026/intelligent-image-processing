"""Contract tests for the Module 2 request/result controller."""

from __future__ import annotations

import numpy as np
import pytest

from src.region_engine.controller import (
    InvalidRegionRequestError,
    RegionBackendUnavailableError,
    RegionRequest,
    capabilities,
    resolve_region,
)
from src.region_engine.face_detector import FaceDetectorUnavailableError
from src.region_engine.segmentation_backend import SegmentationUnavailableError

IMAGE = np.zeros((5, 8, 3), dtype=np.uint8)


def test_full_request_returns_explicit_mask_and_geometry_metadata() -> None:
    result = resolve_region(IMAGE, {"kind": "full"})

    assert result.status == "ok"
    assert result.mask.dtype == np.float32
    np.testing.assert_array_equal(result.mask, np.ones((5, 8), dtype=np.float32))
    assert result.metadata == {"backend": "geometry", "kind": "full", "feather_radius": 15}


def test_mapping_infers_bbox_and_preserves_half_open_geometry() -> None:
    result = resolve_region(
        IMAGE,
        {"bbox": [2, 1, 5, 3], "feather_radius": 0},
    )

    expected = np.zeros((5, 8), dtype=np.float32)
    expected[1:3, 2:5] = 1.0
    np.testing.assert_array_equal(result.mask, expected)
    assert result.metadata["kind"] == "bbox"
    assert result.metadata["bbox"] == (2, 1, 5, 3)


def test_spatial_request_uses_quadrant_without_ai_backend() -> None:
    result = resolve_region(
        IMAGE,
        RegionRequest(kind="spatial", quadrant="right", feather_radius=0),
    )

    assert result.status == "ok"
    assert np.all(result.mask[:, 4:] == 1.0)
    assert np.all(result.mask[:, :4] == 0.0)


def test_binary_mask_is_feathered_once() -> None:
    binary = np.zeros((5, 8), dtype=np.uint8)
    binary[1:4, 2:6] = 255

    result = resolve_region(
        IMAGE,
        RegionRequest(kind="binary_mask", binary_mask=binary, feather_radius=0),
    )

    assert result.status == "ok"
    assert result.mask.dtype == np.float32
    np.testing.assert_array_equal(result.mask[1:4, 2:6], 1.0)


def test_face_result_merges_instances_and_keeps_provenance() -> None:
    first = np.zeros((5, 8), dtype=np.float32)
    first[1:3, 1:4] = 1.0
    second = np.zeros((5, 8), dtype=np.float32)
    second[2:4, 3:6] = 0.75

    def fake_faces(_image, *, feather_radius, expand_ratio):
        assert feather_radius == 0
        assert expand_ratio == 0.2
        return [first, second]

    result = resolve_region(
        IMAGE,
        RegionRequest(kind="face", feather_radius=0, expand_ratio=0.2),
        _face_resolver=fake_faces,
    )

    assert result.status == "ok"
    assert result.metadata["backend"] == "mediapipe"
    assert result.metadata["count"] == 2
    assert len(result.instance_masks) == 2
    assert result.mask[2, 3] == 1.0
    assert result.mask[3, 4] == 0.75


def test_empty_face_result_is_zero_mask() -> None:
    result = resolve_region(IMAGE, {"kind": "face"}, _face_resolver=lambda *_args, **_kwargs: [])

    assert result.status == "empty"
    assert result.metadata["count"] == 0
    np.testing.assert_array_equal(result.mask, np.zeros((5, 8), dtype=np.float32))


def test_semantic_backend_errors_are_typed_and_preserve_cause() -> None:
    cause = SegmentationUnavailableError("weights missing")

    def unavailable(*_args, **_kwargs):
        raise cause

    with pytest.raises(RegionBackendUnavailableError) as raised:
        resolve_region(IMAGE, {"kind": "semantic", "prompt": "person"}, _semantic_resolver=unavailable)

    assert raised.value.__cause__ is cause


def test_face_backend_errors_are_typed_and_preserve_cause() -> None:
    cause = FaceDetectorUnavailableError("checkpoint missing")

    def unavailable(*_args, **_kwargs):
        raise cause

    with pytest.raises(RegionBackendUnavailableError) as raised:
        resolve_region(IMAGE, {"kind": "face"}, _face_resolver=unavailable)

    assert raised.value.__cause__ is cause


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "bbox"},
        {"kind": "spatial"},
        {"kind": "binary_mask"},
        {"kind": "semantic", "prompt": ""},
        {"kind": "unsupported"},
    ],
)
def test_invalid_request_fails_before_backend_call(payload: dict[str, object]) -> None:
    with pytest.raises(InvalidRegionRequestError):
        resolve_region(IMAGE, payload)


def test_capabilities_separate_geometric_and_ai_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REGION_FACE_MODEL_PATH", "missing-face.tflite")
    monkeypatch.setenv("REGION_DINO_MODEL_PATH", "missing-dino")
    monkeypatch.setenv("REGION_MOBILE_SAM_CHECKPOINT", "missing-sam.pt")

    report = capabilities()

    assert report["geometric"]["ready"] is True
    assert report["face"]["assets_present"]["model"] is False
    assert report["semantic"]["assets_present"] == {
        "grounding_dino": False,
        "mobile_sam": False,
    }
    assert isinstance(report["face"]["inference_verified"], bool)
    assert isinstance(report["semantic"]["inference_verified"], bool)
