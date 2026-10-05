"""Contract tests for the MediaPipe adapter and bbox-to-mask conversion."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import src.region_engine.face_detector as face_detector
from src.region_engine.face_detector import (
    FaceDetectionError,
    FaceDetectionRecord,
    FaceDetectorUnavailableError,
    detect_faces,
)

IMAGE = np.zeros((10, 12, 3), dtype=np.uint8)


class FakeBackend:
    def __init__(self, records: object = ()) -> None:
        self.records = records
        self.detect_calls = 0
        self.seen_images: list[np.ndarray] = []
        self.close_calls = 0

    def detect(self, image: np.ndarray) -> object:
        self.detect_calls += 1
        self.seen_images.append(image)
        if isinstance(self.records, BaseException):
            raise self.records
        return self.records

    def close(self) -> None:
        self.close_calls += 1


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    backend = FakeBackend()
    face_detector.reset_face_detector()
    monkeypatch.setattr(face_detector, "_DETECTOR_FACTORY", lambda _path: backend)
    yield backend
    face_detector.reset_face_detector()


def record(x: float, y: float, width: float, height: float, score: float = 0.9) -> FaceDetectionRecord:
    return FaceDetectionRecord(x=x, y=y, width=width, height=height, score=score)


def test_center_bbox_uses_half_open_pixel_geometry(fake_backend: FakeBackend) -> None:
    fake_backend.records = [record(4, 3, 4, 4)]

    masks = detect_faces(IMAGE, feather_radius=0, expand_ratio=0)

    expected = np.zeros(IMAGE.shape[:2], dtype=np.float32)
    expected[3:7, 4:8] = 1.0
    np.testing.assert_array_equal(masks, [expected])


@pytest.mark.parametrize(
    ("ratio", "expected_bbox", "expected_area"),
    [
        (0.25, (3, 2, 9, 8), 36),
        (0.15, (3, 2, 9, 8), 36),
    ],
)
def test_bbox_expansion_uses_floor_ceil_before_clipping(
    fake_backend: FakeBackend,
    ratio: float,
    expected_bbox: tuple[int, int, int, int],
    expected_area: int,
) -> None:
    fake_backend.records = [record(4, 3, 4, 4)]

    masks = detect_faces(IMAGE, feather_radius=0, expand_ratio=ratio)

    xmin, ymin, xmax, ymax = expected_bbox
    expected = np.zeros(IMAGE.shape[:2], dtype=np.float32)
    expected[ymin:ymax, xmin:xmax] = 1.0
    assert int(masks[0].sum()) == expected_area
    np.testing.assert_array_equal(masks[0], expected)


def test_edge_bbox_is_clipped_after_expansion(fake_backend: FakeBackend) -> None:
    fake_backend.records = [record(0, 0, 4, 4)]

    with warnings.catch_warnings(record=True) as warning:
        warnings.simplefilter("always")
        masks = detect_faces(IMAGE, feather_radius=0, expand_ratio=0.25)

    assert warning == []
    expected = np.zeros(IMAGE.shape[:2], dtype=np.float32)
    expected[:5, :5] = 1.0
    np.testing.assert_array_equal(masks[0], expected)


def test_completely_outside_bbox_is_skipped_with_warning(fake_backend: FakeBackend) -> None:
    fake_backend.records = [record(20, 20, 2, 2)]

    with pytest.warns(RuntimeWarning, match="completely outside"):
        masks = detect_faces(IMAGE, feather_radius=0, expand_ratio=0)

    assert masks == []


def test_successful_empty_inference_returns_empty_list(fake_backend: FakeBackend) -> None:
    fake_backend.records = []

    assert detect_faces(IMAGE, feather_radius=0) == []
    assert fake_backend.detect_calls == 1


def test_multiple_masks_are_sorted_by_raw_bbox_and_do_not_alias(fake_backend: FakeBackend) -> None:
    fake_backend.records = [
        record(4, 5, 2, 2, score=0.99),
        record(0, 2, 2, 2, score=0.80),
    ]

    masks = detect_faces(IMAGE, feather_radius=0, expand_ratio=0)

    assert len(masks) == 2
    assert masks[0][2:4, 0:2].sum() == 4
    assert masks[1][5:7, 4:6].sum() == 4
    assert not np.shares_memory(masks[0], masks[1])


@pytest.mark.parametrize(
    "raw",
    [
        {"x": 1, "y": 2, "width": 3, "height": 4, "score": 0.6},
        type("Record", (), {"x": 1, "y": 2, "width": 3, "height": 4, "score": 0.6})(),
    ],
)
def test_backend_record_adapters_are_supported(fake_backend: FakeBackend, raw: object) -> None:
    fake_backend.records = [raw]

    masks = detect_faces(IMAGE, feather_radius=0, expand_ratio=0)

    assert len(masks) == 1
    assert masks[0][2:6, 1:4].sum() == 12


@pytest.mark.parametrize(
    "raw",
    [
        record(math.nan, 0, 2, 2),
        record(0, 0, 0, 2),
        record(0, 0, 2, -1),
        record(0, 0, 2, 2, score=-0.1),
        record(0, 0, 2, 2, score=1.1),
    ],
)
def test_malformed_backend_detection_raises_face_detection_error(
    fake_backend: FakeBackend, raw: FaceDetectionRecord
) -> None:
    fake_backend.records = [raw]

    with pytest.raises(FaceDetectionError):
        detect_faces(IMAGE, feather_radius=0)


@pytest.mark.parametrize(
    ("image", "error"),
    [
        (None, TypeError),
        (np.zeros((10, 12, 3), dtype=np.float32), TypeError),
        (np.zeros((10, 12), dtype=np.uint8), ValueError),
        (np.zeros((10, 12, 4), dtype=np.uint8), ValueError),
        (np.zeros((0, 12, 3), dtype=np.uint8), ValueError),
    ],
)
def test_public_image_validation_happens_before_backend_load(
    monkeypatch: pytest.MonkeyPatch, image: object, error: type[Exception]
) -> None:
    calls = 0

    def factory(_path: object) -> FakeBackend:
        nonlocal calls
        calls += 1
        return FakeBackend([])

    monkeypatch.setattr(face_detector, "_DETECTOR_FACTORY", factory)
    face_detector.reset_face_detector()

    with pytest.raises(error):
        detect_faces(image)  # type: ignore[arg-type]

    assert calls == 0
    face_detector.reset_face_detector()


@pytest.mark.parametrize("radius", [True, np.bool_(False), -1, 1.5])
def test_invalid_feather_radius_is_rejected(fake_backend: FakeBackend, radius: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        detect_faces(IMAGE, feather_radius=radius)  # type: ignore[arg-type]

    assert fake_backend.detect_calls == 0


@pytest.mark.parametrize("ratio", [True, np.bool_(False), "0.2", None])
def test_invalid_expand_ratio_type_is_rejected(fake_backend: FakeBackend, ratio: object) -> None:
    with pytest.raises(TypeError):
        detect_faces(IMAGE, expand_ratio=ratio)  # type: ignore[arg-type]

    assert fake_backend.detect_calls == 0


@pytest.mark.parametrize("ratio", [-0.01, 1.01, math.nan, math.inf, -math.inf])
def test_invalid_expand_ratio_value_is_rejected(fake_backend: FakeBackend, ratio: float) -> None:
    with pytest.raises(ValueError):
        detect_faces(IMAGE, expand_ratio=ratio)

    assert fake_backend.detect_calls == 0


def test_read_only_noncontiguous_input_is_copied_for_backend(fake_backend: FakeBackend) -> None:
    source = np.zeros((10, 24, 3), dtype=np.uint8)
    source[0, 0] = [1, 2, 3]
    image = source[:, ::2, :]
    image.setflags(write=False)
    fake_backend.records = []

    assert detect_faces(image, feather_radius=0) == []

    seen = fake_backend.seen_images[0]
    assert seen.flags.c_contiguous
    np.testing.assert_array_equal(seen[0, 0], [1, 2, 3])


def test_backend_inference_error_preserves_cause(fake_backend: FakeBackend) -> None:
    cause = RuntimeError("backend exploded")
    fake_backend.records = cause

    with pytest.raises(FaceDetectionError) as raised:
        detect_faces(IMAGE, feather_radius=0)

    assert raised.value.__cause__ is cause


def test_backend_is_initialized_once_and_close_hook_releases_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend([])
    factory_calls = 0

    def factory(_path: object) -> FakeBackend:
        nonlocal factory_calls
        factory_calls += 1
        return backend

    monkeypatch.setattr(face_detector, "_DETECTOR_FACTORY", factory)
    face_detector.reset_face_detector()

    detect_faces(IMAGE, feather_radius=0)
    detect_faces(IMAGE, feather_radius=0)
    assert factory_calls == 1
    assert backend.detect_calls == 2

    face_detector.close_face_detector()
    assert backend.close_calls == 1


def test_unavailable_factory_error_does_not_poison_subsequent_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    face_detector.reset_face_detector()
    monkeypatch.setattr(
        face_detector,
        "_DETECTOR_FACTORY",
        lambda _path: (_ for _ in ()).throw(RuntimeError("missing dependency")),
    )

    with pytest.raises(FaceDetectorUnavailableError) as raised:
        detect_faces(IMAGE, feather_radius=0)
    assert isinstance(raised.value.__cause__, RuntimeError)

    backend = FakeBackend([])
    monkeypatch.setattr(face_detector, "_DETECTOR_FACTORY", lambda _path: backend)
    assert detect_faces(IMAGE, feather_radius=0) == []
    assert backend.detect_calls == 1
    face_detector.reset_face_detector()


def test_output_contract_is_float32_finite_and_bounded(fake_backend: FakeBackend) -> None:
    fake_backend.records = [record(1, 1, 3, 3)]

    masks = detect_faces(IMAGE, feather_radius=3)

    assert len(masks) == 1
    mask = masks[0]
    assert mask.dtype == np.float32
    assert mask.shape == IMAGE.shape[:2]
    assert np.isfinite(mask).all()
    assert 0.0 <= float(mask.min()) <= float(mask.max()) <= 1.0
