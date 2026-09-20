"""Contract tests for prompt normalization and the semantic backend seam."""

from __future__ import annotations

import math

import numpy as np
import pytest

import src.region_engine.segmentation_backend as backend_module
from src.region_engine.detector import segment_by_prompt
from src.region_engine.segmentation_backend import (
    GroundingDetection,
    SegmentationInferenceError,
    SegmentationUnavailableError,
)

IMAGE = np.zeros((4, 6, 3), dtype=np.uint8)


class FakeBackend:
    def __init__(self, detections: object = ()) -> None:
        self.detections = detections
        self.masks: dict[tuple[float, float, float, float], object] = {}
        self.detect_calls: list[tuple[np.ndarray, str]] = []
        self.set_images: list[np.ndarray] = []
        self.predict_boxes: list[tuple[float, float, float, float]] = []
        self.reset_calls = 0
        self.close_calls = 0

    def detect(self, image: np.ndarray, prompt: str) -> object:
        self.detect_calls.append((image, prompt))
        if isinstance(self.detections, BaseException):
            raise self.detections
        return self.detections

    def set_image(self, image: np.ndarray) -> None:
        self.set_images.append(image)

    def predict(self, box: tuple[float, float, float, float]) -> object:
        self.predict_boxes.append(box)
        return self.masks.get(box, np.zeros(IMAGE.shape[:2], dtype=bool))

    def reset_image(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        self.close_calls += 1


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    fake = FakeBackend()
    backend_module.reset_segmentation_backend()
    monkeypatch.setattr(backend_module, "_SEGMENTATION_FACTORY", lambda: fake)
    yield fake
    backend_module.reset_segmentation_backend()


def detection(
    box: tuple[float, float, float, float], score: float = 0.9, phrase: str = "person"
) -> GroundingDetection:
    return GroundingDetection(box=box, score=score, phrase=phrase)


def test_exact_commands_do_not_load_semantic_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def factory() -> FakeBackend:
        nonlocal calls
        calls += 1
        return FakeBackend([])

    monkeypatch.setattr(backend_module, "_SEGMENTATION_FACTORY", factory)
    backend_module.reset_segmentation_backend()

    np.testing.assert_array_equal(
        segment_by_prompt(IMAGE, " full_image ", feather_radius=0),
        np.ones(IMAGE.shape[:2], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        segment_by_prompt(IMAGE, "GIỮA", feather_radius=0),
        np.array([[0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=np.float32),
    )
    assert calls == 0
    backend_module.reset_segmentation_backend()


def test_vietnamese_alias_is_normalized_and_sent_to_backend(fake_backend: FakeBackend) -> None:
    box = (1.0, 1.0, 3.0, 3.0)
    fake_backend.detections = [detection(box)]
    expected = np.zeros(IMAGE.shape[:2], dtype=bool)
    expected[1:3, 1:3] = True
    fake_backend.masks = {box: expected}

    mask = segment_by_prompt(IMAGE, "  NGƯỜI\n", feather_radius=0)

    assert fake_backend.detect_calls[0][1] == "person."
    assert mask.dtype == np.float32
    assert mask.shape == IMAGE.shape[:2]
    assert mask[1:3, 1:3].sum() == 4.0


def test_all_people_is_semantic_not_full_image(fake_backend: FakeBackend) -> None:
    fake_backend.detections = []

    mask = segment_by_prompt(IMAGE, "all people", feather_radius=0)

    assert fake_backend.detect_calls[0][1] == "all people."
    np.testing.assert_array_equal(mask, np.zeros(IMAGE.shape[:2], dtype=np.float32))


def test_no_detection_returns_zero_without_calling_sam(fake_backend: FakeBackend) -> None:
    fake_backend.detections = []

    mask = segment_by_prompt(IMAGE, "person", feather_radius=0)

    np.testing.assert_array_equal(mask, np.zeros(IMAGE.shape[:2], dtype=np.float32))
    assert fake_backend.set_images == []
    assert fake_backend.predict_boxes == []
    assert fake_backend.reset_calls == 0


def test_multiple_masks_are_unioned_before_one_feather(fake_backend: FakeBackend) -> None:
    first_box = (1.0, 0.0, 3.0, 2.0)
    second_box = (2.0, 1.0, 5.0, 3.0)
    fake_backend.detections = [detection(first_box), detection(second_box, score=0.8)]
    first = np.zeros(IMAGE.shape[:2], dtype=bool)
    first[0:2, 1:3] = True
    second = np.zeros(IMAGE.shape[:2], dtype=bool)
    second[1:3, 2:5] = True
    fake_backend.masks = {first_box: first, second_box: second}

    actual = segment_by_prompt(IMAGE, "person", feather_radius=0)

    expected = np.zeros(IMAGE.shape[:2], dtype=np.float32)
    expected[0, 1:3] = 1.0
    expected[1, 1:5] = 1.0
    expected[2, 2:5] = 1.0
    np.testing.assert_array_equal(actual, expected)
    assert fake_backend.predict_boxes == [first_box, second_box]
    assert fake_backend.reset_calls == 1


def test_duplicate_boxes_are_removed_by_class_agnostic_nms(fake_backend: FakeBackend) -> None:
    duplicate = (1.0, 1.0, 4.0, 4.0)
    other = (0.0, 0.0, 1.0, 1.0)
    fake_backend.detections = [
        detection(duplicate, score=0.9),
        detection(duplicate, score=0.8),
        detection(other, score=0.7),
    ]

    segment_by_prompt(IMAGE, "person", feather_radius=0)

    assert fake_backend.predict_boxes == [duplicate, other]


def test_boxes_are_clipped_before_mobile_sam(fake_backend: FakeBackend) -> None:
    fake_backend.detections = [detection((-2.5, -1.0, 8.0, 5.0))]

    segment_by_prompt(IMAGE, "person", feather_radius=0)

    assert fake_backend.predict_boxes == [(0.0, 0.0, 6.0, 4.0)]


@pytest.mark.parametrize(
    "raw",
    [
        [detection((0, 0, math.nan, 2))],
        [detection((0, 0, 2, 2), score=math.nan)],
        [detection((0, 0, 2, 2), score=1.1)],
        [{"box": (0, 0, 2), "score": 0.9, "phrase": "person"}],
    ],
)
def test_malformed_detection_raises(fake_backend: FakeBackend, raw: object) -> None:
    fake_backend.detections = raw

    with pytest.raises(SegmentationInferenceError):
        segment_by_prompt(IMAGE, "person", feather_radius=0)


def test_threshold_and_empty_phrase_are_filtered(fake_backend: FakeBackend) -> None:
    fake_backend.detections = [
        detection((0, 0, 2, 2), score=0.349),
        detection((0, 0, 2, 2), score=0.9, phrase=""),
    ]

    np.testing.assert_array_equal(
        segment_by_prompt(IMAGE, "person", feather_radius=0),
        np.zeros(IMAGE.shape[:2], dtype=np.float32),
    )
    assert fake_backend.set_images == []


@pytest.mark.parametrize("raw_mask", [np.ones((3, 6), dtype=bool), np.ones((4, 6), dtype=np.float32)])
def test_invalid_mask_contract_raises_and_resets(
    fake_backend: FakeBackend, raw_mask: np.ndarray
) -> None:
    fake_backend.detections = [detection((0, 0, 2, 2))]
    fake_backend.masks = {(0.0, 0.0, 2.0, 2.0): raw_mask}

    with pytest.raises(SegmentationInferenceError):
        segment_by_prompt(IMAGE, "person", feather_radius=0)

    assert fake_backend.reset_calls == 1


def test_standard_mobile_sam_tuple_output_is_supported(fake_backend: FakeBackend) -> None:
    box = (0.0, 0.0, 2.0, 2.0)
    fake_backend.detections = [detection(box)]
    masks = np.zeros((1, *IMAGE.shape[:2]), dtype=bool)
    masks[0, :2, :2] = True
    fake_backend.masks = {box: (masks, np.array([0.9]), None)}

    result = segment_by_prompt(IMAGE, "person", feather_radius=0)

    assert result[:2, :2].sum() == 4.0


def test_input_is_validated_before_backend_init(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def factory() -> FakeBackend:
        nonlocal calls
        calls += 1
        return FakeBackend([])

    monkeypatch.setattr(backend_module, "_SEGMENTATION_FACTORY", factory)
    backend_module.reset_segmentation_backend()
    with pytest.raises(TypeError):
        segment_by_prompt(np.zeros((4, 6, 3), dtype=np.float32), "person")
    with pytest.raises(ValueError):
        segment_by_prompt(IMAGE, "   ")
    with pytest.raises(TypeError):
        segment_by_prompt(IMAGE, "person", feather_radius=True)
    assert calls == 0
    backend_module.reset_segmentation_backend()


def test_prompt_token_limit_is_checked_before_backend_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def factory() -> FakeBackend:
        nonlocal calls
        calls += 1
        return FakeBackend([])

    monkeypatch.setattr(backend_module, "_SEGMENTATION_FACTORY", factory)
    backend_module.reset_segmentation_backend()

    with pytest.raises(ValueError, match="256"):
        segment_by_prompt(IMAGE, "word " * 257)

    assert calls == 0
    backend_module.reset_segmentation_backend()


def test_noncontiguous_read_only_input_is_copied(fake_backend: FakeBackend) -> None:
    source = np.zeros((4, 12, 3), dtype=np.uint8)
    source[0, 0] = [1, 2, 3]
    image = source[:, ::2, :]
    image.setflags(write=False)
    fake_backend.detections = []

    segment_by_prompt(image, "person", feather_radius=0)

    seen = fake_backend.detect_calls[0][0]
    assert seen.flags.c_contiguous
    np.testing.assert_array_equal(seen[0, 0], [1, 2, 3])


def test_inference_error_preserves_cause(fake_backend: FakeBackend) -> None:
    cause = RuntimeError("DINO failed")
    fake_backend.detections = cause

    with pytest.raises(SegmentationInferenceError) as raised:
        segment_by_prompt(IMAGE, "person")

    assert raised.value.__cause__ is cause


def test_backend_is_cached_and_close_hook_releases_it(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeBackend([])
    calls = 0

    def factory() -> FakeBackend:
        nonlocal calls
        calls += 1
        return fake

    monkeypatch.setattr(backend_module, "_SEGMENTATION_FACTORY", factory)
    backend_module.reset_segmentation_backend()
    segment_by_prompt(IMAGE, "person", feather_radius=0)
    segment_by_prompt(IMAGE, "person", feather_radius=0)
    assert calls == 1
    backend_module.close_segmentation_backend()
    assert fake.close_calls == 1


def test_backend_cache_key_includes_model_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    first = FakeBackend([])
    second = FakeBackend([])
    backends = iter((first, second))
    calls = 0

    def factory() -> FakeBackend:
        nonlocal calls
        calls += 1
        return next(backends)

    monkeypatch.setattr(backend_module, "_SEGMENTATION_FACTORY", factory)
    monkeypatch.setenv(backend_module.DINO_MODEL_ENV, "dino-a")
    monkeypatch.setenv(backend_module.MOBILE_SAM_CHECKPOINT_ENV, "sam-a")
    backend_module.reset_segmentation_backend()

    segment_by_prompt(IMAGE, "person", feather_radius=0)
    monkeypatch.setenv(backend_module.DINO_MODEL_ENV, "dino-b")
    segment_by_prompt(IMAGE, "person", feather_radius=0)

    assert calls == 2
    assert first.close_calls == 1
    backend_module.reset_segmentation_backend()


def test_unknown_prompt_does_not_fallback_to_full_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_module.reset_segmentation_backend()
    monkeypatch.setattr(
        backend_module,
        "_SEGMENTATION_FACTORY",
        lambda: (_ for _ in ()).throw(RuntimeError("missing weights")),
    )

    with pytest.raises(SegmentationUnavailableError):
        segment_by_prompt(IMAGE, "a completely unknown object")
