"""Hợp đồng C2 của phân vùng chủ thể, chạy offline với backend giả."""

from __future__ import annotations

import numpy as np
import pytest

import src.region_engine.segmentation_backend as backend_module
import src.region_engine.subject as subject
from src.region_engine import (
    box2d_to_xyxy,
    clear_subject_mask_cache,
    refine_bbox_mask,
    resolve_subject_mask,
    segment_box,
)
from src.region_engine.face_detector import FaceDetectorUnavailableError
from src.region_engine.segmentation_backend import (
    DINO_MODEL_ENV,
    MOBILE_SAM_CHECKPOINT_ENV,
    GroundingDetection,
    SegmentationInferenceError,
    SegmentationUnavailableError,
)

HEIGHT, WIDTH = 100, 200


class FakeSemanticBackend:
    """Backend giả: DINO trả detections cố định, SAM trả hình chữ nhật đúng bbox."""

    def __init__(self, detections: object = ()) -> None:
        self.detections = detections
        self.detect_prompts: list[str] = []
        self.set_image_calls = 0
        self.predict_boxes: list[tuple[float, float, float, float]] = []
        self.reset_calls = 0

    def detect(self, image: np.ndarray, prompt: str) -> object:
        self.detect_prompts.append(prompt)
        if isinstance(self.detections, BaseException):
            raise self.detections
        return self.detections

    def set_image(self, image: np.ndarray) -> None:
        self.set_image_calls += 1

    def predict(self, box: tuple[float, float, float, float]) -> np.ndarray:
        self.predict_boxes.append(box)
        xmin, ymin, xmax, ymax = (int(round(value)) for value in box)
        mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
        mask[ymin:ymax, xmin:xmax] = True
        return mask

    def reset_image(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        pass


@pytest.fixture
def rgb_image() -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.integers(0, 256, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)


@pytest.fixture
def no_models(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv(DINO_MODEL_ENV, str(tmp_path / "missing-dino"))
    monkeypatch.setenv(MOBILE_SAM_CHECKPOINT_ENV, str(tmp_path / "missing.pt"))
    backend_module.reset_segmentation_backend()
    clear_subject_mask_cache()
    yield
    backend_module.reset_segmentation_backend()
    clear_subject_mask_cache()


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> FakeSemanticBackend:
    fake = FakeSemanticBackend()
    backend_module.reset_segmentation_backend()
    clear_subject_mask_cache()
    monkeypatch.setattr(backend_module, "_SEGMENTATION_FACTORY", lambda: fake)
    yield fake
    backend_module.reset_segmentation_backend()
    clear_subject_mask_cache()


def _detection(box: tuple[float, float, float, float], score: float = 0.9) -> GroundingDetection:
    return GroundingDetection(box=box, score=score, phrase="person")


def _assert_valid_mask(mask: np.ndarray) -> None:
    assert mask.dtype == np.float32
    assert mask.shape == (HEIGHT, WIDTH)
    assert np.isfinite(mask).all()
    assert 0.0 <= float(mask.min()) and float(mask.max()) <= 1.0


GEMINI_BOX_2D = (100, 100, 900, 900)  # -> pixel (20, 10, 180, 90)
GEMINI_BOX_XYXY = (20.0, 10.0, 180.0, 90.0)
# Vật thể sáng có nhiễu nhẹ trên nền tối; bbox Gemini bao sát vật thể.
OBJECT_BOX_2D = (280, 250, 850, 750)  # -> pixel (50, 28, 150, 85)
OBJECT_BOX_XYXY = (50.0, 28.0, 150.0, 85.0)


def _object_image() -> np.ndarray:
    rng = np.random.default_rng(5)
    image = np.full((HEIGHT, WIDTH, 3), (30, 40, 50), dtype=np.uint8)
    image[30:80, 60:140] = (220, 180, 60)
    noisy = image.astype(np.int16) + rng.normal(0, 4, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def test_box2d_to_xyxy_scales_gemini_order_to_pixels() -> None:
    assert box2d_to_xyxy((0, 0, 1000, 1000), (100, 200)) == (0.0, 0.0, 200.0, 100.0)
    assert box2d_to_xyxy((100, 250, 500, 750), (1000, 1000)) == (250.0, 100.0, 750.0, 500.0)


def test_box2d_to_xyxy_rejects_degenerate_boxes() -> None:
    with pytest.raises(ValueError):
        box2d_to_xyxy((500, 100, 500, 200), (100, 100))
    with pytest.raises(ValueError):
        box2d_to_xyxy((0, 0, 1000), (100, 100))  # type: ignore[arg-type]


def test_capabilities_reports_flat_booleans(no_models: None) -> None:
    report = subject.capabilities()

    assert report == {"face": report["face"], "semantic_models": False}
    assert isinstance(report["face"], bool)


def test_geometric_fallback_without_model_or_bbox(rgb_image: np.ndarray, no_models: None) -> None:
    sky = resolve_subject_mask(rgb_image, "sky", None, use_semantic=False)
    other = resolve_subject_mask(rgb_image, "vật lạ", None, use_semantic=False)

    assert sky.source == "quadrant"
    assert sky.box is None
    _assert_valid_mask(sky.mask)
    assert sky.mask[0, 0] > 0.9 and sky.mask[-1, 0] < 0.1
    assert other.source == "full"
    assert np.all(other.mask == 1.0)


def test_grabcut_fallback_without_model(no_models: None) -> None:
    result = resolve_subject_mask(_object_image(), "person", OBJECT_BOX_2D, use_semantic=False)

    assert result.source == "gemini_grabcut"
    assert result.box == OBJECT_BOX_XYXY
    _assert_valid_mask(result.mask)
    # Feather lan ra ngoài bbox vài pixel nên chỉ kiểm tra các góc xa bbox.
    assert result.mask[0, 0] == 0.0 and result.mask[-1, -1] == 0.0
    assert result.mask[0, -1] == 0.0 and result.mask[-1, 0] == 0.0


def test_semantic_path_uses_dino_box_that_matches_gemini(
    rgb_image: np.ndarray, fake_backend: FakeSemanticBackend
) -> None:
    fake_backend.detections = [_detection((22.0, 12.0, 178.0, 88.0), score=0.8)]

    result = resolve_subject_mask(rgb_image, "person", GEMINI_BOX_2D, use_semantic=True)

    assert result.source == "dino_sam"
    assert result.confidence == pytest.approx(0.8)
    assert fake_backend.predict_boxes == [(22.0, 12.0, 178.0, 88.0)]
    _assert_valid_mask(result.mask)


def test_full_frame_dino_detection_is_discarded(
    rgb_image: np.ndarray, fake_backend: FakeSemanticBackend
) -> None:
    # IoU với bbox Gemini là 0.64, nên chỉ bộ lọc phủ ≥ 90 % khung mới loại được nó.
    fake_backend.detections = [_detection((0.0, 0.0, 200.0, 100.0), score=0.99)]

    result = resolve_subject_mask(rgb_image, "sky", GEMINI_BOX_2D, use_semantic=True)

    assert result.source == "gemini_sam"
    assert fake_backend.predict_boxes == [GEMINI_BOX_XYXY]


def test_unmatched_dino_detection_falls_back_to_gemini_box(
    rgb_image: np.ndarray, fake_backend: FakeSemanticBackend
) -> None:
    fake_backend.detections = [_detection((0.0, 0.0, 20.0, 20.0))]

    result = resolve_subject_mask(rgb_image, "person", GEMINI_BOX_2D, use_semantic=True)

    assert result.source == "gemini_sam"
    assert result.confidence < 0.9
    assert fake_backend.predict_boxes == [GEMINI_BOX_XYXY]


def test_unanchored_dino_without_bbox_uses_best_non_full_detection(
    rgb_image: np.ndarray, fake_backend: FakeSemanticBackend
) -> None:
    fake_backend.detections = [
        _detection((0.0, 0.0, 200.0, 100.0), score=0.99),
        _detection((60.0, 30.0, 120.0, 70.0), score=0.7),
    ]

    result = resolve_subject_mask(rgb_image, "sky", None, use_semantic=True)

    assert result.source == "dino_sam"
    assert result.confidence <= subject.UNANCHORED_DINO_CONFIDENCE
    assert fake_backend.predict_boxes == [(60.0, 30.0, 120.0, 70.0)]


def test_encoder_runs_once_for_several_boxes_on_one_image(
    rgb_image: np.ndarray, fake_backend: FakeSemanticBackend
) -> None:
    fake_backend.detections = [_detection((22.0, 12.0, 178.0, 88.0))]

    resolve_subject_mask(rgb_image, "person", GEMINI_BOX_2D, use_semantic=True)
    resolve_subject_mask(rgb_image, "sky", None, use_semantic=True)
    segment_box(rgb_image, (30.0, 20.0, 90.0, 60.0))

    assert fake_backend.set_image_calls == 1
    assert len(fake_backend.predict_boxes) == 3


def test_encoder_runs_again_when_image_changes(
    rgb_image: np.ndarray, fake_backend: FakeSemanticBackend
) -> None:
    segment_box(rgb_image, (30.0, 20.0, 90.0, 60.0))
    other = rgb_image.copy()
    other[0, 0, 0] ^= 0xFF
    segment_box(other, (30.0, 20.0, 90.0, 60.0))

    assert fake_backend.set_image_calls == 2


def test_cache_hit_skips_detection_on_second_call(
    rgb_image: np.ndarray, fake_backend: FakeSemanticBackend
) -> None:
    fake_backend.detections = [_detection((22.0, 12.0, 178.0, 88.0))]

    first = resolve_subject_mask(
        rgb_image, "person", GEMINI_BOX_2D, use_semantic=True, cache_key="img-a"
    )
    second = resolve_subject_mask(
        rgb_image, "person", GEMINI_BOX_2D, use_semantic=True, cache_key="img-a"
    )

    assert second is first
    assert len(fake_backend.detect_prompts) == 1


def test_clear_cache_forces_recompute(
    rgb_image: np.ndarray, fake_backend: FakeSemanticBackend
) -> None:
    fake_backend.detections = []
    resolve_subject_mask(rgb_image, "person", GEMINI_BOX_2D, use_semantic=True, cache_key="k")
    clear_subject_mask_cache()
    resolve_subject_mask(rgb_image, "person", GEMINI_BOX_2D, use_semantic=True, cache_key="k")

    assert len(fake_backend.detect_prompts) == 2


def test_use_semantic_false_never_loads_backend(
    rgb_image: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def factory() -> FakeSemanticBackend:
        nonlocal calls
        calls += 1
        return FakeSemanticBackend()

    backend_module.reset_segmentation_backend()
    monkeypatch.setattr(backend_module, "_SEGMENTATION_FACTORY", factory)
    resolve_subject_mask(rgb_image, "person", GEMINI_BOX_2D, use_semantic=False)

    assert calls == 0
    backend_module.reset_segmentation_backend()


@pytest.mark.parametrize(
    "error",
    [SegmentationUnavailableError("no model"), SegmentationInferenceError("boom")],
)
def test_backend_errors_fall_back_to_grabcut(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def factory() -> FakeSemanticBackend:
        if isinstance(error, SegmentationUnavailableError):
            raise error
        return FakeSemanticBackend(detections=error)

    backend_module.reset_segmentation_backend()
    clear_subject_mask_cache()
    monkeypatch.setattr(backend_module, "_SEGMENTATION_FACTORY", factory)

    result = resolve_subject_mask(_object_image(), "person", OBJECT_BOX_2D, use_semantic=True)

    assert result.source == "gemini_grabcut"
    _assert_valid_mask(result.mask)
    backend_module.reset_segmentation_backend()
    clear_subject_mask_cache()


def test_face_name_uses_mediapipe_union(
    rgb_image: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    second = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    first[10:20, 10:20] = 1.0
    second[50:60, 50:60] = 1.0
    monkeypatch.setattr(subject, "detect_faces", lambda image: [first, second])

    result = resolve_subject_mask(rgb_image, "face", None, use_semantic=False)

    assert result.source == "face_mediapipe"
    assert result.mask[15, 15] == 1.0 and result.mask[55, 55] == 1.0
    _assert_valid_mask(result.mask)


def test_face_without_detection_is_empty_mask(
    rgb_image: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subject, "detect_faces", lambda image: [])

    result = resolve_subject_mask(rgb_image, "khuôn mặt", None, use_semantic=False)

    assert result.source == "face_mediapipe"
    assert result.confidence == 0.0
    assert not result.mask.any()


def test_face_unavailable_continues_to_generic_chain(
    rgb_image: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(image: np.ndarray) -> list[np.ndarray]:
        raise FaceDetectorUnavailableError("no face model")

    monkeypatch.setattr(subject, "detect_faces", unavailable)

    result = resolve_subject_mask(rgb_image, "face", None, use_semantic=False)

    assert result.source == "full"


def test_grayscale_input_is_accepted(no_models: None) -> None:
    gray = np.full((HEIGHT, WIDTH), 128, dtype=np.uint8)

    result = resolve_subject_mask(gray, "sky", None, use_semantic=False)

    _assert_valid_mask(result.mask)


def test_refine_bbox_mask_keeps_object_inside_bbox_iou_at_least_0_7() -> None:
    rng = np.random.default_rng(11)
    image = np.full((200, 200, 3), (30, 40, 50), dtype=np.uint8)
    image[50:150, 60:140] = (220, 180, 60)
    image = np.clip(image.astype(np.int16) + rng.normal(0, 4, image.shape), 0, 255).astype(np.uint8)
    truth = np.zeros((200, 200), dtype=bool)
    truth[50:150, 60:140] = True

    mask = refine_bbox_mask(image, (50.0, 40.0, 150.0, 160.0), feather_radius=0)

    predicted = mask >= 0.5
    iou = np.count_nonzero(predicted & truth) / np.count_nonzero(predicted | truth)
    assert iou >= 0.7
    assert mask.dtype == np.float32


def test_refine_bbox_mask_downscales_large_images_and_keeps_shape() -> None:
    # Ô kẻ 64 px và nhiễu nhẹ còn lại sau khi thu về 256 px; ảnh phẳng thì GrabCut không tách được.
    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[0:1200, 0:1600]
    texture = ((yy // 64 + xx // 64) % 2).astype(np.float32)
    inside = np.zeros((1200, 1600), dtype=bool)
    inside[300:900, 400:1200] = True
    image = np.where(inside, 200 + 30 * texture, 40 + 10 * texture)
    image = image[..., np.newaxis] + rng.normal(0, 3, (1200, 1600, 3))
    image = np.clip(image, 0, 255).astype(np.uint8)

    mask = refine_bbox_mask(image, (300.0, 400.0, 900.0, 1200.0), max_side=256)

    assert mask.shape == (1200, 1600)
    assert mask.dtype == np.float32
    assert np.isfinite(mask).all() and 0.0 <= float(mask.min()) and float(mask.max()) <= 1.0
    assert mask[0, 0] == 0.0 and mask[-1, -1] == 0.0
    assert np.count_nonzero(mask >= 0.5) > 0


def test_refine_bbox_mask_degenerate_box_returns_empty() -> None:
    image = np.zeros((50, 60, 3), dtype=np.uint8)

    mask = refine_bbox_mask(image, (70.0, 70.0, 80.0, 80.0))

    assert mask.shape == (50, 60)
    assert not mask.any()


def test_segment_box_without_model_raises_unavailable(
    rgb_image: np.ndarray, no_models: None
) -> None:
    with pytest.raises(SegmentationUnavailableError):
        segment_box(rgb_image, (1.0, 1.0, 20.0, 20.0))
