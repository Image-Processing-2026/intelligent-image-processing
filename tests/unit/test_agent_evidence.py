"""
Unit tests cho bằng chứng VLM (Analysis v2 P5, src/agent/evidence.py): crop đúng vị trí trên
ảnh synthetic, nhãn, 100 % độ phân giải, overlay. Chạy offline, không gọi Gemini.
"""

import dataclasses
from typing import Tuple

import numpy as np
import pytest

from src.agent.evidence import (
    CROP_SIZE,
    MAX_CROPS,
    build_overlays,
    describe_crops,
    select_evidence,
    summarize_measurement,
)
from src.analyzer_evaluator.measurement import measure_image

HEIGHT, WIDTH = 1024, 1536
FLAT_RIGHT = 768  # cột x < FLAT_RIGHT là vùng phẳng
CLIP_BOX = (1100, 100, 1400, 400)  # (x0, y0, x1, y1) khối cháy sáng trong vùng texture


def _scene() -> np.ndarray:
    """Trái phẳng (mức 120), phải kẻ sọc mạnh, khối cháy 255; nhiễu σ = 4 khắp ảnh."""
    rng = np.random.default_rng(3)
    image = np.full((HEIGHT, WIDTH), 120.0)
    xx = np.arange(WIDTH - FLAT_RIGHT)
    image[:, FLAT_RIGHT:] = 128 + 70 * np.sign(np.sin(xx / 6.0))[None, :]
    image += rng.normal(0, 4, image.shape)
    x0, y0, x1, y1 = CLIP_BOX
    image[y0:y1, x0:x1] = 255
    return np.repeat(np.clip(image, 0, 255).astype(np.uint8)[..., None], 3, axis=2)


@pytest.fixture(scope="module")
def scene() -> Tuple[np.ndarray, object]:
    image = _scene()
    return image, measure_image(image)


def _inside(box: Tuple[int, int, int, int], outer: Tuple[int, int, int, int]) -> bool:
    return box[0] >= outer[0] and box[1] >= outer[1] and box[2] <= outer[2] and box[3] <= outer[3]


def _by_kind(crops):
    return {crop.kind: crop for crop in crops}


def test_flat_crop_lies_in_flat_region(scene) -> None:
    image, measurement = scene
    crops = _by_kind(select_evidence(image, measurement))
    assert _inside(crops["flat"].box, (0, 0, FLAT_RIGHT, HEIGHT))


def test_clip_crop_covers_clipped_block(scene) -> None:
    image, measurement = scene
    crops = _by_kind(select_evidence(image, measurement))
    assert _inside(CLIP_BOX, crops["clip"].box)
    assert "cháy sáng" in crops["clip"].description


def test_edge_crops_follow_subject_and_background_masks(scene) -> None:
    image, measurement = scene
    subject = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    subject[512:, FLAT_RIGHT:] = 1.0  # chủ thể: góc dưới phải (có texture)
    crops = _by_kind(select_evidence(image, measurement, {"person": subject}))

    x0, y0, x1, y1 = crops["edge_subject"].box
    assert subject[y0:y1, x0:x1].mean() >= 0.6
    assert crops["edge_subject"].region == "subject"
    bx0, by0, bx1, by1 = crops["edge_background"].box
    assert subject[by0:by1, bx0:bx1].mean() <= 0.4
    assert "chủ thể" in crops["edge_subject"].description


def test_face_crop_is_first_and_centred_on_face(scene) -> None:
    image, measurement = scene
    face = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    face[700:800, 300:400] = 1.0
    crops = select_evidence(image, measurement, {"face": face})

    assert crops[0].kind == "face" and crops[0].label == "A"
    assert _inside((300, 700, 400, 800), crops[0].box)
    assert "Khuôn mặt" in crops[0].description


def test_noisy_crop_targets_highest_sigma_tiles(scene) -> None:
    image, measurement = scene
    sigma = np.full_like(measurement.maps["noise.sigma_luma"], 2.0)
    sigma[8:16, 0:8] = 20.0  # tile hàng 8–15, cột 0–7 → pixel y 512–1024, x 0–512
    noisy = dataclasses.replace(measurement, maps={**measurement.maps, "noise.sigma_luma": sigma})
    crops = _by_kind(select_evidence(image, noisy))

    x0, y0, x1, y1 = crops["noisy"].box
    assert x0 < 256 < x1 and y0 < 768 < y1
    assert "σ nhiễu" in crops["noisy"].description


def test_crops_are_labelled_full_resolution_and_disjoint(scene) -> None:
    image, measurement = scene
    face = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    face[700:800, 300:400] = 1.0
    crops = select_evidence(image, measurement, {"face": face})

    assert 1 <= len(crops) <= MAX_CROPS
    assert [crop.label for crop in crops] == list("ABCDEF"[: len(crops)])
    for crop in crops:
        x0, y0, x1, y1 = crop.box
        assert (y1 - y0, x1 - x0) == (CROP_SIZE, CROP_SIZE)
        assert np.array_equal(crop.image, image[y0:y1, x0:x1])  # 100 %, không thu phóng
    text = describe_crops(crops)
    assert all(f"Crop {crop.label}" in text for crop in crops)


def test_small_image_has_no_crops() -> None:
    image = np.full((400, 600, 3), 120, dtype=np.uint8)
    assert select_evidence(image, measure_image(image)) == []


def test_mismatched_measurement_is_rejected(scene) -> None:
    image, _ = scene
    other = measure_image(np.full((900, 1200, 3), 100, dtype=np.uint8))
    with pytest.raises(ValueError):
        select_evidence(image, other)


def test_grayscale_image_is_supported() -> None:
    gray = _scene()[..., 0]
    crops = select_evidence(gray, measure_image(gray))
    assert crops and all(crop.image.shape[2] == 3 for crop in crops)


def test_overlays_mark_clipping_and_noise(scene) -> None:
    image, measurement = scene
    before = image.copy()
    overlays = {overlay.label: overlay for overlay in build_overlays(image, measurement)}

    assert set(overlays) == {"clip", "noise"}
    clip = overlays["clip"].image
    assert max(clip.shape[:2]) <= 768
    red = (clip[..., 0] == 255) & (clip[..., 1] == 0) & (clip[..., 2] == 0)
    assert red.any()
    assert "cháy" in overlays["clip"].description
    assert np.array_equal(image, before)  # overlay chỉ để xem, không đổi ảnh vào


def test_no_clip_overlay_without_clipping() -> None:
    image = np.full((600, 800, 3), 120, dtype=np.uint8)
    labels = [overlay.label for overlay in build_overlays(image, measure_image(image))]
    assert "clip" not in labels


def test_measurement_summary_lists_values_with_confidence(scene) -> None:
    _, measurement = scene
    text = summarize_measurement(measurement.report)
    assert "σ luma" in text and "tin cậy" in text and "L* trung bình" in text
