"""Kiểm thử đo màu: nguồn sáng từ pixel trung tính, cảnh ấm, C* — ảnh synthetic trong test."""

import math

import numpy as np
import pytest

from src.analyzer_evaluator.measurement import measure_image

SIZE = 256


def _encode_srgb(linear: np.ndarray) -> np.ndarray:
    """Mã hóa tuyến tính [0, 1] sang sRGB uint8 (đảo của srgb_to_linear)."""
    encoded = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)
    return np.clip(np.round(encoded * 255.0), 0, 255).astype(np.uint8)


def _gray_patches_with_gain(gains: np.ndarray, seed: int = 0) -> np.ndarray:
    """Patch xám ngẫu nhiên nhân kênh biết trước trên RGB tuyến tính, rồi mã hóa sRGB."""
    rng = np.random.default_rng(seed)
    level = rng.uniform(0.2, 0.7, (SIZE, SIZE, 1)).astype(np.float32)
    return _encode_srgb(np.clip(level * gains.astype(np.float32), 0.0, 1.0))


def _true_angle_deg(gains: np.ndarray) -> float:
    """Góc giữa vector nhân kênh và trục xám (1, 1, 1)."""
    unit = gains / np.linalg.norm(gains)
    return math.degrees(math.acos(float(np.dot(unit, np.ones(3) / math.sqrt(3.0)))))


@pytest.mark.parametrize(
    "gains",
    [
        np.array([1.1, 1.0, 0.9]),
        np.array([0.9, 1.0, 1.15]),
        np.array([1.0, 1.08, 0.92]),
    ],
)
def test_channel_gains_on_grey_patches_recovered_within_3_degrees(gains: np.ndarray) -> None:
    report = measure_image(_gray_patches_with_gain(gains)).report
    estimated = report.color["illuminant_angle_deg"].value
    assert abs(estimated - _true_angle_deg(gains)) <= 3.0


def test_warm_scene_with_neutral_patches_reports_no_cast() -> None:
    image = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    # Nửa trên: trời hoàng hôn ấm (C* lớn), không dùng để ước lượng nguồn sáng.
    image[: SIZE // 2] = (230, 140, 60)
    # Nửa dưới: patch xám thật (trung tính) — nguồn sáng phải được coi là trung tính.
    image[SIZE // 2 :] = (128, 128, 128)
    report = measure_image(image).report
    assert report.color["illuminant_angle_deg"].value < 1.0
    assert report.color["chroma_p95"].value > 20.0  # trời vẫn được ghi nhận là ấm


def test_scene_without_neutral_pixels_has_no_illuminant_keys_and_a_note() -> None:
    image = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    image[...] = (230, 140, 60)
    report = measure_image(image).report
    assert report.color["neutral_fraction"].value == pytest.approx(0.0)
    assert "illuminant_angle_deg" not in report.color
    assert any("neutral" in note for note in report.notes)


def test_neutral_fraction_counts_only_unclipped_pixels() -> None:
    image = np.full((SIZE, SIZE, 3), 128, dtype=np.uint8)
    image[:, : SIZE // 2] = 255  # nửa ảnh cháy không được tính là trung tính
    report = measure_image(image).report
    assert report.color["neutral_fraction"].value == pytest.approx(0.5, abs=0.01)


def test_warm_image_has_high_chroma_tile_map() -> None:
    image = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    image[...] = (200, 120, 60)
    result = measure_image(image, tile_size=64)
    chroma = result.maps["color.chroma"]
    assert chroma.shape == (4, 4)
    assert float(np.nanmin(chroma)) > 20.0


@pytest.mark.parametrize("fill", [0, 255])
def test_black_and_white_images_have_finite_colour_report(fill: int) -> None:
    report = measure_image(np.full((64, 64, 3), fill, dtype=np.uint8)).report
    assert all(np.isfinite(e.value) for e in report.color.values())


def test_grayscale_input_is_neutral() -> None:
    gray = np.full((SIZE, SIZE), 110, dtype=np.uint8)
    report = measure_image(gray).report
    assert report.color["illuminant_angle_deg"].value == pytest.approx(0.0, abs=1e-3)
    assert report.color["neutral_fraction"].value == pytest.approx(1.0)
