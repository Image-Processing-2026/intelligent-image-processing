"""Kiểm thử đo phơi sáng: clip theo kênh, vùng cháy (blob ≥ 0.5 %), L*, key — ảnh synthetic."""

import numpy as np
import pytest

from src.analyzer_evaluator.measurement import measure_image

SIZE = 256


def _grey(level: int = 128) -> np.ndarray:
    return np.full((SIZE, SIZE, 3), level, dtype=np.uint8)


def test_small_specular_dot_is_not_reported_as_burnt() -> None:
    image = _grey(120)
    image[100:105, 100:105] = 255  # 25 px ≈ 0.04 % ảnh: phản xạ gương
    report = measure_image(image).report
    assert report.exposure["clip_high_any"].value > 0.0
    assert report.exposure["largest_clip_blob"].value == 0.0


def test_large_burnt_area_is_reported_with_its_fraction() -> None:
    image = _grey(120)
    image[64:128, 64:128] = 255  # 64×64 = 1/16 ảnh
    report = measure_image(image).report
    assert report.exposure["largest_clip_blob"].value == pytest.approx(64 * 64 / (SIZE * SIZE))


def test_clip_any_and_all_differ_when_only_one_channel_saturates() -> None:
    image = _grey(100)
    image[..., 0] = 255  # chỉ kênh đỏ cháy
    report = measure_image(image).report
    assert report.exposure["clip_high_any"].value == pytest.approx(1.0)
    assert report.exposure["clip_high_all"].value == pytest.approx(0.0)


def test_low_clip_is_measured_per_channel() -> None:
    image = _grey(100)
    image[:64] = 0  # một phần tư ảnh đen tuyền
    report = measure_image(image).report
    assert report.exposure["clip_low_all"].value == pytest.approx(0.25)
    assert report.exposure["clip_low_any"].value == pytest.approx(0.25)


def test_mid_grey_has_expected_lightness_and_key() -> None:
    report = measure_image(_grey(128)).report
    # L* của sRGB 128 ≈ 53.6; key của ảnh phẳng ≈ luminance tuyến tính 0.216
    assert report.exposure["mean_l"].value == pytest.approx(53.6, abs=0.5)
    assert report.exposure["key"].value == pytest.approx(0.216, abs=0.01)
    assert report.exposure["p5_l"].value == pytest.approx(report.exposure["mean_l"].value, abs=0.1)


def test_percentiles_are_ordered_around_mean() -> None:
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, size=(SIZE, SIZE, 3), dtype=np.uint8)
    report = measure_image(image).report
    p5 = report.exposure["p5_l"].value
    p95 = report.exposure["p95_l"].value
    assert p5 < report.exposure["mean_l"].value < p95


def test_exposure_maps_follow_tile_grid() -> None:
    image = _grey(200)
    image[:, :32] = 255
    result = measure_image(image, tile_size=64)
    clip_high = result.maps["exposure.clip_high"]
    assert clip_high.shape == (4, 4)
    assert clip_high[0, 0] == pytest.approx(0.5)  # nửa tile (32/64 cột) cháy
    assert clip_high[0, 3] == pytest.approx(0.0)
    assert result.maps["exposure.mean_l"].shape == (4, 4)
