"""Kiểm thử ước lượng nhiễu (Liu 2013), σ(I), Immerkær và bản đồ tile — tạo ảnh synthetic trong test."""

import numpy as np
import pytest

from src.analyzer_evaluator.measurement import measure_image
from src.analyzer_evaluator.measurement.noise import measure_noise
from src.analyzer_evaluator.measurement.tiles import tile_grid_shape

SIZE = 256


def _texture(amplitude: float) -> np.ndarray:
    """Texture hình sin có biên độ amplitude quanh 128 (không nhiễu)."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    return 128.0 + amplitude * np.sin(2 * np.pi * xx / 23.0) * np.cos(2 * np.pi * yy / 31.0)


def _noisy_rgb(sigma: float, amplitude: float, seed: int = 0) -> np.ndarray:
    """RGB uint8: texture + nhiễu Gaussian độc lập trên từng kênh (luma σ ≈ 0.69·σ)."""
    rng = np.random.default_rng(seed)
    base = _texture(amplitude)
    channels = [base + rng.normal(0.0, sigma, base.shape) for _ in range(3)]
    return np.clip(np.stack(channels, axis=2), 0, 255).astype(np.uint8)


def _noisy_gray(sigma: float, amplitude: float, seed: int = 0) -> np.ndarray:
    """Ảnh xám (H, W) uint8: texture + nhiễu Gaussian σ, nên σ luma đúng bằng σ."""
    rng = np.random.default_rng(seed)
    gray = _texture(amplitude) + rng.normal(0.0, sigma, (SIZE, SIZE))
    return np.clip(gray, 0, 255).astype(np.uint8)


def _gray_rgb(sigma: float, amplitude: float, seed: int = 0) -> np.ndarray:
    """RGB có ba kênh bằng nhau từ ảnh xám: chroma không nhiễu, σ luma = σ."""
    return np.stack([_noisy_gray(sigma, amplitude, seed)] * 3, axis=2)


@pytest.mark.parametrize("amplitude", [0.0, 25.0])
@pytest.mark.parametrize("sigma", [2.0, 5.0, 10.0, 20.0, 30.0])
def test_sigma_luma_within_15_percent_on_flat_and_textured_images(
    sigma: float, amplitude: float
) -> None:
    result = measure_image(_gray_rgb(sigma, amplitude))
    estimate = result.report.noise["sigma_luma"]
    assert abs(estimate.value - sigma) / sigma <= 0.15
    assert estimate.confidence > 0.9


def test_clean_texture_has_low_sigma() -> None:
    rgb = np.clip(np.stack([_texture(40.0)] * 3, axis=2), 0, 255).astype(np.uint8)
    result = measure_image(rgb)
    assert result.report.noise["sigma_luma"].value < 1.0


def test_constant_image_has_zero_noise_and_no_nan() -> None:
    result = measure_image(np.full((SIZE, SIZE, 3), 90, dtype=np.uint8))
    for estimate in result.report.noise.values():
        assert np.isfinite(estimate.value)
    assert result.report.noise["sigma_luma"].value == pytest.approx(0.0, abs=1e-6)


def test_black_and_white_images_do_not_crash() -> None:
    for fill in (0, 255):
        result = measure_image(np.full((64, 64, 3), fill, dtype=np.uint8))
        assert all(np.isfinite(e.value) for e in result.report.noise.values())


def test_chroma_noise_is_reported_for_colour_images() -> None:
    result = measure_image(_noisy_rgb(10.0, 0.0))
    assert result.report.noise["sigma_cb"].value > 1.0
    assert result.report.noise["sigma_cr"].value > 1.0


def test_grayscale_reports_zero_chroma_with_note() -> None:
    result = measure_image(_noisy_gray(10.0, 0.0))
    assert result.report.noise["sigma_cb"].value == 0.0
    assert result.report.noise["sigma_cb"].confidence == 0.0
    assert any("grayscale" in note for note in result.report.notes)
    assert result.report.noise["sigma_luma"].value == pytest.approx(10.0, rel=0.15)


def test_sigma_dark_ratio_is_below_one_for_signal_dependent_noise() -> None:
    rng = np.random.default_rng(4)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    # Nhiễu tăng theo độ sáng: vùng tối (trái) nhiễu ít, vùng sáng (phải) nhiễu nhiều.
    level = 40.0 + 170.0 * xx / SIZE
    sigma_map = 1.0 + 0.12 * level
    noisy = level + rng.normal(0.0, 1.0, level.shape) * sigma_map
    gray = np.clip(noisy, 0, 255).astype(np.uint8)
    result = measure_image(np.stack([gray] * 3, axis=2))
    ratio = result.report.noise["sigma_dark_ratio"].value
    assert 0.0 < ratio < 1.0


def test_immerkaer_is_close_on_flat_noise_and_marked_secondary() -> None:
    result = measure_image(_gray_rgb(10.0, 0.0))
    estimate = result.report.noise["immerkaer"]
    assert abs(estimate.value - 10.0) / 10.0 <= 0.15
    assert estimate.confidence == pytest.approx(0.5)


def test_sigma_map_is_on_tile_grid_and_close_to_truth() -> None:
    result = measure_image(_gray_rgb(10.0, 0.0), tile_size=64)
    grid = result.maps["noise.sigma_luma"]
    assert grid.shape == tile_grid_shape(SIZE, SIZE, 64)
    finite = grid[np.isfinite(grid)]
    assert finite.size == grid.size
    assert abs(float(np.median(finite)) - 10.0) / 10.0 <= 0.15


def test_few_flat_patches_add_warning_and_keep_report_finite() -> None:
    rng = np.random.default_rng(5)
    tiny = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    result = measure_image(tiny)
    assert "sigma_luma" not in result.report.noise
    assert any("few flat patches" in note for note in result.report.notes)


def test_measure_noise_returns_sigma_for_downstream_denoising() -> None:
    rgb = _gray_rgb(8.0, 0.0)
    section, sigma = measure_noise(rgb, 64, grayscale=False)
    assert sigma is not None
    assert abs(sigma - 8.0) / 8.0 <= 0.15
    assert section.estimates["sigma_luma"].value == pytest.approx(sigma)
