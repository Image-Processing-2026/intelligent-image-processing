"""Kiểm thử đo độ nét: độ rộng cạnh, defocus, hướng gradient — ảnh bậc thang Gaussian synthetic."""

import cv2
import numpy as np
import pytest

from src.analyzer_evaluator.measurement import measure_image

SIZE = 256


def _step(blur_sigma: float, noise_sigma: float = 0.0, seed: int = 0) -> np.ndarray:
    """Bậc thang dọc 60 → 200 tại x = 128, làm mờ Gaussian σ, thêm nhiễu tùy chọn."""
    image = np.full((SIZE, SIZE), 60.0, dtype=np.float32)
    image[:, SIZE // 2 :] = 200.0
    if blur_sigma > 0:
        image = cv2.GaussianBlur(image, (0, 0), blur_sigma)
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        image = image + rng.normal(0.0, noise_sigma, image.shape).astype(np.float32)
    gray = np.clip(image, 0, 255).astype(np.uint8)
    return np.stack([gray] * 3, axis=2)


def _sharpness(rgb: np.ndarray):
    return measure_image(rgb).report.sharpness


def test_edge_width_increases_monotonically_with_blur() -> None:
    widths = [_sharpness(_step(sigma))["edge_width_px"].value for sigma in (0.8, 1.5, 2.5, 4.0)]
    assert all(later > earlier for earlier, later in zip(widths, widths[1:]))


def test_defocus_sigma_increases_with_blur() -> None:
    sigmas = [_sharpness(_step(s))["defocus_sigma"].value for s in (1.0, 2.0, 3.0)]
    assert all(later > earlier for earlier, later in zip(sigmas, sigmas[1:]))


def test_laplacian_variance_drops_when_blurred() -> None:
    sharp = _sharpness(_step(0.0))["laplacian_var"].value
    blurred = _sharpness(_step(3.0))["laplacian_var"].value
    assert blurred < sharp


def test_noise_does_not_make_a_blurred_edge_look_sharper() -> None:
    clean = _sharpness(_step(3.0))["edge_width_px"].value
    noisy = _sharpness(_step(3.0, noise_sigma=10.0))["edge_width_px"].value
    assert noisy >= 0.9 * clean


def test_noise_does_not_lower_defocus_estimate() -> None:
    clean = _sharpness(_step(3.0))["defocus_sigma"].value
    noisy = _sharpness(_step(3.0, noise_sigma=10.0))["defocus_sigma"].value
    assert noisy >= 0.85 * clean


def test_normalised_width_scales_to_long_side_of_1000_px() -> None:
    report = _sharpness(_step(2.0))
    expected = report["edge_width_px"].value * 1000.0 / SIZE
    assert report["edge_width_norm"].value == pytest.approx(expected)


def test_motion_coherence_is_high_for_one_dominant_orientation() -> None:
    stripes = np.tile((np.arange(SIZE) // 8 % 2 * 180 + 30).astype(np.uint8)[:, None], (1, SIZE))
    rgb = np.stack([stripes] * 3, axis=2)
    report = _sharpness(rgb)
    assert report["motion_coherence"].value > 0.9
    # Gradient của vân ngang hướng theo trục y: góc ≈ 90°
    assert abs(report["motion_angle_deg"].value - 90.0) < 5.0


def test_flat_image_has_no_edges_and_no_nan() -> None:
    result = measure_image(np.full((SIZE, SIZE, 3), 77, dtype=np.uint8))
    assert "edge_width_px" not in result.report.sharpness
    assert result.report.sharpness["laplacian_var"].value == 0.0
    assert any("no gradient" in note for note in result.report.notes)


def test_sharpness_maps_are_on_tile_grid() -> None:
    result = measure_image(_step(2.0), tile_size=64)
    width_map = result.maps["sharpness.edge_width"]
    assert width_map.shape == (4, 4)
    assert np.isfinite(width_map[:, 1:3]).any()  # cạnh x = 128 nằm ở giữa lưới
    assert np.isnan(width_map[0, 0])  # tile bên trái đồng nhất không có cạnh
