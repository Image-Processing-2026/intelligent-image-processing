"""Kiểm thử blockiness (JPEG) và dark channel (sương mù) — ảnh synthetic trong test."""

import cv2
import numpy as np

from src.analyzer_evaluator.measurement import measure_image

SIZE = 256


def _smooth_texture(seed: int = 0) -> np.ndarray:
    """Ảnh màu mượt (nhiễu lọc thấp tần): JPEG chất lượng thấp để lộ biên khối rõ."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.0, 1.0, (SIZE, SIZE, 3)).astype(np.float32)
    smooth = cv2.GaussianBlur(base, (0, 0), 6.0)
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())
    return np.round(smooth * 255).astype(np.uint8)


def _jpeg_round_trip(rgb: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    assert ok
    return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def test_heavy_jpeg_has_higher_blockiness_than_original() -> None:
    original = _smooth_texture()
    blocky = _jpeg_round_trip(original, quality=10)
    original_b = measure_image(original).report.compression["blockiness"].value
    blocky_b = measure_image(blocky).report.compression["blockiness"].value
    assert blocky_b > original_b + 0.1
    assert blocky_b > 1.2


def test_flat_image_has_neutral_blockiness_of_one() -> None:
    report = measure_image(np.full((SIZE, SIZE, 3), 90, dtype=np.uint8)).report
    assert report.compression["blockiness"].value == 1.0


def test_tiny_image_skips_blockiness_with_note() -> None:
    report = measure_image(np.zeros((6, 6, 3), dtype=np.uint8)).report
    assert "blockiness" not in report.compression
    assert any("too small" in note for note in report.notes)


def _hazy(rgb: np.ndarray, transmission: float, airlight: float) -> np.ndarray:
    """Mô hình sương I = J·t + A·(1 − t) trên RGB [0, 1]."""
    scene = rgb.astype(np.float32) / 255.0
    hazed = scene * transmission + airlight * (1.0 - transmission)
    return np.round(np.clip(hazed, 0.0, 1.0) * 255).astype(np.uint8)


def test_haze_raises_dark_channel_median() -> None:
    clear = _smooth_texture(seed=3)
    clear_dc = measure_image(clear).report.haze["dark_channel_median"].value
    hazy_dc = measure_image(_hazy(clear, 0.6, 0.9)).report.haze["dark_channel_median"].value
    assert hazy_dc > clear_dc + 0.2


def test_sky_region_is_excluded_from_dark_channel() -> None:
    image = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    image[: SIZE // 2] = (200, 215, 240)  # trời sáng, phẳng, ít màu
    image[SIZE // 2 :] = (25, 25, 25)  # đất tối
    value = measure_image(image).report.haze["dark_channel_median"].value
    assert value < 0.2  # nếu trời được tính, median sẽ ≈ 0.78


def test_uniform_bright_image_without_dark_pixels_reads_as_hazy() -> None:
    image = np.full((SIZE, SIZE, 3), 0, dtype=np.uint8)
    image[...] = (210, 225, 250)  # không có chỗ nào tối: dark channel rất cao
    value = measure_image(image).report.haze["dark_channel_median"].value
    assert value > 0.7


def test_grayscale_haze_and_compression_are_finite() -> None:
    gray = _smooth_texture(seed=5)[..., 0]
    report = measure_image(gray).report
    assert np.isfinite(report.haze["dark_channel_median"].value)
    assert np.isfinite(report.compression["blockiness"].value)
