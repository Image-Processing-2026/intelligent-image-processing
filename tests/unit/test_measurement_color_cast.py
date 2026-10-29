"""Kiểm thử chọn pixel trung tính lặp: ám màu mạnh không bị đo thiếu, màu bão hòa không trôi vào."""

import math

import numpy as np
import pytest

from src.analyzer_evaluator.measurement import measure_image


def _encode_srgb(linear: np.ndarray) -> np.ndarray:
    encoded = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)
    return np.clip(np.round(encoded * 255.0), 0, 255).astype(np.uint8)


def _grey_blocks(gains: np.ndarray, seed: int = 0, blocks: int = 16) -> np.ndarray:
    """Khối xám ngẫu nhiên (tối tới sáng) nhân kênh trên RGB tuyến tính."""
    rng = np.random.default_rng(seed)
    levels = rng.uniform(0.02, 0.6, (blocks, blocks, 1)).astype(np.float32)
    linear = np.kron(levels, np.ones((16, 16, 1), dtype=np.float32))
    return _encode_srgb(np.clip(linear * gains.astype(np.float32), 0.0, 1.0))


def _true_angle_deg(gains: np.ndarray) -> float:
    unit = gains / np.linalg.norm(gains)
    return math.degrees(math.acos(float(np.dot(unit, np.ones(3) / math.sqrt(3.0)))))


@pytest.mark.parametrize(
    "gains", [np.array([1.4, 1.0, 0.7]), np.array([0.7, 1.0, 1.4]), np.array([1.25, 1.0, 0.8])]
)
def test_strong_cast_is_recovered_within_3_degrees(gains: np.ndarray) -> None:
    report = measure_image(_grey_blocks(gains)).report
    estimated = report.color["illuminant_angle_deg"].value
    assert abs(estimated - _true_angle_deg(gains)) <= 3.0


def test_strong_cast_starts_from_few_neutral_pixels() -> None:
    # Phần lớn khối xám có C* > 12 dưới ám mạnh: cổng C* một lượt chỉ thấy khối tối.
    gains = np.array([1.4, 1.0, 0.7])
    report = measure_image(_grey_blocks(gains)).report
    assert report.color["neutral_fraction"].value < 0.5
    assert abs(report.color["illuminant_angle_deg"].value - _true_angle_deg(gains)) <= 3.0


def test_saturated_colour_does_not_drift_into_neutral_set() -> None:
    # Nửa trên: cỏ bão hòa (C* > 30); nửa dưới: khối xám thật. Nguồn sáng phải gần trung tính.
    image = _grey_blocks(np.ones(3))
    image[: image.shape[0] // 2] = (70, 150, 40)
    report = measure_image(image).report
    assert report.color["illuminant_angle_deg"].value < 1.0
