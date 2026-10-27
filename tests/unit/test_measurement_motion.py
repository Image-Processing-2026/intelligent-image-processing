"""Kiểm thử xác nhận mờ chuyển động: dị hướng σ defocus theo hướng pháp tuyến cạnh (lane calib)."""

import math

import cv2
import numpy as np
import pytest

from src.analyzer_evaluator.measurement import measure_image
from src.analyzer_evaluator.measurement.calibration import severity_from_measurement

SIZE = 384


def _disc_scene(seed: int = 0) -> np.ndarray:
    """Cảnh nhiều đĩa màu: cạnh đủ mọi hướng."""
    rng = np.random.default_rng(seed)
    image = np.full((SIZE, SIZE, 3), 120, dtype=np.uint8)
    for _ in range(140):
        center = (int(rng.integers(0, SIZE)), int(rng.integers(0, SIZE)))
        colour = tuple(int(v) for v in rng.integers(20, 235, 3))
        cv2.circle(image, center, int(rng.integers(6, 30)), colour, -1, lineType=cv2.LINE_AA)
    return image


def _motion_blur(image: np.ndarray, length: int, angle_deg: float) -> np.ndarray:
    """Kernel đường thẳng dài `length` px theo góc `angle_deg` (trục y hướng xuống)."""
    kernel = np.zeros((length, length), dtype=np.float32)
    center = (length - 1) / 2.0
    dx = math.cos(math.radians(angle_deg)) * center
    dy = math.sin(math.radians(angle_deg)) * center
    start = (round(center - dx), round(center - dy))
    end = (round(center + dx), round(center + dy))
    cv2.line(kernel, start, end, 1.0, 1)
    kernel /= kernel.sum()
    blurred = cv2.filter2D(image.astype(np.float32), -1, kernel, borderType=cv2.BORDER_REFLECT)
    return np.clip(np.rint(blurred), 0, 255).astype(np.uint8)


def _orientation_error(a: float, b: float) -> float:
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


@pytest.mark.parametrize("length, angle", [(9, 30.0), (15, 120.0)])
def test_motion_blur_is_confirmed_with_direction(length: int, angle: float) -> None:
    report = measure_image(_motion_blur(_disc_scene(), length, angle)).report
    sharpness = report.sharpness
    assert sharpness["motion_anisotropy"].value > 1.5
    assert _orientation_error(sharpness["motion_blur_angle_deg"].value, angle) <= 20.0
    severity, confidence = severity_from_measurement("motion_blur", report)
    assert severity >= 1
    assert 0.0 < confidence <= 1.0
    # blur gộp mờ chuyển động đã xác nhận (combine_max)
    assert severity_from_measurement("blur", report)[0] >= severity


def test_clean_scene_is_isotropic_and_not_confirmed() -> None:
    report = measure_image(_disc_scene()).report
    assert report.sharpness["motion_anisotropy"].value < 1.2
    assert severity_from_measurement("motion_blur", report) == (0, 0.0)


def test_gaussian_blur_is_blur_but_not_motion() -> None:
    report = measure_image(cv2.GaussianBlur(_disc_scene(), (0, 0), 2.0)).report
    assert report.sharpness["motion_anisotropy"].value < 1.2
    assert severity_from_measurement("motion_blur", report) == (0, 0.0)
    assert severity_from_measurement("blur", report)[0] >= 1


def test_noise_lowers_motion_confidence_and_is_not_confirmed() -> None:
    rng = np.random.default_rng(1)
    noisy = np.clip(_disc_scene() + rng.normal(0.0, 10.0, (SIZE, SIZE, 3)), 0, 255)
    report = measure_image(noisy.astype(np.uint8)).report
    anisotropy = report.sharpness.get("motion_anisotropy")
    assert anisotropy is None or anisotropy.confidence <= 0.5
    assert severity_from_measurement("motion_blur", report) == (0, 0.0)


def test_flat_image_has_no_motion_keys() -> None:
    report = measure_image(np.full((128, 128, 3), 90, dtype=np.uint8)).report
    assert "motion_anisotropy" not in report.sharpness
    assert severity_from_measurement("motion_blur", report) == (0, 0.0)
