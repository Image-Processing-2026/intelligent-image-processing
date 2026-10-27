"""Kiểm thử đường dự phòng của ước lượng nguồn sáng khi không có pixel trung tính (lane color).

Ám rất mạnh đẩy mọi pixel xám ra khỏi cổng C* < 12: nới cổng từng bậc, rồi tới mọi pixel không
cháy. Confidence phải thấp hơn rõ đường chuẩn, và hoàng hôn không được thành ám lạnh.
"""

import math

import cv2
import numpy as np
import pytest

from src.analyzer_evaluator.measurement import color as color_module
from src.analyzer_evaluator.measurement import measure_image
from src.analyzer_evaluator.measurement.calibration import defect_severities


def _encode_srgb(linear: np.ndarray) -> np.ndarray:
    encoded = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)
    return np.clip(np.round(encoded * 255.0), 0, 255).astype(np.uint8)


def _grey_blocks(gains: np.ndarray, low: float, high: float, seed: int = 0) -> np.ndarray:
    """Khối xám ngẫu nhiên trong [low, high] (tuyến tính) nhân kênh, mã hóa sRGB."""
    rng = np.random.default_rng(seed)
    levels = rng.uniform(low, high, (16, 16, 1)).astype(np.float32)
    linear = np.kron(levels, np.ones((16, 16, 1), dtype=np.float32))
    return _encode_srgb(np.clip(linear * gains.astype(np.float32), 0.0, 1.0))


def _true_angle_deg(gains: np.ndarray) -> float:
    unit = gains / np.linalg.norm(gains)
    return math.degrees(math.acos(float(np.dot(unit, np.ones(3) / math.sqrt(3.0)))))


def _min_chroma(image: np.ndarray) -> float:
    lab = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    return float(np.hypot(lab[..., 1], lab[..., 2]).min())


def test_standard_path_records_method_and_gate() -> None:
    report = measure_image(_grey_blocks(np.array([1.1, 1.0, 0.9]), 0.2, 0.6)).report
    assert report.color["illuminant_method"].value == color_module.METHOD_NEUTRAL
    assert report.color["illuminant_gate_chroma"].value == color_module.NEUTRAL_CHROMA


@pytest.mark.parametrize(
    "gains, gate",
    [(np.array([1.5, 1.0, 0.55]), 18.0), (np.array([1.6, 1.0, 0.45]), 24.0)],
)
def test_relaxed_gate_estimates_strong_cast_with_reduced_confidence(
    gains: np.ndarray, gate: float
) -> None:
    image = _grey_blocks(gains, 0.1, 0.6)
    assert _min_chroma(image) > color_module.NEUTRAL_CHROMA  # không còn pixel C* < 12
    report = measure_image(image).report
    assert report.color["illuminant_method"].value == color_module.METHOD_RELAXED_GATE
    assert report.color["illuminant_gate_chroma"].value == gate
    angle = report.color["illuminant_angle_deg"]
    assert abs(angle.value - _true_angle_deg(gains)) <= 3.0
    assert angle.confidence <= color_module.NEUTRAL_CHROMA / gate + 1e-6
    assert any("relaxed neutral gate" in note for note in report.notes)


def test_all_pixel_fallback_when_no_pixel_is_near_neutral() -> None:
    # Ám cực mạnh (kiểu giấy trắng dưới đèn sợi đốt): mọi pixel có C* > 30.
    gains = np.array([2.0, 1.0, 0.2])
    image = _grey_blocks(gains, 0.2, 0.45)
    assert _min_chroma(image) > color_module.CAST_MAX_CHROMA
    report = measure_image(image).report
    assert report.color["illuminant_method"].value == color_module.METHOD_ALL_PIXELS
    assert "illuminant_gate_chroma" not in report.color
    angle = report.color["illuminant_angle_deg"]
    assert abs(angle.value - _true_angle_deg(gains)) <= 3.0
    assert angle.confidence <= color_module.ALL_PIXELS_CONFIDENCE + 1e-6
    severities = defect_severities(report)
    assert severities["color_cast_warm"][0] >= 1
    assert severities["color_cast_cool"][0] == 0


def test_fallback_confidence_is_clearly_below_standard_path() -> None:
    standard = measure_image(_grey_blocks(np.array([1.2, 1.0, 0.8]), 0.1, 0.6)).report
    fallback = measure_image(_grey_blocks(np.array([2.0, 1.0, 0.2]), 0.2, 0.45)).report
    assert standard.color["illuminant_method"].value == color_module.METHOD_NEUTRAL
    assert (
        fallback.color["illuminant_angle_deg"].confidence
        <= 0.5 * standard.color["illuminant_angle_deg"].confidence
    )


def _synthetic_sunset() -> np.ndarray:
    """Trời hoàng hôn cam–đỏ có mây (texture), nửa dưới là bóng đen (L* < 15)."""
    size = 256
    rng = np.random.default_rng(1)
    ramp = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None]
    sky = np.stack(
        [
            np.full((size, size), 0.9, dtype=np.float32),
            np.broadcast_to(0.25 + 0.35 * ramp, (size, size)),
            np.broadcast_to(0.05 + 0.10 * ramp, (size, size)),
        ],
        axis=2,
    )
    cloud = cv2.GaussianBlur(rng.uniform(0.6, 1.0, (size, size)).astype(np.float32), (0, 0), 6)
    linear = sky * cloud[..., None]
    linear[int(size * 0.75) :] = 0.004
    return _encode_srgb(np.clip(linear, 0.0, 1.0))


def test_synthetic_sunset_is_never_a_cool_cast() -> None:
    report = measure_image(_synthetic_sunset()).report
    severities = defect_severities(report)
    assert severities["color_cast_cool"][0] == 0
    light = [report.color[f"illuminant_{c}"].value for c in "rgb"]
    assert light[0] > light[2]  # ước lượng (đường dự phòng) theo hướng ấm


@pytest.mark.parametrize("rgb", [(230, 140, 60), (168, 128, 108)])
def test_flat_colour_field_still_has_no_estimate(rgb: tuple) -> None:
    # Mặt màu phẳng (C* > 30, và 12 < C* < 30 nên cổng nới bắt được): không có cạnh thì không
    # phân biệt được màu vật với màu nguồn sáng, nên không ước lượng.
    image = np.full((64, 64, 3), rgb, dtype=np.uint8)
    report = measure_image(image).report
    assert "illuminant_angle_deg" not in report.color
    assert "illuminant_method" not in report.color
    assert any("no image structure" in note for note in report.notes)
