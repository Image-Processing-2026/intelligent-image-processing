"""Test hàm thuần của scripts/benchmark_measurement.py (không cần bộ synthetic)."""

import math

import numpy as np
import pytest

from scripts.benchmark_measurement import (
    LUMA_NOISE_GAIN,
    _expected_illuminant,
    _orientation_error_deg,
    case_rows,
    summarize,
)
from src.analyzer_evaluator.measurement import Estimate, MeasurementReport


def _report(**sections: dict) -> MeasurementReport:
    """MeasurementReport giả với các giá trị cho trước (confidence 1)."""
    payload = {
        name: {key: Estimate(value=value, confidence=1.0) for key, value in values.items()}
        for name, values in sections.items()
    }
    return MeasurementReport(width=64, height=64, tile_size=64, **payload)


def test_orientation_error_wraps_at_180() -> None:
    assert _orientation_error_deg(179.0, 1.0) == pytest.approx(2.0)
    assert _orientation_error_deg(10.0, 100.0) == pytest.approx(90.0)


def test_expected_illuminant_neutral_gains_keep_control() -> None:
    control = np.array([0.3, 0.4, 0.3])
    assert _expected_illuminant(control, (1.0, 1.0, 1.0)) == pytest.approx(control)
    warm = _expected_illuminant(np.ones(3) / 3, (1.2, 1.0, 0.75))
    assert warm[0] > warm[1] > warm[2]
    assert warm.sum() == pytest.approx(1.0)


def test_noise_truth_adds_control_sigma_in_quadrature() -> None:
    params = {"degradations": [{"type": "gaussian_noise", "sigma": 10.0}]}
    control = _report(noise={"sigma_luma": 1.0})
    report = _report(noise={"sigma_luma": 6.8})
    (row,) = case_rows("x__noise_s10", params, report, control)
    assert row["quantity"] == "noise_sigma_luma"
    assert row["truth"] == pytest.approx(math.hypot(1.0, LUMA_NOISE_GAIN * 10.0))


def test_motion_truth_is_perpendicular_and_summary_uses_orientation() -> None:
    params = {"degradations": [{"type": "motion_blur", "length": 15, "angle": 100.0}]}
    control = _report(sharpness={"motion_coherence": 0.1, "motion_angle_deg": 40.0})
    report = _report(sharpness={"motion_coherence": 0.7, "motion_angle_deg": 12.0})
    rows = case_rows("x__blur_m15", params, report, control)
    assert rows[0]["truth"] == pytest.approx(10.0)
    assert summarize(rows)["motion_angle_deg"]["mae"] == pytest.approx(2.0)


def test_missing_estimate_is_counted_not_scored() -> None:
    params = {"degradations": [{"type": "gaussian_blur", "sigma": 1.5}]}
    rows = case_rows("x__blur_g1.5", params, _report(), _report())
    summary = summarize(rows)
    assert summary["blur_defocus_sigma"]["n"] == 0
    assert summary["blur_defocus_sigma"]["missing"] == 1
