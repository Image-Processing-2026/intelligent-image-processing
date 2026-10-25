"""Kiểm thử hiệu chỉnh severity (hợp đồng C4): bảng calibration.yaml và severity_from_measurement."""

import math
from pathlib import Path
from typing import Dict

import pytest

from src.agent.state import DEFECT_TYPES
from src.analyzer_evaluator.measurement import Estimate, MeasurementReport, RegionStats
from src.analyzer_evaluator.measurement import calibration as calib
from src.analyzer_evaluator.measurement.calibration import (
    calibration_version,
    defect_severities,
    load_calibration,
    severity_for_score,
    severity_from_calibration,
    severity_from_measurement,
)

P6_DEFECTS = ("motion_blur", "jpeg_artifacts", "haze", "highlight_clipping")


def _est(value: float, confidence: float = 1.0) -> Estimate:
    return Estimate(value=value, confidence=confidence, n_samples=100)


def _report(
    width: int = 1000, height: int = 750, **sections: Dict[str, float]
) -> MeasurementReport:
    """Báo cáo tối thiểu: sections = {"noise": {"sigma_luma": 5.0}, ...}."""
    return MeasurementReport(
        width=width,
        height=height,
        tile_size=64,
        **{name: {k: _est(v) for k, v in values.items()} for name, values in sections.items()},
    )


def _thresholds(defect: str) -> list:
    return [float(t) for t in load_calibration()["defects"][defect]["thresholds"]]


def test_calibration_file_is_complete_and_consistent() -> None:
    data = load_calibration()
    assert calibration_version() == str(data["version"])
    assert {"date", "dataset", "reference_long_side", "levels"} <= set(data)
    for defect in (*DEFECT_TYPES, *P6_DEFECTS):
        assert defect in data["defects"], defect
    for defect, entry in data["defects"].items():
        thresholds = [float(t) for t in entry["thresholds"]]
        assert thresholds == sorted(thresholds), defect
        assert len(thresholds) in (0, 3), defect
        if entry["score"] is not None:
            assert entry["score"] in calib.SCORE_FUNCTIONS, defect
        if thresholds:
            assert 0.0 <= float(entry.get("reliability", 1.0)) <= 1.0


def test_severity_for_score_counts_reached_thresholds() -> None:
    thresholds = [1.0, 2.0, math.inf]
    assert severity_for_score(0.5, thresholds) == 0
    assert severity_for_score(1.0, thresholds) == 1
    assert severity_for_score(5.0, thresholds) == 2


@pytest.mark.parametrize("defect", ["backlit_subject", "not_a_defect"])
def test_unmeasurable_or_unknown_defect_returns_zero(defect: str) -> None:
    assert severity_from_measurement(defect, _report(noise={"sigma_luma": 30.0})) == (0, 0.0)


def test_missing_measurement_returns_zero() -> None:
    assert severity_from_measurement("noise", _report()) == (0, 0.0)


def test_noise_severity_is_monotone_and_bounded() -> None:
    previous = 0
    for sigma in (0.0, 1.0, 3.0, 6.0, 12.0, 25.0):
        severity, confidence = severity_from_measurement(
            "noise", _report(noise={"sigma_luma": sigma})
        )
        assert previous <= severity <= 3
        assert 0.0 <= confidence <= 1.0
        previous = severity
    assert previous == 3


def test_dark_concentrated_noise_is_less_severe() -> None:
    sigma = _thresholds("noise")[0] * 1.1
    flat = severity_from_measurement("noise", _report(noise={"sigma_luma": sigma}))
    dark = severity_from_measurement(
        "noise", _report(noise={"sigma_luma": sigma, "sigma_dark_ratio": 3.0})
    )
    assert dark[0] < flat[0]


def test_chroma_noise_alone_is_detected() -> None:
    report = _report(noise={"sigma_luma": 0.1, "sigma_cb": 8.0, "sigma_cr": 8.0})
    assert severity_from_measurement("noise", report)[0] >= 1


def test_region_stats_use_map_keys() -> None:
    region = RegionStats(region="face", area_fraction=0.2, values={"exposure.mean_l": _est(10.0)})
    assert severity_from_measurement("underexposed", region)[0] >= 1
    assert severity_from_measurement("overexposed", region)[0] == 0
    # Vùng không có phân vị L* nên tương phản không đo được
    assert severity_from_measurement("low_contrast", region) == (0, 0.0)


def test_blur_is_normalised_to_reference_resolution() -> None:
    sigma = _thresholds("blur")[0] * 1.2
    small = _report(width=1024, height=768, sharpness={"defocus_sigma": sigma})
    large = _report(width=4096, height=3072, sharpness={"defocus_sigma": sigma})
    assert severity_from_measurement("blur", small)[0] >= 1
    assert severity_from_measurement("blur", large)[0] == 0
    region = RegionStats(
        region="person", area_fraction=0.3, values={"sharpness.defocus_sigma": _est(sigma)}
    )
    assert severity_from_measurement("blur", region)[0] >= 1
    assert severity_from_measurement("blur", region, long_side=4096)[0] == 0


def test_motion_needs_confirmation_gate() -> None:
    params = load_calibration()["defects"]["motion_blur"]["params"]
    extent = max(float(params["min_extent"]) * 3.0, 3.0)
    weak = _report(sharpness={"motion_anisotropy": 1.0, "motion_extent_sigma": extent})
    strong = _report(sharpness={"motion_anisotropy": 3.0, "motion_extent_sigma": extent})
    assert severity_from_measurement("motion_blur", weak) == (0, 0.0)
    assert severity_from_measurement("motion_blur", strong)[0] >= 1
    # blur lấy mức cao hơn giữa defocus và mờ chuyển động
    assert severity_from_measurement("blur", strong)[0] >= 1


def test_noisy_image_needs_longer_motion_extent() -> None:
    params = load_calibration()["defects"]["motion_blur"]["params"]
    extent = float(params["min_extent"]) + 0.5 * float(params["noisy_extra_extent"])
    sharp = {"motion_anisotropy": 3.0, "motion_extent_sigma": extent}
    clean = _report(sharpness=sharp, noise={"sigma_luma": 0.5})
    noisy = _report(sharpness=sharp, noise={"sigma_luma": float(params["noisy_sigma"]) + 1.0})
    assert severity_from_measurement("motion_blur", clean)[0] >= 1
    assert severity_from_measurement("motion_blur", noisy) == (0, 0.0)


@pytest.mark.parametrize(
    "rgb, defect",
    [((0.45, 0.33, 0.22), "color_cast_warm"), ((0.22, 0.33, 0.45), "color_cast_cool")],
)
def test_colour_cast_direction(rgb: tuple, defect: str) -> None:
    report = _report(color=dict(zip(("illuminant_r", "illuminant_g", "illuminant_b"), rgb)))
    severities = defect_severities(report)
    assert severities[defect][0] >= 1
    other = "color_cast_cool" if defect == "color_cast_warm" else "color_cast_warm"
    assert severities[other][0] == 0


def test_prior_thresholds_have_reduced_confidence() -> None:
    report = _report(exposure={"largest_clip_blob": 0.5})
    severity, confidence = severity_from_measurement("highlight_clipping", report)
    assert severity == 3
    assert confidence <= calib.PRIOR_CONFIDENCE_SCALE


def test_confidence_grows_with_distance_from_threshold() -> None:
    t1 = _thresholds("noise")[0]
    near = severity_from_measurement("noise", _report(noise={"sigma_luma": t1 * 1.01}))
    far = severity_from_measurement("noise", _report(noise={"sigma_luma": 0.0}))
    assert near[1] < far[1]


def test_combine_max_applies_when_own_score_is_missing() -> None:
    calibration = {
        "reference_long_side": 1024,
        "defects": {
            "blur": {"score": "defocus", "thresholds": [], "combine_max": ["noise"]},
            "noise": {"score": "noise_sigma", "params": {}, "thresholds": [1.0, 2.0, 3.0]},
        },
    }
    report = _report(noise={"sigma_luma": 2.5})
    assert severity_from_calibration("blur", report, calibration)[0] == 2


def test_invalid_calibration_file_raises(tmp_path: Path) -> None:
    broken = tmp_path / "calibration.yaml"
    broken.write_text("defects: {}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_calibration(str(broken))
