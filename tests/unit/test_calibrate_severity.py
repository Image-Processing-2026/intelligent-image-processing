"""Unit test offline cho phần khớp ngưỡng (scripts/calibrate_severity.py), không cần bộ synthetic."""

import math
from typing import Any, Dict, List

import pytest

from scripts.calibrate_severity import (
    DEFECT_SPECS,
    balanced_mae,
    case_labels,
    cross_validate,
    fit_motion_gate,
    fit_thresholds,
    is_probe,
    predict,
    probe_rows,
    write_report,
)
from src.analyzer_evaluator.measurement import Estimate, MeasurementReport
from src.analyzer_evaluator.measurement.calibration import SCORE_FUNCTIONS


def test_fit_separable_labels_exactly() -> None:
    scores = [0.0, 0.1, 1.0, 1.1, 2.0, 2.2, 3.0, 3.5]
    labels = [0, 0, 1, 1, 2, 2, 3, 3]
    thresholds = fit_thresholds(scores, labels)
    assert [predict(s, thresholds) for s in scores] == labels
    # hòa chi phí: ngưỡng ở giữa khe
    assert thresholds[0] == pytest.approx(0.55)


def test_fit_minimises_absolute_error_with_overlap() -> None:
    scores = [0.0, 0.2, 0.4, 0.5, 0.6, 1.0]
    labels = [0, 0, 1, 0, 1, 1]
    thresholds = fit_thresholds(scores, labels)
    errors = sum(abs(predict(s, thresholds) - y) for s, y in zip(scores, labels))
    assert errors == 1


def test_unreached_levels_are_infinite() -> None:
    thresholds = fit_thresholds([0.0, 1.0, 2.0], [0, 1, 1])
    assert math.isfinite(thresholds[0])
    assert thresholds[1:] == [math.inf, math.inf]
    assert fit_thresholds([], []) == [math.inf] * 3


def test_unscored_case_predicts_zero() -> None:
    assert predict(None, [0.0, 1.0, 2.0]) == 0


def test_case_labels_add_p6_rules() -> None:
    params = {
        "degradations": [
            {"type": "motion_blur", "length": 15, "angle": 10.0},
            {"type": "jpeg", "quality": 20},
            {"type": "haze", "transmission": 0.6},
        ],
        "expected": {"defects": [{"type": "blur", "region": "full", "severity": 3}]},
    }
    labels = case_labels(params)
    assert labels == {"blur": 3, "motion_blur": 3, "jpeg_artifacts": 2, "haze": 2}
    q70 = case_labels(
        {"degradations": [{"type": "jpeg", "quality": 70}], "expected": {"defects": []}}
    )
    assert q70.get("jpeg_artifacts", 0) == 0


def _rows(scores: List[float], labels: List[int], sources: List[str]) -> List[Dict[str, Any]]:
    return [
        {"stem": f"c{i}", "source": s, "score": x, "label": y}
        for i, (x, y, s) in enumerate(zip(scores, labels, sources))
    ]


def test_balanced_mae_weights_classes_equally() -> None:
    rows = _rows([0.0] * 4, [0, 0, 0, 2], ["a"] * 4)
    assert balanced_mae(rows, [0, 0, 0, 0]) == pytest.approx(1.0)
    assert balanced_mae(rows, [0, 0, 0, 2]) == pytest.approx(0.0)


def test_cross_validation_beats_null_on_informative_score() -> None:
    scores = [0.0, 0.1, 2.0, 2.1] * 3
    labels = [0, 0, 2, 2] * 3
    sources = ["a"] * 4 + ["b"] * 4 + ["c"] * 4
    mae, balanced = cross_validate(_rows(scores, labels, sources))
    assert mae == pytest.approx(0.0)
    assert balanced < balanced_mae(_rows(scores, labels, sources), [0] * 12)


def _motion_report(anisotropy: float, extent: float) -> MeasurementReport:
    estimate = lambda v: Estimate(value=v, confidence=1.0, n_samples=100)  # noqa: E731
    return MeasurementReport(
        width=100,
        height=100,
        tile_size=64,
        sharpness={
            "motion_anisotropy": estimate(anisotropy),
            "motion_extent_sigma": estimate(extent),
        },
    )


def test_motion_gate_excludes_every_non_motion_case() -> None:
    cases, reports = [], {}
    points = {"m1": (2.0, 1.2, True), "m2": (1.6, 0.9, True), "n1": (1.1, 1.5, False)}
    points["n2"] = (1.9, 0.3, False)
    for stem, (a, e, motion) in points.items():
        degs = [{"type": "motion_blur", "length": 5, "angle": 0.0}] if motion else []
        cases.append({"stem": stem, "source": stem, "params": {"degradations": degs}})
        reports[stem] = _motion_report(a, e)
    gate = fit_motion_gate(cases, reports)
    for stem, (a, e, motion) in points.items():
        confirmed = a >= gate["min_anisotropy"] and e >= gate["min_extent"]
        assert confirmed == motion, stem


def test_every_spec_uses_a_known_score_function() -> None:
    for defect, spec in DEFECT_SPECS.items():
        assert spec["score"] is None or spec["score"] in SCORE_FUNCTIONS, defect


def _cct_case(stem: str, cct: int, label: str) -> Dict[str, Any]:
    params = {
        "degradations": [{"type": "illuminant_shift", "cct": cct}],
        "expected": {"defects": [{"type": label, "region": "full", "severity": 1}]},
    }
    return {"stem": stem, "source": stem.split("__")[0], "params": params}


def test_mild_cast_probes_are_not_fitted() -> None:
    assert is_probe(_cct_case("a__cct_5000", 5000, "color_cast_warm"))
    assert is_probe(_cct_case("a__cct_7000", 7000, "color_cast_cool"))
    assert not is_probe(_cct_case("a__cct_8000", 8000, "color_cast_cool"))
    assert not is_probe({"params": {"degradations": [{"type": "gamma", "gamma": 1.5}]}})


def test_probe_rows_and_report_section(tmp_path) -> None:
    calibration = {
        "defects": {
            "color_cast_warm": {"score": "cast_warm", "params": {}, "thresholds": [0.3, 0.8, 0.8]},
            "color_cast_cool": {"score": "cast_cool", "params": {}, "thresholds": [0.5]},
        }
    }
    light = {"illuminant_r": 0.45, "illuminant_g": 0.33, "illuminant_b": 0.22}
    report = MeasurementReport(
        width=10,
        height=10,
        tile_size=64,
        color={k: Estimate(value=v, confidence=1.0) for k, v in light.items()},
    )
    case = _cct_case("a__cct_5000", 5000, "color_cast_warm")
    rows = probe_rows([case], {"a__cct_5000": report}, calibration)
    assert rows[0]["color_cast_warm"] == (pytest.approx(math.log(0.45 / 0.22), abs=1e-3), 1)
    assert rows[0]["color_cast_cool"][1] == 0
    path = tmp_path / "report.md"
    stats = {"color_cast_warm": {"status": "prior"}}
    write_report(path, _report_calibration(calibration), stats, _empty_evaluation(), rows)
    text = path.read_text(encoding="utf-8")
    assert "Mild-cast probe" in text and "a__cct_5000" in text


def _report_calibration(calibration: Dict[str, Any]) -> Dict[str, Any]:
    """Bổ sung các trường mà write_report đọc."""
    defects = dict(calibration["defects"])
    defects["blur"] = {"params": {"defocus_bias": 0.1}, "thresholds": []}
    defects["motion_blur"] = {"params": {"min_anisotropy": 1.2, "min_extent": 0.7}}
    return {
        "version": "test",
        "dataset": {"images": 1, "sources": ["a"]},
        "defects": defects,
    }


def _empty_evaluation() -> Dict[str, Any]:
    return {"mae_matched": None, "per_defect": {}, "rows": []}
