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
    predict,
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
