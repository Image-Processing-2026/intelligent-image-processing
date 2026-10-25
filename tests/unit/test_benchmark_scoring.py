"""Unit test offline cho phần chấm điểm benchmark (scripts/benchmark_scoring.py)."""

import pytest

from scripts.benchmark_scoring import (
    aggregate,
    angle_error_deg,
    estimation_summary,
    match_defects,
    region_matches,
    relative_error,
    score_image,
    type_matches,
)


def _exp(type_: str, region: str = "full", min_sev: int = 1, sev: int | None = None) -> dict:
    return {
        "type": type_,
        "region": region,
        "min_severity": min_sev,
        "severity": min_sev if sev is None else sev,
    }


def _pred(type_: str, region: str = "full", sev: int = 2, origin: str = "vlm") -> dict:
    return {"type": type_, "region": region, "severity": sev, "origin": origin}


def test_region_full_matches_everything_and_synonyms_match():
    assert region_matches("full", "face")
    assert region_matches("ground", "field")
    assert region_matches("sky", "clouds")
    assert not region_matches("sky", "face")


def test_dark_types_are_interchangeable_only_on_subjects():
    assert type_matches("underexposed", "face", "backlit_subject")
    assert type_matches("backlit_subject", "person", "underexposed")
    assert not type_matches("underexposed", "sky", "backlit_subject")
    assert not type_matches("noise", "full", "blur")


def test_match_is_one_to_one_so_duplicates_do_not_count_twice():
    expected = [_exp("noise")]
    predicted = [_pred("noise", sev=2), _pred("noise", sev=3)]
    pairs = match_defects(expected, predicted)
    assert len(pairs) == 1
    # Ghép với mức severity gần nhất với đáp án (đáp án severity = min_sev = 1)
    assert pairs[0] == (0, 0)


def test_min_severity_threshold_is_respected():
    score = score_image([_exp("noise", min_sev=2)], [], [_pred("noise", sev=1)], [])
    assert score["matched"] == []
    assert score["missed"] == [{"type": "noise", "region": "full"}]
    assert score["false"] == [{"type": "noise", "region": "full", "severity": 1}]


def test_severity_zero_predictions_are_not_defects():
    score = score_image([], [], [_pred("noise", sev=0)], [])
    assert score["false"] == []
    assert score["is_clean"]


def test_score_image_reports_matched_missed_and_preserve():
    expected = [_exp("underexposed", "face", min_sev=1, sev=2), _exp("noise")]
    predicted = [
        _pred("backlit_subject", "face", sev=2, origin="measured"),
        _pred("color_cast_warm", "full", sev=1),
    ]
    score = score_image(expected, ["low_key"], predicted, ["low_key", "warm_tone"])
    assert [m["type"] for m in score["matched"]] == ["underexposed"]
    assert score["matched"][0]["origin"] == "measured"
    assert score["matched"][0]["expected_severity"] == 2
    assert score["matched"][0]["predicted_severity"] == 2
    assert score["missed"] == [{"type": "noise", "region": "full"}]
    assert score["false"] == [{"type": "color_cast_warm", "region": "full", "severity": 1}]
    assert score["preserve_matched"] == ["low_key"]
    assert score["preserve_missed"] == []


def test_aggregate_precision_recall_per_type_and_fp_rate_on_clean_images():
    scores = [
        score_image([_exp("noise")], [], [_pred("noise")], []),  # TP noise
        score_image([_exp("noise")], [], [], []),  # FN noise
        score_image([], [], [_pred("blur")], []),  # clean, FP blur
        score_image([], [], [], []),  # clean, no FP
    ]
    summary = aggregate(scores)
    assert summary["by_type"]["noise"] == {
        "tp": 1,
        "fp": 0,
        "fn": 1,
        "precision": 1.0,
        "recall": 0.5,
    }
    assert summary["by_type"]["blur"]["precision"] == 0.0
    assert summary["precision"] == 0.5
    assert summary["recall"] == 0.5
    assert summary["clean_images"] == 2
    assert summary["fp_image_rate"] == 0.5
    assert summary["fp_per_clean_image"] == 0.5


def test_aggregate_severity_mae_region_accuracy_and_preserve_recall():
    scores = [
        score_image(
            [_exp("noise", "face", min_sev=1, sev=2)],
            ["warm_tone", "soft_focus"],
            [_pred("noise", "face", sev=3)],
            ["warm_tone"],
        ),
        score_image(
            [_exp("blur", "sky", min_sev=1, sev=1)],
            [],
            [_pred("blur", "ground", sev=1)],
            [],
        ),
    ]
    summary = aggregate(scores)
    # |3-2| = 1 và |1-1| = 0 → MAE 0.5
    assert summary["severity_mae"] == 0.5
    # face khớp face; sky không khớp ground → 1/2
    assert summary["region_accuracy"] == 0.5
    assert summary["preserve_expected"] == 2
    assert summary["preserve_recall"] == 0.5


def test_aggregate_handles_empty_input():
    summary = aggregate([])
    assert summary["precision"] is None
    assert summary["recall"] is None
    assert summary["fp_image_rate"] is None
    assert summary["preserve_recall"] is None


def test_relative_error_and_zero_truth():
    assert relative_error(11.0, 10.0) == pytest.approx(0.1)
    assert relative_error(1.0, 0.0) is None


def test_angle_error_wraps_around_circle():
    assert angle_error_deg(359.0, 1.0) == pytest.approx(2.0)
    assert angle_error_deg(10.0, 350.0) == pytest.approx(20.0)
    assert angle_error_deg(0.0, 180.0) == pytest.approx(180.0)


def test_estimation_summary_relative_and_circular():
    rel = estimation_summary([(11.0, 10.0), (20.0, 10.0)], tolerance=0.15)
    assert rel["n"] == 2
    assert rel["mae"] == pytest.approx(0.55)
    assert rel["within_tolerance"] == 0.5

    ang = estimation_summary([(359.0, 1.0), (30.0, 0.0)], tolerance=3.0, circular=True)
    assert ang["n"] == 2
    assert ang["within_tolerance"] == 0.5


def test_estimation_summary_empty():
    assert estimation_summary([], tolerance=0.1) == {
        "n": 0,
        "mae": None,
        "median": None,
        "within_tolerance": None,
    }
