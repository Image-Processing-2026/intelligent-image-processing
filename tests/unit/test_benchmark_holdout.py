"""Hold-out benchmark: nhãn chưa duyệt không được chấm nếu không cho phép rõ ràng."""

import json
from pathlib import Path

import pytest

from scripts import benchmark_analysis
from scripts.benchmark_analysis import load_cases, unreviewed_holdout_ids


def _manifest(tmp_path: Path, statuses: list) -> Path:
    """Manifest hold-out giả với label_status cho trước."""
    images = [
        {
            "id": f"h{i}",
            "category": "already_good",
            "file": f"already_good/h{i}.jpg",
            "expected": {"defects": [], "preserve": [], "ambiguous": False},
            "label_status": status,
        }
        for i, status in enumerate(statuses)
    ]
    path = tmp_path / "holdout_manifest.json"
    path.write_text(json.dumps({"images": images}), encoding="utf-8")
    return path


def test_unreviewed_ids_lists_everything_not_reviewed(tmp_path: Path) -> None:
    path = _manifest(tmp_path, ["reviewed", "unreviewed", None])
    assert unreviewed_holdout_ids(path) == ["h1", "h2"]


def test_holdout_refuses_unreviewed_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark_analysis, "HOLDOUT_MANIFEST", _manifest(tmp_path, ["unreviewed"]))
    with pytest.raises(SystemExit):
        load_cases("holdout")


def test_holdout_loads_when_reviewed_or_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _manifest(tmp_path, ["unreviewed"])
    monkeypatch.setattr(benchmark_analysis, "HOLDOUT_MANIFEST", path)
    monkeypatch.setattr(
        benchmark_analysis,
        "load_real_cases",
        lambda manifest, image_dir, dataset: [manifest, dataset],
    )
    assert load_cases("holdout", allow_unreviewed=True) == [path, "holdout"]


def test_holdout_reviewed_only_skips_unreviewed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _manifest(tmp_path, ["reviewed", "unreviewed", "reviewed"])
    monkeypatch.setattr(benchmark_analysis, "HOLDOUT_MANIFEST", path)
    cases = load_cases("holdout", reviewed_only=True)
    assert [c.case_id for c in cases] == ["h0", "h2"]
