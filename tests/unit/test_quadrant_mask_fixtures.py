"""Regression tests against independent quadrant hard/soft fixtures."""

import hashlib
import json
from pathlib import Path

import numpy as np

from src.region_engine.spatial import create_quadrant_mask

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "quadrant_mask"
MANIFEST = json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixture_manifest_checksums_are_intact() -> None:
    for case in MANIFEST["cases"].values():
        hard_path = FIXTURE_DIR / case["hard"]
        assert _sha256(hard_path) == case["hard_sha256"]
        for reference in case["soft"].values():
            expected_path = FIXTURE_DIR / reference["expected"]
            assert _sha256(expected_path) == reference["expected_sha256"]


def test_all_hard_fixture_cases_match_reference() -> None:
    for case in MANIFEST["cases"].values():
        actual = create_quadrant_mask(
            tuple(case["shape"]), case["quadrant"], feather_radius=0
        )
        expected = np.load(FIXTURE_DIR / case["hard"])

        np.testing.assert_array_equal(actual, expected.astype(np.float32))


def test_all_soft_fixture_cases_match_scipy_reference() -> None:
    for case in MANIFEST["cases"].values():
        shape = tuple(case["shape"])
        for radius, reference in case["soft"].items():
            actual = create_quadrant_mask(shape, case["quadrant"], int(radius))
            expected = np.load(FIXTURE_DIR / reference["expected"])

            assert actual.dtype == np.float32
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)

