"""Regression tests against the checked-in independent SciPy fixtures."""

import hashlib
import json
from pathlib import Path

import numpy as np

from src.region_engine.mask_utils import create_soft_mask

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "soft_mask"
MANIFEST = json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixture_manifest_checksums_are_intact() -> None:
    for case in MANIFEST["cases"].values():
        assert _sha256(FIXTURE_DIR / case["input"]) == case["input_sha256"]
        assert _sha256(FIXTURE_DIR / case["expected"]) == case["expected_sha256"]


def test_all_fixture_cases_match_reference() -> None:
    for case in MANIFEST["cases"].values():
        input_mask = np.load(FIXTURE_DIR / case["input"])
        expected = np.load(FIXTURE_DIR / case["expected"])
        actual = create_soft_mask(input_mask, feather_radius=case["feather_radius"])

        assert actual.dtype == np.float32
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
