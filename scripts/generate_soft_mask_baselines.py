"""Generate independent SciPy baselines for Module 2 soft-mask tests.

This script deliberately does not import the production implementation. The
saved manifest records the reference configuration and SHA-256 checksums so
the fixtures can be audited or regenerated explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
from scipy import __version__ as scipy_version
from scipy.ndimage import gaussian_filter


def _normalize(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == np.bool_:
        return mask.astype(np.float64)
    values = np.unique(mask)
    if np.any(values == 255):
        return mask.astype(np.float64) / 255.0
    return mask.astype(np.float64)


def _reference(mask: np.ndarray, feather_radius: int) -> np.ndarray:
    normalized = _normalize(mask)
    if feather_radius in (0, 1):
        return normalized.copy()
    kernel_size = feather_radius if feather_radius % 2 else feather_radius + 1
    return gaussian_filter(
        normalized,
        sigma=(kernel_size / 3.0, kernel_size / 3.0),
        order=0,
        mode="mirror",
        radius=(kernel_size - 1) // 2,
        output=np.float64,
    )


def _cases() -> dict[str, tuple[np.ndarray, int]]:
    cases: dict[str, tuple[np.ndarray, int]] = {}
    cases["zero"] = (np.zeros((32, 48), dtype=np.uint8), 15)
    cases["one_01"] = (np.ones((32, 48), dtype=np.uint8), 15)
    cases["one_0255"] = (np.full((32, 48), 255, dtype=np.uint8), 15)
    cases["one_bool"] = (np.ones((32, 48), dtype=bool), 15)

    identity = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    cases["identity_f0"] = (identity, 0)
    cases["identity_f1"] = (identity, 1)

    impulse = np.zeros((5, 5), dtype=np.uint8)
    impulse[2, 2] = 255
    cases["impulse"] = (impulse, 3)

    step = np.zeros((64, 64), dtype=np.uint8)
    step[:, 32:] = 1
    cases["step"] = (step, 15)

    rect = np.zeros((64, 64), dtype=np.uint8)
    rect[16:48, 16:48] = 1
    cases["rect"] = (rect, 15)
    cases["rect_even"] = (rect, 14)

    touch_edge = np.zeros((32, 32), dtype=np.uint8)
    touch_edge[:16, :12] = 1
    cases["touch_edge"] = (touch_edge, 15)

    corner = np.zeros((5, 5), dtype=np.uint8)
    corner[0, 0] = 1
    cases["corner"] = (corner, 3)

    thin = np.zeros((65, 65), dtype=np.uint8)
    thin[:, 32] = 1
    cases["thin"] = (thin, 15)

    tiny_1x1 = np.array([[1]], dtype=np.uint8)
    tiny_1x9 = np.zeros((1, 9), dtype=np.uint8)
    tiny_1x9[0, 4] = 1
    tiny_9x1 = tiny_1x9.T.copy()
    cases["tiny_1x1"] = (tiny_1x1, 15)
    cases["tiny_1x9"] = (tiny_1x9, 15)
    cases["tiny_9x1"] = (tiny_9x1, 15)

    rng = np.random.default_rng(20260921)
    random_mask = rng.integers(0, 2, (31, 47), dtype=np.uint8)
    for feather_radius in (3, 15, 25):
        cases[f"random_f{feather_radius}"] = (random_mask, feather_radius)

    noncontiguous_source = np.zeros((64, 96), dtype=np.uint8)
    noncontiguous_source[16:48, 16:80] = 1
    cases["noncontiguous"] = (noncontiguous_source[:, ::2], 15)
    return cases


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "source": "scipy.ndimage.gaussian_filter, independently generated",
        "scipy_version": scipy_version,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "reference": {
            "mode": "mirror",
            "order": 0,
            "output": "float64",
            "sigma": "(kernel_size / 3.0, kernel_size / 3.0)",
            "radius": "(kernel_size - 1) // 2",
        },
        "cases": {},
    }

    for case_id, (mask, feather_radius) in _cases().items():
        input_path = output / f"{case_id}_input.npy"
        expected_path = output / f"{case_id}_expected.npy"
        expected = _reference(mask, feather_radius)
        np.save(input_path, mask)
        np.save(expected_path, expected)
        kernel_size = (
            feather_radius
            if feather_radius == 0 or feather_radius % 2
            else feather_radius + 1
        )
        manifest["cases"][case_id] = {
            "input": input_path.name,
            "expected": expected_path.name,
            "dtype": str(mask.dtype),
            "shape": list(mask.shape),
            "feather_radius": feather_radius,
            "kernel_size": kernel_size,
            "input_sha256": _sha256(input_path),
            "expected_sha256": _sha256(expected_path),
        }

    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/soft_mask"),
        help="directory for input, expected, and manifest files",
    )
    args = parser.parse_args()
    manifest = generate(args.output)
    print(f"Generated {len(manifest['cases'])} soft-mask baselines in {args.output}")


if __name__ == "__main__":
    main()
