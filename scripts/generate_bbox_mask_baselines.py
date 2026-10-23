"""Generate independent baselines for ``create_bbox_mask``.

The generator intentionally does not import the production region engine. Hard
masks are produced by coordinate comparisons and soft references by SciPy's
float64 ``gaussian_filter`` with the documented equivalent of OpenCV's
``BORDER_REFLECT_101``.
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

FEATHER_RADII = (0, 1, 2, 3, 14, 15, 25)


def _hard_reference(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    height, width = shape
    xmin, ymin, xmax, ymax = bbox
    yy, xx = np.indices((height, width))
    return ((xx >= xmin) & (xx < xmax) & (yy >= ymin) & (yy < ymax)).astype(np.float64)


def _soft_reference(hard: np.ndarray, feather_radius: int) -> np.ndarray:
    if feather_radius in (0, 1):
        return hard.copy()
    kernel_size = feather_radius if feather_radius % 2 else feather_radius + 1
    return gaussian_filter(
        hard,
        sigma=kernel_size / 3.0,
        radius=kernel_size // 2,
        mode="mirror",
        output=np.float64,
    )


def _fixed_cases() -> list[tuple[str, tuple[int, int], tuple[int, int, int, int], str]]:
    return [
        ("web_extent", (5, 5), (1, 1, 4, 4), "scikit-image rectangle extent converted to half-open bbox"),
        ("web_end", (5, 5), (1, 0, 4, 4), "scikit-image inclusive end converted to half-open bbox"),
        ("rect", (4, 6), (1, 1, 5, 3), "project-designed geometry"),
        ("clip_tl", (4, 6), (-2, -1, 3, 2), "project-designed clipping"),
        ("clip_br", (4, 6), (4, 2, 9, 8), "project-designed clipping"),
        ("full", (4, 6), (-5, -5, 10, 10), "project-designed full canvas"),
        ("pixel", (4, 6), (5, 3, 6, 4), "project-designed one pixel"),
        ("outside_right", (4, 6), (6, 0, 9, 3), "project-designed empty intersection"),
        ("outside_left", (4, 6), (-3, 0, -1, 3), "project-designed empty intersection"),
        ("outside_bottom", (4, 6), (0, 4, 3, 6), "project-designed empty intersection"),
        ("outside_top", (4, 6), (0, -3, 3, -1), "project-designed empty intersection"),
        ("empty_x", (4, 6), (2, 1, 2, 3), "project-designed zero-width bbox"),
        ("empty_y", (4, 6), (1, 2, 4, 2), "project-designed zero-height bbox"),
        ("empty_point", (4, 6), (1, 1, 1, 1), "project-designed point bbox"),
        ("tiny", (1, 1), (0, 0, 1, 1), "project-designed tiny canvas"),
        ("row", (1, 5), (1, 0, 4, 1), "project-designed one-row canvas"),
        ("column", (5, 1), (0, 1, 1, 4), "project-designed one-column canvas"),
    ]


def _cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for case_id, shape, bbox, source in _fixed_cases():
        cases.append({"id": case_id, "shape": shape, "bbox": bbox, "source": source})

    rng = np.random.default_rng(20260921)
    shape = (17, 23)
    for index in range(100):
        points = rng.integers(-12, 36, size=4)
        xmin, xmax = sorted((int(points[0]), int(points[2])))
        ymin, ymax = sorted((int(points[1]), int(points[3])))
        cases.append(
            {
                "id": f"random_{index:03d}",
                "shape": shape,
                "bbox": (xmin, ymin, xmax, ymax),
                "source": "project random valid bbox, default_rng seed 20260921",
            }
        )
    return cases


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    hard_dir = output / "hard"
    soft_dir = output / "soft"
    hard_dir.mkdir(exist_ok=True)
    soft_dir.mkdir(exist_ok=True)

    manifest: dict[str, Any] = {
        "source": "independent coordinate oracle and scipy.ndimage.gaussian_filter",
        "numpy_version": np.__version__,
        "scipy_version": scipy_version,
        "platform": platform.platform(),
        "seed": 20260921,
        "feather_radii": list(FEATHER_RADII),
        "reference": {
            "hard": "((xx >= xmin) & (xx < xmax) & (yy >= ymin) & (yy < ymax)).astype(float64)",
            "feather_identity_radii": [0, 1],
            "kernel_size": "r if r is odd else r + 1",
            "sigma": "kernel_size / 3.0",
            "radius": "kernel_size // 2",
            "mode": "mirror",
            "output": "float64",
        },
        "cases": {},
    }

    for case in _cases():
        case_id = case["id"]
        shape = tuple(case["shape"])
        bbox = tuple(case["bbox"])
        hard = _hard_reference(shape, bbox)
        hard_path = hard_dir / f"{case_id}.npy"
        np.save(hard_path, hard)
        case_manifest: dict[str, Any] = {
            "shape": list(shape),
            "bbox": list(bbox),
            "source": case["source"],
            "hard": hard_path.relative_to(output).as_posix(),
            "hard_sha256": _sha256(hard_path),
            "soft": {},
        }
        for feather_radius in FEATHER_RADII:
            expected = _soft_reference(hard, feather_radius)
            expected_path = soft_dir / f"{case_id}_f{feather_radius}.npy"
            np.save(expected_path, expected)
            kernel_size = (
                feather_radius
                if feather_radius in (0, 1) or feather_radius % 2
                else feather_radius + 1
            )
            case_manifest["soft"][str(feather_radius)] = {
                "expected": expected_path.relative_to(output).as_posix(),
                "expected_sha256": _sha256(expected_path),
                "kernel_size": kernel_size,
                "sigma": 0.0 if feather_radius in (0, 1) else kernel_size / 3.0,
            }
        manifest["cases"][case_id] = case_manifest

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output",
        type=Path,
        default=Path("tests/fixtures/bbox_mask"),
        help="directory for hard masks, SciPy references, and manifest",
    )
    args = parser.parse_args()
    manifest = generate(args.output)
    print(f"Generated {len(manifest['cases'])} bbox cases in {args.output}")


if __name__ == "__main__":
    main()
