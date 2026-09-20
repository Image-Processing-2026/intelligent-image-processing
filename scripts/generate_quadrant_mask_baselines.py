"""Generate independent hard and SciPy soft-mask baselines.

This generator deliberately does not import any production region-engine
function. Hard masks use coordinate comparisons or literal contract matrices;
soft references use SciPy float64 Gaussian filtering.
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

QUADRANTS = ("top", "bottom", "left", "right", "center")
FEATHER_RADII = (0, 1, 2, 3, 24, 25)


def _coordinate_oracle(shape: tuple[int, int], quadrant: str) -> np.ndarray:
    height, width = shape
    yy, xx = np.indices(shape)
    if quadrant == "top":
        selected = yy < height // 2
    elif quadrant == "bottom":
        selected = yy >= height // 2
    elif quadrant == "left":
        selected = xx < width // 2
    elif quadrant == "right":
        selected = xx >= width // 2
    else:
        selected = (
            (yy >= height // 4)
            & (yy < (3 * height) // 4)
            & (xx >= width // 4)
            & (xx < (3 * width) // 4)
        )
    return selected.astype(np.float64)


def _literal_contract(shape: tuple[int, int], quadrant: str) -> np.ndarray:
    """Return small literal matrices from the documented geometry tables."""
    if shape == (4, 6):
        matrices = {
            "top": [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
            "bottom": [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]],
            "left": [[1, 1, 1, 0, 0, 0]] * 4,
            "right": [[0, 0, 0, 1, 1, 1]] * 4,
            "center": [[0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0]],
        }
    elif shape == (5, 7):
        matrices = {
            "top": [[1] * 7, [1] * 7, [0] * 7, [0] * 7, [0] * 7],
            "bottom": [[0] * 7, [0] * 7, [1] * 7, [1] * 7, [1] * 7],
            "left": [[1, 1, 1, 0, 0, 0, 0]] * 5,
            "right": [[0, 0, 0, 1, 1, 1, 1]] * 5,
            "center": [[0] * 7, [0, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0, 0], [0] * 7, [0] * 7],
        }
    elif shape == (1, 1):
        matrices = {"top": [[0]], "bottom": [[1]], "left": [[0]], "right": [[1]], "center": [[0]]}
    elif shape == (1, 5):
        matrices = {
            "top": [[0, 0, 0, 0, 0]],
            "bottom": [[1, 1, 1, 1, 1]],
            "left": [[1, 1, 0, 0, 0]],
            "right": [[0, 0, 1, 1, 1]],
            "center": [[0, 0, 0, 0, 0]],
        }
    elif shape == (5, 1):
        matrices = {
            "top": [[1], [1], [0], [0], [0]],
            "bottom": [[0], [0], [1], [1], [1]],
            "left": [[0], [0], [0], [0], [0]],
            "right": [[1], [1], [1], [1], [1]],
            "center": [[0], [0], [0], [0], [0]],
        }
    else:
        raise ValueError(f"no literal contract matrix for {shape}")
    return np.asarray(matrices[quadrant], dtype=np.float64)


def _hard_reference(shape: tuple[int, int], quadrant: str, source_kind: str) -> np.ndarray:
    if source_kind == "literal_contract":
        return _literal_contract(shape, quadrant)
    if source_kind == "web_right":
        result = np.zeros(shape, dtype=np.float64)
        result[0, 5:] = 1.0
        return result
    if source_kind == "web_center":
        result = np.zeros(shape, dtype=np.float64)
        result[1:4, 1:4] = 1.0
        return result
    return _coordinate_oracle(shape, quadrant)


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


def _case_definitions() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for label, shape in (
        ("even_4x6", (4, 6)),
        ("odd_5x7", (5, 7)),
        ("tiny_1x1", (1, 1)),
        ("tiny_1x5", (1, 5)),
        ("tiny_5x1", (5, 1)),
    ):
        for quadrant in QUADRANTS:
            cases.append(
                {
                    "id": f"{label}_{quadrant}",
                    "shape": shape,
                    "quadrant": quadrant,
                    "source": "literal geometry table from project contract",
                    "source_kind": "literal_contract",
                }
            )

    for label, shape in (
        ("shape_2x2", (2, 2)),
        ("shape_3x3", (3, 3)),
        ("shape_6x10", (6, 10)),
        ("shape_7x9", (7, 9)),
        ("shape_8x12", (8, 12)),
    ):
        for quadrant in QUADRANTS:
            cases.append(
                {
                    "id": f"{label}_{quadrant}",
                    "shape": shape,
                    "quadrant": quadrant,
                    "source": "independent np.indices coordinate oracle",
                    "source_kind": "coordinate_oracle",
                }
            )

    cases.extend(
        [
            {
                "id": "web_right",
                "shape": (1, 10),
                "quadrant": "right",
                "source": "NumPy slicing example x[5:] converted to mask",
                "source_kind": "web_right",
            },
            {
                "id": "web_center",
                "shape": (6, 6),
                "quadrant": "center",
                "source": "scikit-image rectangle example padded to 6x6",
                "source_kind": "web_center",
            },
        ]
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
        "source": "independent literal/coordinate hard oracle and scipy.ndimage.gaussian_filter",
        "numpy_version": np.__version__,
        "scipy_version": scipy_version,
        "platform": platform.platform(),
        "feather_radii": list(FEATHER_RADII),
        "quadrants": list(QUADRANTS),
        "reference": {
            "hard_coordinate": "np.indices; direct axis inequalities from project contract",
            "feather_identity_radii": [0, 1],
            "kernel_size": "r if r is odd else r + 1",
            "sigma": "kernel_size / 3.0",
            "radius": "kernel_size // 2",
            "mode": "mirror",
            "output": "float64",
        },
        "cases": {},
    }

    for case in _case_definitions():
        case_id = case["id"]
        shape = tuple(case["shape"])
        quadrant = case["quadrant"]
        hard = _hard_reference(shape, quadrant, case["source_kind"])
        hard_path = hard_dir / f"{case_id}.npy"
        np.save(hard_path, hard)
        case_manifest: dict[str, Any] = {
            "shape": list(shape),
            "quadrant": quadrant,
            "source": case["source"],
            "source_kind": case["source_kind"],
            "hard": hard_path.relative_to(output).as_posix(),
            "hard_sha256": _sha256(hard_path),
            "soft": {},
        }
        for feather_radius in FEATHER_RADII:
            expected = _soft_reference(hard, feather_radius)
            expected_path = soft_dir / f"{case_id}_r{feather_radius}.npy"
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
                "border": "mirror / BORDER_REFLECT_101",
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
        default=Path("tests/fixtures/quadrant_mask"),
        help="directory for hard masks, SciPy references, and manifest",
    )
    args = parser.parse_args()
    manifest = generate(args.output)
    print(f"Generated {len(manifest['cases'])} quadrant cases in {args.output}")


if __name__ == "__main__":
    main()
