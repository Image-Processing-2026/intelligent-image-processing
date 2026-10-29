"""Generate independent literal union and SciPy feather baselines."""

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

FEATHER_RADII = (0, 1, 3, 15)


def _cases() -> list[dict[str, Any]]:
    return [
        {
            "id": "MULTI_UNION",
            "shape": [4, 6],
            "boxes": [[1, 0, 3, 2], [2, 1, 5, 3]],
            "source": "literal unit contract, two overlapping instances",
        },
        {
            "id": "EMPTY",
            "shape": [4, 6],
            "boxes": [],
            "source": "successful no-object inference",
        },
        {
            "id": "EDGE",
            "shape": [4, 6],
            "boxes": [[-2, -1, 2, 2]],
            "source": "bbox clipped to image boundary",
        },
    ]


def _box_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    height, width = shape
    xmin, ymin, xmax, ymax = box
    result = np.zeros(shape, dtype=np.float64)
    x0, x1 = max(0, xmin), min(width, xmax)
    y0, y1 = max(0, ymin), min(height, ymax)
    if x0 < x1 and y0 < y1:
        result[y0:y1, x0:x1] = 1.0
    return result


def _soft(union: np.ndarray, radius: int) -> np.ndarray:
    if radius in (0, 1):
        return union.copy()
    kernel = radius if radius % 2 else radius + 1
    return gaussian_filter(
        union,
        sigma=kernel / 3.0,
        radius=kernel // 2,
        mode="mirror",
        output=np.float64,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    hard_dir, soft_dir = output / "hard", output / "soft"
    hard_dir.mkdir(exist_ok=True)
    soft_dir.mkdir(exist_ok=True)
    manifest: dict[str, Any] = {
        "source": "independent literal bbox union and scipy.ndimage.gaussian_filter",
        "numpy_version": np.__version__,
        "scipy_version": scipy_version,
        "platform": platform.platform(),
        "feather_radii": list(FEATHER_RADII),
        "cases": {},
    }
    for case in _cases():
        case_id = case["id"]
        shape = tuple(case["shape"])
        masks = [_box_mask(shape, tuple(box)) for box in case["boxes"]]
        union = np.maximum.reduce(masks) if masks else np.zeros(shape, dtype=np.float64)
        hard_path = hard_dir / f"{case_id}.npy"
        np.save(hard_path, union)
        item: dict[str, Any] = {
            "shape": list(shape),
            "boxes": case["boxes"],
            "source": case["source"],
            "hard": hard_path.relative_to(output).as_posix(),
            "hard_sha256": _sha256(hard_path),
            "soft": {},
        }
        for radius in FEATHER_RADII:
            soft_path = soft_dir / f"{case_id}_f{radius}.npy"
            np.save(soft_path, _soft(union, radius))
            item["soft"][str(radius)] = {
                "path": soft_path.relative_to(output).as_posix(),
                "sha256": _sha256(soft_path),
            }
        manifest["cases"][case_id] = item
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output",
        type=Path,
        default=Path("tests/fixtures/prompt_segmentation"),
    )
    args = parser.parse_args()
    result = generate(args.output)
    print(f"generated {len(result['cases'])} prompt-mask cases in {args.output}")


if __name__ == "__main__":
    main()
