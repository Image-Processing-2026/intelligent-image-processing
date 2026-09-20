"""Generate independent mock face-detection and mask baselines.

The expected masks are produced from the documented pixel geometry and SciPy,
without importing the production face detector or ``create_bbox_mask``.
These fixtures test the adapter contract while remaining independent of
MediaPipe and a downloaded checkpoint.
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

FEATHER_RADII = (0, 1, 3, 20, 21)


def _case_definitions() -> list[dict[str, Any]]:
    return [
        {
            "id": "CENTER",
            "shape": [10, 12],
            "detections": [{"x": 4, "y": 3, "width": 4, "height": 4, "score": 0.9}],
            "expand_ratio": 0.0,
        },
        {
            "id": "EXPAND",
            "shape": [10, 12],
            "detections": [{"x": 4, "y": 3, "width": 4, "height": 4, "score": 0.9}],
            "expand_ratio": 0.25,
        },
        {
            "id": "DEFAULT_EXPAND",
            "shape": [10, 12],
            "detections": [{"x": 4, "y": 3, "width": 4, "height": 4, "score": 0.9}],
            "expand_ratio": 0.15,
        },
        {
            "id": "EDGE",
            "shape": [10, 12],
            "detections": [{"x": 0, "y": 0, "width": 4, "height": 4, "score": 0.9}],
            "expand_ratio": 0.25,
        },
        {
            "id": "OUTSIDE",
            "shape": [10, 12],
            "detections": [{"x": 20, "y": 20, "width": 2, "height": 2, "score": 0.9}],
            "expand_ratio": 0.0,
        },
        {
            "id": "EMPTY",
            "shape": [10, 12],
            "detections": [],
            "expand_ratio": 0.15,
        },
        {
            "id": "MULTI",
            "shape": [10, 12],
            "detections": [
                {"x": 4, "y": 5, "width": 2, "height": 2, "score": 0.99},
                {"x": 0, "y": 2, "width": 2, "height": 2, "score": 0.80},
            ],
            "expand_ratio": 0.0,
        },
    ]


def _hard_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    height, width = shape
    xmin, ymin, xmax, ymax = bbox
    x0, x1 = max(0, xmin), min(width, xmax)
    y0, y1 = max(0, ymin), min(height, ymax)
    result = np.zeros(shape, dtype=np.float64)
    if x0 < x1 and y0 < y1:
        result[y0:y1, x0:x1] = 1.0
    return result


def _bbox(detection: dict[str, Any], ratio: float) -> tuple[int, int, int, int]:
    x, y = float(detection["x"]), float(detection["y"])
    width, height = float(detection["width"]), float(detection["height"])
    return (
        int(np.floor(x - ratio * width)),
        int(np.floor(y - ratio * height)),
        int(np.ceil(x + width + ratio * width)),
        int(np.ceil(y + height + ratio * height)),
    )


def _soft_mask(hard: np.ndarray, feather_radius: int) -> np.ndarray:
    if feather_radius in (0, 1):
        return hard.copy()
    kernel = feather_radius if feather_radius % 2 else feather_radius + 1
    return gaussian_filter(
        hard,
        sigma=kernel / 3.0,
        radius=kernel // 2,
        mode="mirror",
        output=np.float64,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    hard_dir = output / "hard"
    soft_dir = output / "soft"
    hard_dir.mkdir(exist_ok=True)
    soft_dir.mkdir(exist_ok=True)
    manifest: dict[str, Any] = {
        "source": "independent pixel bbox oracle and scipy.ndimage.gaussian_filter",
        "numpy_version": np.__version__,
        "scipy_version": scipy_version,
        "platform": platform.platform(),
        "feather_radii": list(FEATHER_RADII),
        "reference": {
            "bbox": "floor(x-r*w), floor(y-r*h), ceil(x+w+r*w), ceil(y+h+r*h)",
            "clip": "half-open intersection with [0,W) x [0,H)",
            "feather_identity_radii": [0, 1],
            "sigma": "odd_kernel / 3.0",
            "mode": "mirror",
            "output": "float64",
        },
        "cases": {},
    }

    for case in _case_definitions():
        case_id = case["id"]
        shape = tuple(case["shape"])
        ordered = sorted(
            case["detections"],
            key=lambda item: (
                item["y"],
                item["x"],
                item["y"] + item["height"],
                item["x"] + item["width"],
                -item["score"],
            ),
        )
        bboxes = [_bbox(item, case["expand_ratio"]) for item in ordered]
        height, width = shape
        valid_bboxes = [
            bbox
            for bbox in bboxes
            if bbox[2] > 0 and bbox[3] > 0 and bbox[0] < width and bbox[1] < height
        ]
        hard_masks = [_hard_mask(shape, bbox) for bbox in valid_bboxes]
        case_manifest: dict[str, Any] = {
            "shape": list(shape),
            "detections": ordered,
            "bboxes": [list(bbox) for bbox in bboxes],
            "expand_ratio": case["expand_ratio"],
            "mask_count": len(hard_masks),
            "hard_masks": [],
            "soft_masks": {},
        }
        merged = np.maximum.reduce(hard_masks) if hard_masks else np.zeros(shape, dtype=np.float64)
        merged_path = hard_dir / f"{case_id}_merged.npy"
        np.save(merged_path, merged)
        case_manifest["hard_merged"] = merged_path.relative_to(output).as_posix()
        case_manifest["hard_merged_sha256"] = _sha256(merged_path)
        for index, hard in enumerate(hard_masks):
            path = hard_dir / f"{case_id}_{index}.npy"
            np.save(path, hard)
            case_manifest["hard_masks"].append(
                {"path": path.relative_to(output).as_posix(), "sha256": _sha256(path)}
            )
        for radius in FEATHER_RADII:
            soft_paths: list[dict[str, str]] = []
            for index, hard in enumerate(hard_masks):
                path = soft_dir / f"{case_id}_{index}_f{radius}.npy"
                np.save(path, _soft_mask(hard, radius))
                soft_paths.append(
                    {"path": path.relative_to(output).as_posix(), "sha256": _sha256(path)}
                )
            merged_soft = (
                np.maximum.reduce([_soft_mask(hard, radius) for hard in hard_masks])
                if hard_masks
                else np.zeros(shape, dtype=np.float64)
            )
            merged_path = soft_dir / f"{case_id}_merged_f{radius}.npy"
            np.save(merged_path, merged_soft)
            case_manifest["soft_masks"][str(radius)] = {
                "masks": soft_paths,
                "merged": merged_path.relative_to(output).as_posix(),
                "merged_sha256": _sha256(merged_path),
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
        default=Path("tests/fixtures/face_detection"),
    )
    args = parser.parse_args()
    manifest = generate(args.output)
    print(f"generated {len(manifest['cases'])} face-mask cases in {args.output}")


if __name__ == "__main__":
    main()
