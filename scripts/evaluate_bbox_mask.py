"""Evaluate bbox masks against checked-in independent fixtures and save plots."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from src.region_engine.mask_utils import blend_regions  # noqa: E402
from src.region_engine.spatial import create_bbox_mask  # noqa: E402

MAXAE_THRESHOLD = 1e-6
MAE_THRESHOLD = 1e-7


def _metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    absolute = np.abs(difference)
    return {
        "maxae": float(np.max(absolute)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(difference**2))),
    }


def _hard_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    actual_pixels = actual == 1.0
    expected_pixels = expected == 1.0
    intersection = np.count_nonzero(actual_pixels & expected_pixels)
    union = np.count_nonzero(actual_pixels | expected_pixels)
    return {
        "pixel_mismatch": int(np.count_nonzero(actual != expected)),
        "actual_area": float(np.sum(actual)),
        "expected_area": float(np.sum(expected)),
        "iou": 1.0 if union == 0 else float(intersection / union),
    }


def _save_image(array: np.ndarray, path: Path) -> None:
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB").save(path)


def _mask_plot(
    cases: list[tuple[str, np.ndarray, str, float, float | None]],
    path: Path,
    *,
    columns: int,
) -> None:
    rows = int(np.ceil(len(cases) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.8 * rows), squeeze=False)
    for axis, (title, image, cmap, vmin, vmax) in zip(axes.flat, cases):
        axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")
    for axis in axes.flat[len(cases) :]:
        axis.axis("off")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_web(fixtures: Path, plots: Path, manifest: dict[str, Any]) -> None:
    images: list[tuple[str, np.ndarray, str, float, float | None]] = []
    for case_id in ("web_extent", "web_end"):
        case = manifest["cases"][case_id]
        hard = np.load(fixtures / case["hard"])
        images.append((f"{case_id}: hard baseline", hard, "gray", 0.0, 1.0))
    _mask_plot(images, plots / "01_web_baselines.png", columns=2)


def _plot_clipping(fixtures: Path, plots: Path, manifest: dict[str, Any]) -> None:
    images: list[tuple[str, np.ndarray, str, float, float | None]] = []
    for case_id in ("clip_tl", "clip_br", "empty_x", "full"):
        case = manifest["cases"][case_id]
        hard = np.load(fixtures / case["hard"])
        images.append((f"{case_id}: bbox={case['bbox']}", hard, "gray", 0.0, 1.0))
    _mask_plot(images, plots / "02_clipping_and_empty.png", columns=4)


def _plot_feather(fixtures: Path, plots: Path, manifest: dict[str, Any]) -> None:
    case = manifest["cases"]["rect"]
    images: list[tuple[str, np.ndarray, str, float, float | None]] = []
    for radius in (0, 3, 15, 25):
        reference = case["soft"][str(radius)]
        mask = np.load(fixtures / reference["expected"])
        images.append((f"r={radius}, k={reference['kernel_size']}, σ={reference['sigma']:g}", mask, "gray", 0.0, 1.0))
    _mask_plot(images, plots / "03_feather_comparison.png", columns=4)


def _demo_image() -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    height, width = 256, 384
    yy, xx = np.indices((height, width))
    original = np.stack(
        [
            np.floor(255 * xx / (width - 1)),
            np.floor(255 * yy / (height - 1)),
            80 + 80 * ((xx // 32 + yy // 32) % 2),
        ],
        axis=-1,
    ).astype(np.uint8)
    processed = np.clip(original.astype(np.int16) + 50, 0, 255).astype(np.uint8)
    return original, processed, (96, 64, 288, 192)


def _plot_before_after(artifacts: Path, plots: Path) -> None:
    original, processed, bbox = _demo_image()
    height, width = original.shape[:2]
    hard = create_bbox_mask((height, width), bbox, feather_radius=0)
    soft = create_bbox_mask((height, width), bbox, feather_radius=15)
    blended_hard = blend_regions(original, processed, hard)
    blended_soft = blend_regions(original, processed, soft)

    images_dir = artifacts / "images"
    images_dir.mkdir(exist_ok=True)
    _save_image(original, images_dir / "original.png")
    _save_image(processed, images_dir / "processed.png")
    _save_image(blended_hard, images_dir / "blended_hard.png")
    _save_image(blended_soft, images_dir / "blended_soft.png")

    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    entries = [
        (original, "Original", None),
        (hard, "Hard mask", "gray"),
        (soft, "Soft mask r=15", "gray"),
        (processed, "Processed", None),
        (blended_hard, "Blended hard", None),
        (blended_soft, "Blended soft", None),
    ]
    for axis, (image, title, cmap) in zip(axes.flat, entries):
        axis.imshow(image, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
        axis.set_title(title)
        axis.axis("off")
        if title in {"Original", "Processed"}:
            rectangle = plt.Rectangle(
                (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                fill=False, edgecolor="red", linewidth=1.5,
            )
            axis.add_patch(rectangle)
    figure.savefig(plots / "04_before_after.png", dpi=180)
    plt.close(figure)


def _plot_error(fixtures: Path, plots: Path, manifest: dict[str, Any]) -> None:
    case = manifest["cases"]["random_000"]
    radius = 15
    expected = np.load(fixtures / case["soft"][str(radius)]["expected"])
    actual = create_bbox_mask(tuple(case["shape"]), tuple(case["bbox"]), radius)
    error = np.abs(actual.astype(np.float64) - expected)
    figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    rendered = axis.imshow(error, cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
    axis.set_title(f"random_000 absolute error, bbox={case['bbox']}, r={radius}")
    figure.colorbar(rendered, ax=axis, label="|actual - reference|")
    figure.savefig(plots / "05_error_heatmap.png", dpi=180)
    plt.close(figure)


def _plot_edge_profile(fixtures: Path, plots: Path, manifest: dict[str, Any]) -> None:
    case = manifest["cases"]["rect"]
    radius = 15
    expected = np.load(fixtures / case["soft"][str(radius)]["expected"])
    actual = create_bbox_mask(tuple(case["shape"]), tuple(case["bbox"]), radius)
    row = actual.shape[0] // 2
    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    axis.plot(actual[row], label="Actual")
    axis.plot(expected[row], "--", label="SciPy reference")
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("x")
    axis.set_ylabel("mask")
    axis.set_title(f"Center-row profile, bbox={case['bbox']}, r={radius}")
    axis.legend()
    figure.savefig(plots / "06_edge_profile.png", dpi=180)
    plt.close(figure)


def evaluate(fixtures: Path, output: Path) -> dict[str, Any]:
    manifest = json.loads((fixtures / "manifest.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    arrays = output / "arrays"
    plots = output / "plots"
    arrays.mkdir(exist_ok=True)
    plots.mkdir(exist_ok=True)

    rows: list[dict[str, Any]] = []
    hard_rows: list[dict[str, Any]] = []
    for case_id, case in manifest["cases"].items():
        shape = tuple(case["shape"])
        bbox = tuple(case["bbox"])
        hard_expected = np.load(fixtures / case["hard"])
        hard_actual = create_bbox_mask(shape, bbox, feather_radius=0)
        hard_metrics = _hard_metrics(hard_actual, hard_expected.astype(np.float32))
        hard_rows.append({"case_id": case_id, **hard_metrics})

        for radius_text, reference in case["soft"].items():
            radius = int(radius_text)
            expected = np.load(fixtures / reference["expected"])
            actual = create_bbox_mask(shape, bbox, feather_radius=radius)
            metrics = _metrics(actual, expected)
            rows.append(
                {
                    "case_id": case_id,
                    "shape": "x".join(map(str, shape)),
                    "bbox": str(bbox),
                    "feather_radius": radius,
                    **metrics,
                    "passed": metrics["maxae"] <= MAXAE_THRESHOLD and metrics["mae"] <= MAE_THRESHOLD,
                }
            )
            if case_id in {"web_extent", "web_end", "rect", "random_000"} and radius in (0, 15):
                np.save(arrays / f"{case_id}_r{radius}_actual.npy", actual)
                np.save(arrays / f"{case_id}_r{radius}_expected.npy", expected)

    _plot_web(fixtures, plots, manifest)
    _plot_clipping(fixtures, plots, manifest)
    _plot_feather(fixtures, plots, manifest)
    _plot_before_after(output, plots)
    _plot_error(fixtures, plots, manifest)
    _plot_edge_profile(fixtures, plots, manifest)

    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "hard_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(hard_rows[0]))
        writer.writeheader()
        writer.writerows(hard_rows)

    worst = max(rows, key=lambda row: row["maxae"])
    summary: dict[str, Any] = {
        "case_count": len(manifest["cases"]),
        "soft_comparisons": len(rows),
        "soft_pass": sum(row["passed"] for row in rows),
        "soft_fail": sum(not row["passed"] for row in rows),
        "hard_comparisons": len(hard_rows),
        "hard_pixel_mismatch": sum(row["pixel_mismatch"] for row in hard_rows),
        "hard_min_iou": min(row["iou"] for row in hard_rows),
        "worst_soft_case": worst,
        "threshold": {"maxae": MAXAE_THRESHOLD, "mae": MAE_THRESHOLD},
    }
    summary["passed"] = summary["soft_fail"] == 0 and summary["hard_pixel_mismatch"] == 0
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "fixture_manifest": str((fixtures / "manifest.json").as_posix()),
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "matplotlib": matplotlib.__version__,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# Bbox-mask evaluation\n\n"
        f"Compared {len(rows)} soft-mask outputs from {len(manifest['cases'])} cases against independent SciPy float64 references.\n\n"
        f"Soft pass/fail: {summary['soft_pass']}/{summary['soft_fail']}; hard pixel mismatches: {summary['hard_pixel_mismatch']}; minimum hard IoU: {summary['hard_min_iou']:.6g}.\n\n"
        f"Worst soft case: `{worst['case_id']}` at r={worst['feather_radius']}, MaxAE={worst['maxae']:.6g}, MAE={worst['mae']:.6g}, RMSE={worst['rmse']:.6g}.\n\n"
        "Recreate with `python scripts/generate_bbox_mask_baselines.py` followed by `python scripts/evaluate_bbox_mask.py --output-dir artifacts/module-2/bbox-mask`.\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=Path("tests/fixtures/bbox_mask"))
    parser.add_argument(
        "--output-dir",
        "--output",
        dest="output",
        type=Path,
        default=Path("artifacts/module-2/bbox-mask"),
    )
    args = parser.parse_args()
    summary = evaluate(args.fixtures, args.output)
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
