"""Evaluate quadrant masks against independent fixtures and save diagnostics."""

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
from src.region_engine.spatial import create_quadrant_mask  # noqa: E402

MAXAE_THRESHOLD = 1e-6
MAE_THRESHOLD = 1e-7
QUADRANTS = ("top", "bottom", "left", "right", "center")


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


def _imshow_grid(
    entries: list[tuple[str, np.ndarray, str | None]],
    path: Path,
    *,
    columns: int,
    figsize_scale: tuple[float, float] = (3.2, 3.2),
) -> None:
    rows = int(np.ceil(len(entries) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(figsize_scale[0] * columns, figsize_scale[1] * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, (title, image, cmap) in zip(axes.flat, entries):
        axis.imshow(image, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None, interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")
    for axis in axes.flat[len(entries) :]:
        axis.axis("off")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_five_regions(plots: Path) -> None:
    shape = (4, 6)
    entries: list[tuple[str, np.ndarray, str | None]] = []
    for quadrant in QUADRANTS:
        entries.append((f"{quadrant} hard, r=0", create_quadrant_mask(shape, quadrant, 0), "gray"))
        entries.append((f"{quadrant} soft, r=25", create_quadrant_mask(shape, quadrant, 25), "gray"))
    _imshow_grid(entries, plots / "01_five_regions_hard_soft.png", columns=2)


def _plot_even_odd_tiny(plots: Path) -> None:
    entries: list[tuple[str, np.ndarray, str | None]] = []
    for shape in ((4, 6), (5, 7), (1, 1)):
        for quadrant in QUADRANTS:
            entries.append((f"{shape} {quadrant}", create_quadrant_mask(shape, quadrant, 0), "gray"))
    _imshow_grid(entries, plots / "02_even_odd_tiny.png", columns=5, figsize_scale=(2.2, 2.4))


def _demo_image() -> tuple[np.ndarray, np.ndarray]:
    height, width = 256, 384
    yy, xx = np.indices((height, width))
    original = np.stack(
        [
            np.floor(255 * xx / (width - 1)),
            yy,
            80 + 80 * ((xx // 32 + yy // 32) % 2),
        ],
        axis=-1,
    ).astype(np.uint8)
    processed = np.clip(original.astype(np.int16) + 50, 0, 255).astype(np.uint8)
    return original, processed


def _plot_before_after(output: Path, plots: Path) -> None:
    original, processed = _demo_image()
    image_dir = output / "images"
    image_dir.mkdir(exist_ok=True)
    _save_image(original, image_dir / "original.png")
    _save_image(processed, image_dir / "processed.png")

    figure, axes = plt.subplots(5, 4, figsize=(14, 17), constrained_layout=True)
    for row, quadrant in enumerate(QUADRANTS):
        hard = create_quadrant_mask(original.shape[:2], quadrant, 0)
        soft = create_quadrant_mask(original.shape[:2], quadrant, 25)
        hard_blend = blend_regions(original, processed, hard)
        soft_blend = blend_regions(original, processed, soft)
        _save_image(hard_blend, image_dir / f"blended_{quadrant}_hard.png")
        _save_image(soft_blend, image_dir / f"blended_{quadrant}_soft.png")
        entries = (original, processed, hard_blend, soft_blend)
        titles = ("Original", "Processed", f"{quadrant} hard", f"{quadrant} soft")
        for axis, image, title in zip(axes[row], entries, titles):
            axis.imshow(image, cmap="gray" if image.ndim == 2 else None, vmin=0 if image.ndim == 2 else None, vmax=1 if image.ndim == 2 else None)
            axis.set_title(title)
            axis.axis("off")
    figure.savefig(plots / "03_before_after_all_regions.png", dpi=180)
    plt.close(figure)


def _plot_feather(plots: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(13, 6.5), constrained_layout=True)
    for row, quadrant in enumerate(("top", "center")):
        for axis, radius in zip(axes[row], (0, 3, 15, 25)):
            mask = create_quadrant_mask((64, 96), quadrant, radius)
            kernel = radius if radius in (0, 1) or radius % 2 else radius + 1
            sigma = 0.0 if radius in (0, 1) else kernel / 3.0
            axis.imshow(mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axis.set_title(f"{quadrant}, r={radius}, k={kernel}, σ={sigma:g}")
            axis.axis("off")
    figure.savefig(plots / "04_feather_comparison.png", dpi=180)
    plt.close(figure)


def _plot_errors(fixtures: Path, plots: Path, manifest: dict[str, Any]) -> None:
    case_ids = ("even_4x6_top", "web_center")
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for axis, case_id in zip(axes, case_ids):
        case = manifest["cases"][case_id]
        expected = np.load(fixtures / case["soft"]["25"]["expected"])
        actual = create_quadrant_mask(tuple(case["shape"]), case["quadrant"], 25)
        error = np.abs(actual.astype(np.float64) - expected)
        rendered = axis.imshow(error, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(f"{case_id}, MaxAE={np.max(error):.2e}")
        axis.axis("off")
        figure.colorbar(rendered, ax=axis, fraction=0.046, pad=0.04)
    figure.savefig(plots / "05_error_heatmaps.png", dpi=180)
    plt.close(figure)


def _plot_profiles(fixtures: Path, plots: Path, manifest: dict[str, Any]) -> None:
    cases = (("even_4x6_top", "column"), ("even_4x6_left", "row"), ("odd_5x7_center", "row"))
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, (case_id, direction) in zip(axes, cases):
        case = manifest["cases"][case_id]
        expected = np.load(fixtures / case["soft"]["25"]["expected"])
        actual = create_quadrant_mask(tuple(case["shape"]), case["quadrant"], 25)
        if direction == "column":
            index = actual.shape[1] // 2
            actual_profile, expected_profile = actual[:, index], expected[:, index]
            label = f"column x={index}"
        else:
            index = actual.shape[0] // 2
            actual_profile, expected_profile = actual[index, :], expected[index, :]
            label = f"row y={index}"
        axis.plot(actual_profile, label="Actual")
        axis.plot(expected_profile, "--", label="SciPy reference")
        axis.set_ylim(-0.02, 1.02)
        axis.set_title(f"{case_id}\n{label}")
        axis.set_xlabel("pixel")
        axis.set_ylabel("mask")
        axis.legend(fontsize=8)
    figure.savefig(plots / "06_edge_profiles.png", dpi=180)
    plt.close(figure)


def evaluate(fixtures: Path, output: Path) -> dict[str, Any]:
    manifest = json.loads((fixtures / "manifest.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    arrays = output / "arrays"
    plots = output / "plots"
    logs = output / "logs"
    for directory in (arrays, plots, logs):
        directory.mkdir(exist_ok=True)

    rows: list[dict[str, Any]] = []
    hard_rows: list[dict[str, Any]] = []
    saved_case_ids = {"even_4x6_top", "even_4x6_center", "odd_5x7_center", "web_right", "web_center"}
    for case_id, case in manifest["cases"].items():
        shape = tuple(case["shape"])
        quadrant = case["quadrant"]
        hard_expected = np.load(fixtures / case["hard"])
        hard_actual = create_quadrant_mask(shape, quadrant, 0)
        hard_metrics = _hard_metrics(hard_actual, hard_expected.astype(np.float32))
        hard_rows.append({"case_id": case_id, "quadrant": quadrant, **hard_metrics})
        if case_id in saved_case_ids:
            np.save(arrays / f"{case_id}_hard_actual.npy", hard_actual)
            np.save(arrays / f"{case_id}_hard_expected.npy", hard_expected)

        for radius_text, reference in case["soft"].items():
            radius = int(radius_text)
            expected = np.load(fixtures / reference["expected"])
            actual = create_quadrant_mask(shape, quadrant, radius)
            metrics = _metrics(actual, expected)
            passed = metrics["maxae"] <= MAXAE_THRESHOLD and metrics["mae"] <= MAE_THRESHOLD
            rows.append(
                {
                    "case_id": case_id,
                    "shape": "x".join(map(str, shape)),
                    "quadrant": quadrant,
                    "feather_radius": radius,
                    **metrics,
                    "passed": passed,
                }
            )
            if case_id in saved_case_ids and radius in (0, 25):
                np.save(arrays / f"{case_id}_r{radius}_actual.npy", actual)
                np.save(arrays / f"{case_id}_r{radius}_expected.npy", expected)

    _plot_five_regions(plots)
    _plot_even_odd_tiny(plots)
    _plot_before_after(output, plots)
    _plot_feather(plots)
    _plot_errors(fixtures, plots, manifest)
    _plot_profiles(fixtures, plots, manifest)

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
    report = (
        "# Quadrant-mask evaluation\n\n"
        f"Compared {len(rows)} soft outputs from {len(manifest['cases'])} cases against independent references.\n\n"
        f"Soft pass/fail: {summary['soft_pass']}/{summary['soft_fail']}; hard pixel mismatches: {summary['hard_pixel_mismatch']}; minimum hard IoU: {summary['hard_min_iou']:.6g}.\n\n"
        f"Worst soft case: `{worst['case_id']}` / `{worst['quadrant']}` at r={worst['feather_radius']}, MaxAE={worst['maxae']:.6g}, MAE={worst['mae']:.6g}, RMSE={worst['rmse']:.6g}.\n\n"
        "Recreate with `python scripts/generate_quadrant_mask_baselines.py` followed by `python scripts/evaluate_quadrant_mask.py --output-dir artifacts/module-2/quadrant-mask`.\n"
    )
    (output / "README.md").write_text(report, encoding="utf-8")
    (logs / "evaluation.log").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=Path("tests/fixtures/quadrant_mask"))
    parser.add_argument(
        "--output-dir",
        "--output",
        dest="output",
        type=Path,
        default=Path("artifacts/module-2/quadrant-mask"),
    )
    args = parser.parse_args()
    summary = evaluate(args.fixtures, args.output)
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
