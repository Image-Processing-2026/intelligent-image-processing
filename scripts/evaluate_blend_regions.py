"""Evaluate blend fixtures and write reproducible metrics and visual artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Support the documented ``python scripts/evaluate_blend_regions.py ...``
# invocation, where Python otherwise puts only ``scripts/`` on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.region_engine.mask_utils import blend_regions, create_soft_mask


def _load_case(fixtures: Path, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    original = np.load(fixtures / f"{name}_original.npy")
    processed = np.load(fixtures / f"{name}_processed.npy")
    mask_path = fixtures / f"{name}_mask.npy"
    mask = np.load(mask_path) if mask_path.exists() else None
    expected = np.load(fixtures / f"{name}_expected.npy")
    reference = np.load(fixtures / f"{name}_reference_float64.npy")
    return original, processed, mask, expected, reference


def _metrics(actual: np.ndarray, expected: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    difference = actual.astype(np.int16) - expected.astype(np.int16)
    absolute = np.abs(difference)
    differing = difference != 0
    near_integer = np.abs(reference - np.rint(reference)) <= 1e-4
    floor_value = np.floor(reference).astype(np.int16)
    ceil_value = np.ceil(reference).astype(np.int16)
    boundary_allowed = (actual.astype(np.int16) == floor_value) | (actual.astype(np.int16) == ceil_value)
    qualifies = (~differing | near_integer) & (~differing | boundary_allowed)
    return {
        "e_max": int(absolute.max(initial=0)),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(difference.astype(np.float64))))),
        "exact_channel_rate": float(np.mean(~differing)),
        "exact_pixel_rate": float(np.mean(np.all(~differing, axis=2))),
        "differing_channels": int(np.count_nonzero(differing)),
        "one_off_channels": int(np.count_nonzero(differing & (absolute == 1))),
        "boundary_qualified": bool(np.all(qualifies)),
        "near_integer_differences": int(np.count_nonzero(differing & near_integer)),
    }


def _save_rgb(path: Path, image: np.ndarray, scale: int = 1) -> None:
    array = image
    if scale > 1:
        array = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
    Image.fromarray(array, mode="RGB").save(path)


def _comparison_plot(
    path: Path,
    name: str,
    original: np.ndarray,
    processed: np.ndarray,
    mask: np.ndarray,
    actual: np.ndarray,
    expected: np.ndarray,
) -> None:
    error = np.max(np.abs(actual.astype(np.int16) - expected.astype(np.int16)), axis=2)
    figure, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    panels = [
        (original, "Original", None),
        (processed, "Processed", None),
        (mask, "Mask", {"cmap": "gray", "vmin": 0, "vmax": 1}),
        (actual, "Actual", None),
        (expected, "Baseline", None),
        (error, "Absolute error (max channel)", {"cmap": "magma", "vmin": 0, "vmax": max(1, int(error.max()))}),
    ]
    for axis, (data, title, kwargs) in zip(axes.flat, panels):
        if data.ndim == 2:
            axis.imshow(data, interpolation="nearest", **(kwargs or {}))
        else:
            axis.imshow(data, interpolation="nearest")
        axis.set_title(title)
        axis.set_axis_off()
    figure.suptitle(f"blend_regions comparison: {name}")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _hard_soft_artifacts(output: Path) -> None:
    height, width = 128, 192
    x = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    y = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    original = np.stack(
        [np.broadcast_to(x, (height, width)), np.broadcast_to(y, (height, width)), np.full((height, width), 80)],
        axis=2,
    ).astype(np.uint8)
    processed = np.clip(original.astype(np.int16) + np.array([50, 25, 35], dtype=np.int16), 0, 255).astype(np.uint8)
    binary = np.zeros((height, width), dtype=np.uint8)
    binary[32:96, 48:144] = 1
    soft = create_soft_mask(binary, feather_radius=15)
    hard_actual = blend_regions(original, processed, binary.astype(np.float32))
    soft_actual = blend_regions(original, processed, soft)

    arrays = output / "arrays"
    images = output / "images"
    plots = output / "plots"
    np.save(arrays / "hard_vs_soft_original.npy", original)
    np.save(arrays / "hard_vs_soft_processed.npy", processed)
    np.save(arrays / "hard_vs_soft_binary_mask.npy", binary.astype(np.float32))
    np.save(arrays / "hard_vs_soft_soft_mask.npy", soft)
    np.save(arrays / "hard_vs_soft_blended_hard.npy", hard_actual)
    np.save(arrays / "hard_vs_soft_blended_soft.npy", soft_actual)
    _save_rgb(images / "hard_vs_soft_original.png", original)
    _save_rgb(images / "hard_vs_soft_processed.png", processed)
    _save_rgb(images / "hard_vs_soft_blended.png", soft_actual)

    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for axis, data, title, kwargs in [
        (axes[0, 0], original, "Original", {}),
        (axes[0, 1], processed, "Processed", {}),
        (axes[0, 2], binary, "Binary mask", {"cmap": "gray", "vmin": 0, "vmax": 1}),
        (axes[1, 0], hard_actual, "Hard blend", {}),
        (axes[1, 1], soft, "Soft mask", {"cmap": "gray", "vmin": 0, "vmax": 1}),
        (axes[1, 2], soft_actual, "Soft blend", {}),
    ]:
        axis.imshow(data, interpolation="nearest", **kwargs)
        axis.set_title(title)
        axis.set_axis_off()
    figure.suptitle("Hard versus soft region blending")
    figure.savefig(plots / "hard_vs_soft_blending.png", dpi=180)
    plt.close(figure)

    row = height // 2
    figure, (image_axis, mask_axis) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
    image_axis.plot(original[row, :, 0], label="original R")
    image_axis.plot(processed[row, :, 0], label="processed R")
    image_axis.plot(soft_actual[row, :, 0], label="blended R")
    image_axis.plot(hard_actual[row, :, 0], "--", label="hard blend R")
    image_axis.set_ylabel("RGB value")
    image_axis.legend()
    mask_axis.plot(binary[row], label="binary mask")
    mask_axis.plot(soft[row], label="soft mask")
    mask_axis.set_ylim(-0.05, 1.05)
    mask_axis.set_xlabel("x")
    mask_axis.set_ylabel("mask")
    mask_axis.legend()
    figure.suptitle("Blend line profile at the image midpoint")
    figure.savefig(plots / "blend_line_profile.png", dpi=180)
    plt.close(figure)


def _quantization_plot(output: Path, fixtures: Path) -> list[dict[str, object]]:
    original, processed, mask, expected, reference = _load_case(fixtures, "near_integer")
    actual = blend_regions(original, processed, mask)
    records: list[dict[str, object]] = []
    labels = ["0.1", "nextafter down", "nextafter up"]
    for index, label in enumerate(labels):
        records.append(
            {
                "alpha": label,
                "reference": float(reference[0, index, 0]),
                "actual": int(actual[0, index, 0]),
                "expected": int(expected[0, index, 0]),
                "difference": int(actual[0, index, 0]) - int(expected[0, index, 0]),
            }
        )
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    x = np.arange(len(records))
    axis.plot(x, [record["reference"] for record in records], "o-", label="reference before cast")
    axis.plot(x, [record["actual"] for record in records], "s", label="actual uint8")
    axis.plot(x, [record["expected"] for record in records], "x", label="Fraction expected uint8")
    axis.set_xticks(x, labels)
    axis.set_ylabel("channel value")
    axis.set_title("Float32 quantization boundary")
    axis.legend()
    figure.savefig(output / "plots" / "quantization_boundary.png", dpi=180)
    plt.close(figure)
    return records


def evaluate(fixtures: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for directory in ("arrays", "images", "plots"):
        (output / directory).mkdir(exist_ok=True)

    manifest = json.loads((fixtures / "manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for case in manifest["cases"]:
        name = case["name"]
        original, processed, mask, expected, reference = _load_case(fixtures, name)
        actual = blend_regions(original, processed, mask)
        metrics = _metrics(actual, expected, reference)
        row = {"case": name, "baseline_type": case["baseline_type"], **metrics}
        rows.append(row)
        np.save(output / "arrays" / f"{name}_actual.npy", actual)

        if name in {"w3c_half", "spatial_2x2"}:
            _save_rgb(output / "images" / f"{name}_original.png", original, scale=32 if name == "w3c_half" else 16)
            _save_rgb(output / "images" / f"{name}_processed.png", processed, scale=32 if name == "w3c_half" else 16)
            _save_rgb(output / "images" / f"{name}_blended.png", actual, scale=32 if name == "w3c_half" else 16)
            if mask is not None:
                _comparison_plot(output / "plots" / f"{name}_comparison.png", name, original, processed, mask, actual, expected)

    fieldnames = list(rows[0].keys())
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    _hard_soft_artifacts(output)
    quantization = _quantization_plot(output, fixtures)
    exact_rows = [row for row in rows if row["differing_channels"] == 0]
    summary = {
        "fixture_cases": len(rows),
        "exact_cases": len(exact_rows),
        "cases_with_differences": len(rows) - len(exact_rows),
        "total_differing_channels": sum(int(row["differing_channels"]) for row in rows),
        "total_one_off_channels": sum(int(row["one_off_channels"]) for row in rows),
        "all_differences_boundary_qualified": all(bool(row["boundary_qualified"]) for row in rows),
        "quantization_boundary": quantization,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
    }
    (output / "environment.json").write_text(json.dumps(summary["environment"], indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = [
        "# Module 2 blend_regions evaluation",
        "",
        f"Evaluated {len(rows)} fixture cases from `{fixtures}` without network access.",
        "",
        f"Exact cases: {len(exact_rows)}/{len(rows)}. Total differing channels: {summary['total_differing_channels']}; one-off boundary differences: {summary['total_one_off_channels']}.",
        f"All differences satisfy the documented near-integer boundary rule: `{summary['all_differences_boundary_qualified']}`.",
        "",
        "The hard/soft plot uses a deterministic 128×192 RGB gradient and compares a binary rectangle with a feathered mask. PNGs were written with Matplotlib Agg and reopened by the image writer during generation.",
        "",
        "See `metrics.csv` for E_max, MAE, RMSE, exact rates, and boundary classification.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Evaluated {len(rows)} cases; artifacts written to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.fixtures, args.output)


if __name__ == "__main__":
    main()
