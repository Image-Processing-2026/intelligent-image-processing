"""Evaluate the production soft-mask implementation against SciPy fixtures."""

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

from src.region_engine.mask_utils import create_soft_mask  # noqa: E402


def _display_mask(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask)
    if values.dtype == np.uint8 and np.any(values == 255):
        return values.astype(np.float64) / 255.0
    return values.astype(np.float64)


def _metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    absolute = np.abs(difference)
    return {
        "e_max": float(np.max(absolute)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(difference**2))),
    }


def _comparison_plot(
    input_mask: np.ndarray,
    actual: np.ndarray,
    expected: np.ndarray,
    metrics: dict[str, float],
    path: Path,
) -> None:
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    figure, axes = plt.subplots(1, 4, figsize=(14, 3.5), constrained_layout=True)
    images = [
        (_display_mask(input_mask), "Binary input", "gray", 0.0, 1.0),
        (actual, "Actual", "gray", 0.0, 1.0),
        (expected, "SciPy baseline", "gray", 0.0, 1.0),
        (error, f"Absolute error\nE_max={metrics['e_max']:.3g}", "magma", 0.0, None),
    ]
    for axis, (image, title, cmap, vmin, vmax) in zip(axes, images):
        rendered = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")
        if cmap != "gray":
            figure.colorbar(rendered, ax=axis, fraction=0.046, pad=0.04)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _step_profile(input_mask: np.ndarray, actual: np.ndarray, expected: np.ndarray, path: Path) -> None:
    row = input_mask.shape[0] // 2
    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    axis.plot(_display_mask(input_mask)[row], label="Input", drawstyle="steps-mid")
    axis.plot(actual[row], label="Actual")
    axis.plot(expected[row], "--", label="SciPy baseline")
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("Column")
    axis.set_ylabel("Alpha")
    axis.set_title("STEP center-row profile")
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _parameter_plot(rect: np.ndarray, output: Path) -> None:
    feather_values = (0, 3, 15, 25)
    figure, axes = plt.subplots(1, len(feather_values), figsize=(13, 3.2), constrained_layout=True)
    for axis, feather_radius in zip(axes, feather_values):
        actual = create_soft_mask(rect, feather_radius=feather_radius)
        kernel_size = feather_radius if feather_radius in (0, 1) or feather_radius % 2 else feather_radius + 1
        sigma = 0.0 if feather_radius == 0 else kernel_size / 3.0
        axis.imshow(actual, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(f"f={feather_radius}, k={kernel_size}\nσ={sigma:g}")
        axis.axis("off")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def evaluate(fixtures: Path, output: Path) -> dict[str, Any]:
    manifest = json.loads((fixtures / "manifest.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    arrays = output / "arrays"
    plots = output / "plots"
    arrays.mkdir(exist_ok=True)
    plots.mkdir(exist_ok=True)

    rows: list[dict[str, Any]] = []
    loaded: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]] = {}
    for case_id, case in manifest["cases"].items():
        input_mask = np.load(fixtures / case["input"])
        expected = np.load(fixtures / case["expected"])
        actual = create_soft_mask(input_mask, feather_radius=case["feather_radius"])
        case_metrics = _metrics(actual, expected)
        np.save(arrays / f"{case_id}_actual.npy", actual)
        rows.append({"case_id": case_id, "feather_radius": case["feather_radius"], **case_metrics})
        loaded[case_id] = (input_mask, actual, expected, case_metrics)

    for case_id in ("step", "rect", "touch_edge", "impulse", "thin"):
        if case_id in loaded:
            _comparison_plot(*loaded[case_id], plots / f"{case_id}_comparison.png")
    if "step" in loaded:
        _step_profile(loaded["step"][0], loaded["step"][1], loaded["step"][2], plots / "step_profile.png")
    if "rect" in loaded:
        _parameter_plot(loaded["rect"][0], plots / "feather_parameter_comparison.png")

    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "feather_radius", "e_max", "mae", "rmse"])
        writer.writeheader()
        writer.writerows(rows)

    worst = max(rows, key=lambda row: row["e_max"])
    summary = {
        "case_count": len(rows),
        "passed_atol_1e-6": sum(row["e_max"] <= 1e-6 for row in rows),
        "worst_case": worst,
        "threshold": {"atol": 1e-6, "rtol": 0},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "environment.json").write_text(
        json.dumps(
            {
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
        "# Module 2 soft-mask evaluation\n\n"
        f"Compared {len(rows)} cases against the SciPy float64 reference. "
        f"{summary['passed_atol_1e-6']} cases are within `atol=1e-6`.\n\n"
        f"Worst case: `{worst['case_id']}`, E_max={worst['e_max']:.6g}, "
        f"MAE={worst['mae']:.6g}, RMSE={worst['rmse']:.6g}.\n"
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=Path("tests/fixtures/soft_mask"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/module-2/create-soft-mask"))
    args = parser.parse_args()
    summary = evaluate(args.fixtures, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
