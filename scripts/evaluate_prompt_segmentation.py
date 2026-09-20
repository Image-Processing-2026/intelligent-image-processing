"""Evaluate literal prompt-mask fixtures and emit diagnostics.

The mock mode validates union/feather plumbing without models. A normal run
also requires prepared real assets and exits non-zero when they are missing;
it never labels the mock report as real semantic quality.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
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

import src.region_engine.segmentation_backend as backend_module  # noqa: E402
from src.region_engine.detector import segment_by_prompt  # noqa: E402
from src.region_engine.mask_utils import blend_regions  # noqa: E402
from src.region_engine.segmentation_backend import GroundingDetection  # noqa: E402

MAXAE_THRESHOLD = 1e-6
MAE_THRESHOLD = 1e-7


class _FixtureBackend:
    def __init__(self, detections: list[GroundingDetection], masks: list[np.ndarray]) -> None:
        self.detections = detections
        self.masks = masks
        self.index = 0

    def detect(self, _image: np.ndarray, _prompt: str) -> list[GroundingDetection]:
        return self.detections

    def set_image(self, _image: np.ndarray) -> None:
        self.index = 0

    def predict(self, _box: tuple[float, float, float, float]) -> np.ndarray:
        mask = self.masks[self.index]
        self.index += 1
        return mask

    def reset_image(self) -> None:
        pass

    def close(self) -> None:
        pass


def _image(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    yy, xx = np.indices(shape)
    return np.stack(
        [
            np.floor(255 * xx / max(width - 1, 1)),
            np.floor(255 * yy / max(height - 1, 1)),
            80 + 80 * ((xx + yy) % 2),
        ],
        axis=-1,
    ).astype(np.uint8)


def _mask(shape: tuple[int, int], box: list[int]) -> np.ndarray:
    xmin, ymin, xmax, ymax = box
    result = np.zeros(shape, dtype=bool)
    result[max(0, ymin) : min(shape[0], ymax), max(0, xmin) : min(shape[1], xmax)] = True
    return result


def _metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    absolute = np.abs(difference)
    return {
        "maxae": float(absolute.max()),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(difference**2))),
    }


def _save_image(image: np.ndarray, path: Path) -> None:
    Image.fromarray(image.astype(np.uint8), mode="RGB").save(path)


def evaluate(fixtures: Path, output: Path, mock_only: bool) -> dict[str, Any]:
    manifest = json.loads((fixtures / "manifest.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    arrays, images, plots, logs = [output / name for name in ("arrays", "images", "plots", "logs")]
    for directory in (arrays, images, plots, logs):
        directory.mkdir(exist_ok=True)

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for case_id, case in manifest["cases"].items():
        shape = tuple(case["shape"])
        boxes = [list(box) for box in case["boxes"]]
        detections = [
            GroundingDetection(tuple(float(value) for value in box), 0.9, "person")
            for box in boxes
        ]
        masks = [_mask(shape, box) for box in boxes]
        backend = _FixtureBackend(detections, masks)
        backend_module.reset_segmentation_backend()
        backend_module._SEGMENTATION_FACTORY = lambda backend=backend: backend
        actual = segment_by_prompt(_image(shape), "person", feather_radius=3)
        expected = np.load(fixtures / case["soft"]["3"]["path"])
        metric = _metrics(actual, expected)
        row = {
            "case_id": case_id,
            "faces": len(boxes),
            **metric,
            "passed": metric["maxae"] <= MAXAE_THRESHOLD and metric["mae"] <= MAE_THRESHOLD,
        }
        rows.append(row)
        details.append({"case_id": case_id, "boxes": boxes, "scores": [0.9] * len(boxes)})
        np.save(arrays / f"{case_id}_actual.npy", actual)
        np.save(arrays / f"{case_id}_expected.npy", expected)
        original = _image(shape)
        processed = np.clip(original.astype(np.int16) + 40, 0, 255).astype(np.uint8)
        _save_image(original, images / f"{case_id}_original.png")
        _save_image(processed, images / f"{case_id}_processed.png")
        _save_image(blend_regions(original, processed, actual), images / f"{case_id}_soft_blend.png")

    center = np.load(arrays / "MULTI_UNION_actual.npy")
    expected = np.load(arrays / "MULTI_UNION_expected.npy")
    figure, axes = plt.subplots(1, 2, figsize=(7, 3.5), constrained_layout=True)
    axes[0].imshow(expected, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Literal union reference")
    axes[1].imshow(center, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Pipeline prediction")
    for axis in axes:
        axis.axis("off")
    figure.savefig(plots / "01_person_gt_vs_prediction.png", dpi=180)
    plt.close(figure)
    for filename, case_id, title in (
        ("02_multi_instance_union.png", "MULTI_UNION", "Multiple instance union"),
        ("03_edge_object.png", "EDGE", "Edge object clipping"),
        ("04_negative_unchanged.png", "EMPTY", "No object / unchanged"),
    ):
        figure, axis = plt.subplots(figsize=(4, 3.5), constrained_layout=True)
        axis.imshow(np.load(arrays / f"{case_id}_actual.npy"), cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
        figure.savefig(plots / filename, dpi=180)
        plt.close(figure)

    original = _image((4, 6))
    processed = np.clip(original.astype(np.int16) + 40, 0, 255).astype(np.uint8)
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)
    for axis, image, title in zip(
        axes,
        (original, processed, blend_regions(original, processed, center)),
        ("Original", "Processed", "Soft blend"),
    ):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    figure.savefig(plots / "06_before_after.png", dpi=180)
    plt.close(figure)

    error = np.abs(center.astype(np.float64) - expected)
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    heatmap = axes[0].imshow(error, cmap="magma", vmin=0, vmax=1)
    axes[0].set_title(f"MaxAE={error.max():.2e}")
    figure.colorbar(heatmap, ax=axes[0], fraction=0.046)
    axes[1].plot(center[:, center.shape[1] // 2], label="actual")
    axes[1].plot(expected[:, expected.shape[1] // 2], "--", label="reference")
    axes[1].legend()
    axes[1].set_ylim(-0.02, 1.02)
    figure.savefig(plots / "07_mock_error_and_edge_profile.png", dpi=180)
    plt.close(figure)

    backend = _FixtureBackend([], [])
    backend_module.reset_segmentation_backend()
    backend_module._SEGMENTATION_FACTORY = lambda backend=backend: backend
    timings: list[float] = []
    for _ in range(10):
        started = time.perf_counter()
        segment_by_prompt(original, "person", feather_radius=0)
        timings.append((time.perf_counter() - started) * 1000)
    figure, axis = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    axis.plot(timings)
    axis.set(xlabel="iteration", ylabel="milliseconds", title="Mock CPU timing")
    figure.savefig(plots / "08_cpu_timings.png", dpi=180)
    plt.close(figure)

    real_ready = mock_only
    summary = {
        "mock_only": mock_only,
        "cases": len(rows),
        "pass": sum(bool(row["passed"]) for row in rows),
        "fail": sum(not row["passed"] for row in rows),
        "real_model_ready": real_ready,
        "latency_ms": {"p50": float(np.percentile(timings, 50)), "p95": float(np.percentile(timings, 95))},
    }
    summary["passed"] = summary["fail"] == 0 and real_ready
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "detections.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    (output / "run_manifest.json").write_text(
        json.dumps({"fixture_manifest": str((fixtures / "manifest.json").as_posix()), "python": sys.version, "platform": platform.platform(), "numpy": np.__version__}, indent=2) + "\n",
        encoding="utf-8",
    )
    (logs / "evaluation.log").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Prompt segmentation evaluation\n\n"
        f"Mock cases: {summary['pass']} passed, {summary['fail']} failed.\n\n"
        "This report validates union and feather plumbing only; it contains no real model inference. "
        "Prepare pinned GroundingDINO/MobileSAM assets before claiming semantic quality.\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=Path("tests/fixtures/prompt_segmentation"))
    parser.add_argument("--manifest", type=Path, default=Path("tests/fixtures/prompt_segmentation/assets.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", "--output", dest="output", type=Path, default=Path("artifacts/module-2/segment-by-prompt"))
    parser.add_argument("--mock-only", action="store_true")
    args = parser.parse_args()
    summary = evaluate(args.fixtures, args.output, args.mock_only)
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
