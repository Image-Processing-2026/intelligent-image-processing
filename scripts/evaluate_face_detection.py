"""Evaluate mock face-mask fixtures and emit reproducible diagnostics.

The default run also checks readiness of the real checkpoint/assets and exits
non-zero when they are absent. Use ``--mock-only`` for the dependency-free
geometry report; that mode never claims real-model validation.
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

import src.region_engine.face_detector as face_detector  # noqa: E402
from src.region_engine.face_detector import FaceDetectionRecord, detect_faces  # noqa: E402
from src.region_engine.mask_utils import blend_regions  # noqa: E402

MAXAE_THRESHOLD = 1e-6
MAE_THRESHOLD = 1e-7


class _FixtureBackend:
    def __init__(self, records: list[FaceDetectionRecord]) -> None:
        self.records = records

    def detect(self, _image: np.ndarray) -> list[FaceDetectionRecord]:
        return self.records

    def close(self) -> None:
        pass


def _metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    absolute = np.abs(difference)
    return {
        "maxae": float(np.max(absolute)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(difference**2))),
    }


def _image(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    yy, xx = np.indices(shape)
    return np.stack(
        [
            np.floor(255 * xx / max(width - 1, 1)),
            np.floor(255 * yy / max(height - 1, 1)),
            80 + 80 * ((xx // 2 + yy // 2) % 2),
        ],
        axis=-1,
    ).astype(np.uint8)


def _save_image(array: np.ndarray, path: Path) -> None:
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB").save(path)


def _run_case(
    case: dict[str, Any], fixture_root: Path, arrays: Path, images: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    shape = tuple(case["shape"])
    records = [FaceDetectionRecord(**item) for item in case["detections"]]
    backend = _FixtureBackend(records)
    face_detector.reset_face_detector()
    face_detector._DETECTOR_FACTORY = lambda _path: backend
    image = _image(shape)
    radius = 3
    masks = detect_faces(image, feather_radius=radius, expand_ratio=case["expand_ratio"])
    actual = np.maximum.reduce(masks) if masks else np.zeros(shape, dtype=np.float32)
    expected = np.load(fixture_root / case["soft_masks"][str(radius)]["merged"])
    metrics = _metrics(actual, expected)
    metrics_row: dict[str, Any] = {
        "case_id": next(key for key, value in _CURRENT_MANIFEST["cases"].items() if value is case),
        "height": shape[0],
        "width": shape[1],
        "expected_faces": case["mask_count"],
        "actual_faces": len(masks),
        "expand_ratio": case["expand_ratio"],
        "feather_radius": radius,
        **metrics,
        "passed": len(masks) == case["mask_count"] and metrics["maxae"] <= MAXAE_THRESHOLD and metrics["mae"] <= MAE_THRESHOLD,
    }
    case_id = metrics_row["case_id"]
    np.save(arrays / f"{case_id}_actual.npy", actual)
    np.save(arrays / f"{case_id}_expected.npy", expected)
    for index, mask in enumerate(masks):
        np.save(arrays / f"{case_id}_face_{index}.npy", mask)
    processed = np.clip(image.astype(np.int16) + 40, 0, 255).astype(np.uint8)
    hard_masks = detect_faces(image, feather_radius=0, expand_ratio=case["expand_ratio"])
    hard = np.maximum.reduce(hard_masks) if hard_masks else np.zeros(shape, dtype=np.float32)
    hard_blend = blend_regions(image, processed, hard)
    soft_blend = blend_regions(image, processed, actual)
    _save_image(image, images / f"{case_id}_original.png")
    _save_image(processed, images / f"{case_id}_processed.png")
    _save_image(hard_blend, images / f"{case_id}_hard_blend.png")
    _save_image(soft_blend, images / f"{case_id}_soft_blend.png")
    details = {"case_id": case_id, "bboxes": case["bboxes"], "scores": [item["score"] for item in case["detections"]]}
    return metrics_row, details


def _plot_before_after(output: Path, case_id: str, title: str) -> None:
    images = output / "images"
    entries = [
        ("Original", np.asarray(Image.open(images / f"{case_id}_original.png"))),
        ("Processed", np.asarray(Image.open(images / f"{case_id}_processed.png"))),
        ("Hard blend", np.asarray(Image.open(images / f"{case_id}_hard_blend.png"))),
        ("Soft blend", np.asarray(Image.open(images / f"{case_id}_soft_blend.png"))),
    ]
    figure, axes = plt.subplots(1, 4, figsize=(12, 3.5), constrained_layout=True)
    for axis, (label, image) in zip(axes, entries):
        axis.imshow(image)
        axis.set_title(label)
        axis.axis("off")
    figure.suptitle(title)
    figure.savefig(output / "plots" / ("01_single_face_before_after.png" if case_id == "CENTER" else "02_multiple_faces_before_after.png"), dpi=180)
    plt.close(figure)


def _plot_edge_and_empty(output: Path) -> None:
    images = output / "images"
    figure, axes = plt.subplots(1, 2, figsize=(7, 3.5), constrained_layout=True)
    for axis, case_id, title in zip(axes, ("EDGE", "EMPTY"), ("Edge clipping", "No face: unchanged")):
        axis.imshow(np.asarray(Image.open(images / f"{case_id}_soft_blend.png")))
        axis.set_title(title)
        axis.axis("off")
    figure.savefig(output / "plots" / "03_edge_face_clipping.png", dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(7, 3.5), constrained_layout=True)
    for axis, suffix in zip(axes, ("original", "soft_blend")):
        axis.imshow(np.asarray(Image.open(images / f"EMPTY_{suffix}.png")))
        axis.set_title(f"EMPTY {suffix}")
        axis.axis("off")
    figure.savefig(output / "plots" / "04_no_face_unchanged.png", dpi=180)
    plt.close(figure)


def _plot_bbox_and_error(output: Path, fixture_root: Path, case: dict[str, Any]) -> None:
    plots = output / "plots"
    figure, axis = plt.subplots(figsize=(5, 4), constrained_layout=True)
    shape = tuple(case["shape"])
    axis.imshow(_image(shape))
    for index, (xmin, ymin, xmax, ymax) in enumerate(case["bboxes"]):
        axis.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, label=f"face {index}"))
    axis.legend()
    axis.set_title("Raw/expanded bbox fixture")
    figure.savefig(plots / "05_bbox_baseline_vs_prediction.png", dpi=180)
    plt.close(figure)

    actual = np.load(output / "arrays" / "CENTER_actual.npy")
    expected = np.load(output / "arrays" / "CENTER_expected.npy")
    error = np.abs(actual.astype(np.float64) - expected)
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    heatmap = axes[0].imshow(error, cmap="magma", vmin=0, vmax=1)
    axes[0].set_title(f"MaxAE={error.max():.2e}")
    axes[0].axis("off")
    figure.colorbar(heatmap, ax=axes[0], fraction=0.046)
    index = actual.shape[1] // 2
    axes[1].plot(actual[:, index], label="actual")
    axes[1].plot(expected[:, index], "--", label="reference")
    axes[1].set_title(f"Vertical profile x={index}")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend()
    figure.savefig(plots / "06_mask_error_and_profile.png", dpi=180)
    plt.close(figure)


def _plot_latency(output: Path, image: np.ndarray, records: list[FaceDetectionRecord]) -> list[float]:
    backend = _FixtureBackend(records)
    face_detector.reset_face_detector()
    face_detector._DETECTOR_FACTORY = lambda _path: backend
    detect_faces(image, feather_radius=0)
    timings: list[float] = []
    for _ in range(30):
        started = time.perf_counter()
        detect_faces(image, feather_radius=0)
        timings.append((time.perf_counter() - started) * 1000)
    figure, axis = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    axis.plot(range(1, len(timings) + 1), timings)
    axis.set(xlabel="warm iteration", ylabel="milliseconds", title="Mock CPU mask latency")
    figure.savefig(output / "plots" / "07_cpu_latency.png", dpi=180)
    plt.close(figure)
    return timings


def _real_readiness(manifest_path: Path) -> dict[str, Any]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = data["model"]
    model_path = REPO_ROOT / model["path"]
    cases = data.get("real_model_cases", {})
    missing = [str(model_path)] if not model_path.is_file() else []
    for case in cases.values():
        image_path = REPO_ROOT / case["image"]
        if not image_path.is_file():
            missing.append(str(image_path))
    try:
        import mediapipe  # noqa: F401
    except Exception as exc:
        return {"ready": False, "missing": missing, "dependency_error": str(exc)}
    return {"ready": not missing, "missing": missing, "dependency_error": None}


def evaluate(fixtures: Path, manifest_path: Path, output: Path, mock_only: bool) -> dict[str, Any]:
    global _CURRENT_MANIFEST
    _CURRENT_MANIFEST = json.loads((fixtures / "manifest.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    arrays, images, plots, logs = (output / name for name in ("arrays", "images", "plots", "logs"))
    for directory in (arrays, images, plots, logs):
        directory.mkdir(exist_ok=True)

    rows: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    for case in _CURRENT_MANIFEST["cases"].values():
        row, detail = _run_case(case, fixtures, arrays, images)
        rows.append(row)
        detections.append(detail)
    _plot_before_after(output, "CENTER", "Single face")
    _plot_before_after(output, "MULTI", "Multiple faces")
    _plot_edge_and_empty(output)
    _plot_bbox_and_error(output, fixtures, _CURRENT_MANIFEST["cases"]["CENTER"])
    latency = _plot_latency(output, _image((10, 12)), [FaceDetectionRecord(1, 1, 3, 3, 0.9)])
    real = {"ready": True, "mode": "mock-only"} if mock_only else _real_readiness(manifest_path)
    summary: dict[str, Any] = {
        "mock_only": mock_only,
        "mock_cases": len(rows),
        "mock_pass": sum(bool(row["passed"]) for row in rows),
        "mock_fail": sum(not row["passed"] for row in rows),
        "real_model": real,
        "latency_ms": {"p50": float(np.percentile(latency, 50)), "p95": float(np.percentile(latency, 95))},
    }
    summary["passed"] = summary["mock_fail"] == 0 and (mock_only or real["ready"])
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "detections.json").write_text(json.dumps(detections, indent=2) + "\n", encoding="utf-8")
    (output / "run_manifest.json").write_text(
        json.dumps(
            {"fixture_manifest": str((fixtures / "manifest.json").as_posix()), "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (logs / "evaluation.log").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Face-detection evaluation\n\n"
        f"Mock fixture cases: {summary['mock_pass']} passed, {summary['mock_fail']} failed.\n\n"
        "Mock mode validates bbox-to-mask geometry only; it does not validate MediaPipe inference. "
        "The default command also requires prepared real assets and exits non-zero when they are missing.\n\n"
        "Recreate with `python scripts/generate_face_mask_baselines.py` followed by `python scripts/evaluate_face_detection.py --mock-only`.\n",
        encoding="utf-8",
    )
    return summary


_CURRENT_MANIFEST: dict[str, Any] = {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=Path("tests/fixtures/face_detection"))
    parser.add_argument("--manifest", type=Path, default=Path("tests/fixtures/face_detection/assets.json"))
    parser.add_argument("--output-dir", "--output", dest="output", type=Path, default=Path("artifacts/module-2/detect-faces"))
    parser.add_argument("--mock-only", action="store_true")
    args = parser.parse_args()
    summary = evaluate(args.fixtures, args.manifest, args.output, args.mock_only)
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
