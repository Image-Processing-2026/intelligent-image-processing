"""Run locked Module 2 cases and emit a machine-readable acceptance summary.

This runner deliberately treats missing data/configuration/runtime as exit 2,
quality failures as exit 1, and exits 0 only when every selected case ran and
met its declared gate.  It never injects a fake resolver in ``--mode real``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validate_module2_dataset import validate  # noqa: E402

from src.region_engine.controller import (  # noqa: E402
    RegionBackendUnavailableError,
    RegionEngineError,
    resolve_region,
)


def _read_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def _path(value: str) -> Path:
    return (REPO_ROOT / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _load_hard_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image)
    if value.ndim != 2:
        raise ValueError("ground-truth mask must be a single-channel image")
    return value != 0


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    intersection = int(np.logical_and(prediction, target).sum())
    predicted_area = int(prediction.sum())
    target_area = int(target.sum())
    union = predicted_area + target_area - intersection
    return {
        "iou": float(intersection / union) if union else 1.0,
        "dice": float(2 * intersection / (predicted_area + target_area))
        if predicted_area + target_area
        else 1.0,
        "predicted_area": predicted_area,
        "target_area": target_area,
    }


def _git_state() -> dict[str, str | bool | None]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"sha": sha, "dirty": dirty}
    except Exception:
        return {"sha": None, "dirty": None}


def evaluate(cases_path: Path, *, split: str, mode: str, output_dir: Path) -> tuple[int, dict[str, Any]]:
    preflight = validate(cases_path, require_reviewed=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_summary: dict[str, Any] = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "git": _git_state(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "mode": mode,
            "device": "cpu",
            "offline": os.environ.get("HF_HUB_OFFLINE", "") == "1",
        },
        "cases_path": str(cases_path),
        "preflight": preflight,
        "results": [],
    }
    if not preflight["valid"]:
        base_summary["status"] = "configuration_error" if not preflight["missing"] else "missing_data"
        return 2, base_summary
    selected = [case for case in _read_cases(cases_path) if case["split"] == split]
    if not selected:
        base_summary["status"] = "configuration_error"
        base_summary["preflight"]["errors"].append(f"no cases for split {split!r}")
        return 2, base_summary
    runtime_error = False
    quality_failure = False
    for case in selected:
        case_id = case["case_id"]
        started = time.perf_counter()
        row: dict[str, Any] = {"case_id": case_id, "status": "error"}
        try:
            image = _load_rgb(_path(case["image"]))
            result = resolve_region(image, case["request"])
            if case["gt_kind"] == "empty":
                target = np.zeros(image.shape[:2], dtype=bool)
            else:
                target = _load_hard_mask(_path(case["mask"]))
            if result.instance_masks:
                prediction = np.logical_or.reduce(tuple(mask > 0.0 for mask in result.instance_masks))
            else:
                prediction = result.mask >= 0.5
            metrics = _metrics(prediction, target)
            min_iou = float(case.get("min_iou", 1.0 if case["gt_kind"] == "empty" else 0.0))
            min_dice = float(case.get("min_dice", 1.0 if case["gt_kind"] == "empty" else 0.0))
            passed = metrics["iou"] >= min_iou and metrics["dice"] >= min_dice
            row.update(
                {
                    "status": "passed" if passed else "failed",
                    "region_status": result.status,
                    "metrics": metrics,
                    "thresholds": {"min_iou": min_iou, "min_dice": min_dice},
                    "metadata": result.metadata,
                }
            )
            np.save(output_dir / f"{case_id}_hard_prediction.npy", prediction.astype(bool))
            if not passed:
                quality_failure = True
        except RegionBackendUnavailableError as exc:
            runtime_error = True
            row.update({"status": "runtime_unavailable", "error": str(exc)})
        except (RegionEngineError, OSError, ValueError, KeyError, TypeError) as exc:
            runtime_error = True
            row.update({"status": "error", "error": str(exc)})
        finally:
            row["timing_ms"] = round((time.perf_counter() - started) * 1000, 3)
            base_summary["results"].append(row)
    counts = {status: 0 for status in ("passed", "failed", "error", "runtime_unavailable")}
    for result in base_summary["results"]:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    base_summary["counts"] = counts
    if runtime_error:
        base_summary["status"] = "runtime_error"
        return 2, base_summary
    if quality_failure:
        base_summary["status"] = "quality_failed"
        return 1, base_summary
    base_summary["status"] = "passed"
    return 0, base_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--split", choices=("calibration", "test"), required=True)
    parser.add_argument("--mode", choices=("real",), default="real")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    code, summary = evaluate(args.cases, split=args.split, mode=args.mode, output_dir=args.output_dir)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
