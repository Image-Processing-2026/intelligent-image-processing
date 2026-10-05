"""Real-checkpoint integration tests.

These tests intentionally fail with an actionable message when the explicitly
prepared model or upstream image assets are absent. They must not be changed
to a silent skip that looks like a detector pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.region_engine.face_detector import MODEL_PATH_ENV, detect_faces, reset_face_detector

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "face_detection" / "assets.json"


def _real_case(case_id: str) -> tuple[Path, Path, dict]:
    manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    model_path = REPO_ROOT / manifest["model"]["path"]
    case = manifest["real_model_cases"][case_id]
    image_path = REPO_ROOT / case["image"]
    return model_path, image_path, case


def _require_real_assets(case_id: str) -> tuple[Path, Path, dict]:
    model_path, image_path, case = _real_case(case_id)
    missing = [str(path) for path in (model_path, image_path) if not path.is_file()]
    if missing:
        pytest.fail(
            "real face-detector assets are missing: "
            + ", ".join(missing)
            + ". Run prepare_face_detection_assets.py with a pinned URL and sha256."
        )
    return model_path, image_path, case


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0.0)
    if len(xs) == 0:
        return None
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    ]


def _bbox_iou(first: list[int], second: list[int]) -> float:
    ax0, ay0, aw, ah = first
    bx0, by0, bw, bh = second
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    intersection = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(
        0, min(ay1, by1) - max(ay0, by0)
    )
    union = aw * ah + bw * bh - intersection
    return 0.0 if union <= 0 else intersection / union


@pytest.mark.real_model
def test_full_range_model_detects_upstream_portrait(monkeypatch: pytest.MonkeyPatch) -> None:
    model_path, image_path, case = _require_real_assets("portrait_full_range")
    monkeypatch.setenv(MODEL_PATH_ENV, str(model_path))
    reset_face_detector()

    masks = detect_faces(_load_rgb(image_path), feather_radius=0)

    assert len(masks) == case["expected_count"]
    assert masks[0].dtype == np.float32
    actual_bbox = _bbox_from_mask(masks[0])
    assert actual_bbox is not None
    assert _bbox_iou(actual_bbox, case["expected_bbox"]) >= 0.5
    reset_face_detector()


@pytest.mark.real_model
def test_full_range_model_returns_no_face_for_upstream_negative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path, image_path, case = _require_real_assets("no_face")
    monkeypatch.setenv(MODEL_PATH_ENV, str(model_path))
    reset_face_detector()

    assert detect_faces(_load_rgb(image_path), feather_radius=0) == []
    assert case["expected_count"] == 0
    reset_face_detector()
