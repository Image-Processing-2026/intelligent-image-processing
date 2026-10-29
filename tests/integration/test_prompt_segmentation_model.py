"""Real GroundingDINO + MobileSAM integration checks.

The tests fail with an actionable message when pinned local assets are absent;
they do not silently convert a missing model into a skipped pass.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.region_engine.detector import segment_by_prompt
from src.region_engine.segmentation_backend import (
    DINO_MODEL_ENV,
    MOBILE_SAM_CHECKPOINT_ENV,
    reset_segmentation_backend,
)
from src.region_engine.subject import clear_subject_mask_cache, resolve_subject_mask

REPO_ROOT = Path(__file__).resolve().parents[2]


def _asset_path(environment_name: str, default: Path) -> Path:
    """Ưu tiên biến môi trường (worktree không có models/), rồi mới tới đường dẫn mặc định."""
    configured = os.environ.get(environment_name, "").strip()
    return Path(configured) if configured else default


DINO_PATH = _asset_path(
    DINO_MODEL_ENV, REPO_ROOT / "models" / "segmentation" / "grounding-dino-tiny"
)
SAM_PATH = _asset_path(
    MOBILE_SAM_CHECKPOINT_ENV, REPO_ROOT / "models" / "segmentation" / "mobile_sam.pt"
)
BENCHMARK_ROOT = REPO_ROOT / "data" / "real" / "benchmark"
PERSON_IMAGE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "prompt_segmentation"
    / "penn_fudan"
    / "PNGImages"
    / "FudanPed00046.png"
)
PERSON_MASK = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "prompt_segmentation"
    / "penn_fudan"
    / "PedMasks"
    / "FudanPed00046_mask.png"
)


def _require_assets() -> None:
    missing = [
        str(path) for path in (DINO_PATH, SAM_PATH, PERSON_IMAGE, PERSON_MASK) if not path.exists()
    ]
    if missing:
        pytest.fail(
            "real prompt-segmentation assets are missing: "
            + ", ".join(missing)
            + ". Prepare pinned assets before running -m real_model."
        )


@pytest.mark.real_model
def test_grounding_dino_mobile_sam_person_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_assets()
    monkeypatch.setenv(DINO_MODEL_ENV, str(DINO_PATH))
    monkeypatch.setenv(MOBILE_SAM_CHECKPOINT_ENV, str(SAM_PATH))
    reset_segmentation_backend()
    with Image.open(PERSON_IMAGE) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()

    mask = segment_by_prompt(rgb, "person", feather_radius=0)

    assert mask.dtype == np.float32
    assert mask.shape == rgb.shape[:2]
    assert np.isfinite(mask).all()
    assert 0.0 <= float(mask.min()) <= float(mask.max()) <= 1.0
    with Image.open(PERSON_MASK) as image:
        expected = np.asarray(image.convert("L"), dtype=np.uint8) > 0
    predicted = mask > 0.0
    intersection = np.count_nonzero(predicted & expected)
    union = np.count_nonzero(predicted | expected)
    denominator = np.count_nonzero(predicted) + np.count_nonzero(expected)
    iou = 1.0 if union == 0 else intersection / union
    dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
    assert iou >= 0.3
    assert dice >= 0.45
    reset_segmentation_backend()


def _configure_models(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_assets()
    monkeypatch.setenv(DINO_MODEL_ENV, str(DINO_PATH))
    monkeypatch.setenv(MOBILE_SAM_CHECKPOINT_ENV, str(SAM_PATH))
    reset_segmentation_backend()
    clear_subject_mask_cache()


def _load_rgb(relative: str) -> np.ndarray:
    with Image.open(BENCHMARK_ROOT / relative) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:  # pragma: no cover - đo RSS chỉ là thông tin phụ
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


@pytest.mark.real_model
@pytest.mark.parametrize("relative", ["backlit/backlit_02.jpg", "multi_defect/multi_defect_01.jpg"])
def test_sky_without_bbox_is_never_a_frame_covering_dino_box(
    monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    _configure_models(monkeypatch)
    rgb = _load_rgb(relative)

    started = time.perf_counter()
    subject = resolve_subject_mask(rgb, "sky", None, use_semantic=True)
    elapsed = time.perf_counter() - started
    print(
        f"\n[{relative}] sky source={subject.source} conf={subject.confidence:.2f} "
        f"time={elapsed:.2f}s rss={_rss_mb()} MB"
    )

    assert subject.mask.dtype == np.float32
    assert subject.mask.shape == rgb.shape[:2]
    assert float(np.mean(subject.mask >= 0.5)) < 0.9
    if subject.box is not None:
        x0, y0, x1, y1 = subject.box
        assert (x1 - x0) * (y1 - y0) < 0.9 * rgb.shape[0] * rgb.shape[1]
    reset_segmentation_backend()
    clear_subject_mask_cache()


@pytest.mark.real_model
def test_person_subject_on_group_photo_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_models(monkeypatch)
    rgb = _load_rgb("group_small_faces/group_small_faces_01.jpg")

    started = time.perf_counter()
    subject = resolve_subject_mask(rgb, "person", None, use_semantic=True)
    first = time.perf_counter() - started
    started = time.perf_counter()
    second = resolve_subject_mask(rgb, "sky", None, use_semantic=True)
    reuse = time.perf_counter() - started
    print(
        f"\n[group_small_faces_01] person source={subject.source} "
        f"coverage={float(np.mean(subject.mask >= 0.5)):.3f} time={first:.2f}s; "
        f"sky (encoder reused) source={second.source} time={reuse:.2f}s rss={_rss_mb()} MB"
    )

    assert subject.source == "dino_sam"
    assert subject.mask.any()
    reset_segmentation_backend()
    clear_subject_mask_cache()
