"""Real GroundingDINO + MobileSAM integration checks.

The tests fail with an actionable message when pinned local assets are absent;
they do not silently convert a missing model into a skipped pass.
"""

from __future__ import annotations

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

REPO_ROOT = Path(__file__).resolve().parents[2]
DINO_PATH = REPO_ROOT / "models" / "segmentation" / "grounding-dino-tiny"
SAM_PATH = REPO_ROOT / "models" / "segmentation" / "mobile_sam.pt"
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
        str(path)
        for path in (DINO_PATH, SAM_PATH, PERSON_IMAGE, PERSON_MASK)
        if not path.exists()
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
