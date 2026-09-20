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


def _require_assets() -> None:
    missing = [str(path) for path in (DINO_PATH, SAM_PATH, PERSON_IMAGE) if not path.exists()]
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
    assert np.count_nonzero(mask) > 0
    reset_segmentation_backend()
