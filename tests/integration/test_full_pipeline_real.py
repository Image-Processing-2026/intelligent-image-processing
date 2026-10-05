import os
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from src.agent.graph import run_pipeline

DATA_DIR = os.path.join("data", "real")


def _load_real_image(filename: str) -> np.ndarray:
    """Đọc ảnh mẫu thực (RGB); skip test nếu ảnh không có (data/real bị gitignore)."""
    path = os.path.join(DATA_DIR, filename)
    img = cv2.imread(path)
    if img is None:
        pytest.skip(
            f"Real sample image '{path}' is missing; "
            "run scripts/download_real_samples.py to fetch it."
        )
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@pytest.fixture
def dark_img():
    return _load_real_image("coffee_underexposed.jpg")


@pytest.fixture
def noisy_img():
    return _load_real_image("astro_noisy.jpg")


def test_e2e_dark_img(dark_img):
    with patch.dict(os.environ, clear=True):
        state = run_pipeline(image=dark_img, is_synthetic=False, max_iterations=2)
        assert state["decision"] in ["SHIP", "STOP_BEST_EFFORT"]
        assert len(state["history"]) >= 1


def test_e2e_noisy_img(noisy_img):
    with patch.dict(os.environ, clear=True):
        state = run_pipeline(image=noisy_img, is_synthetic=False, max_iterations=2)
        assert len(state["history"]) > 0


def test_e2e_degradation_guard(noisy_img):
    def mock_execute(*args, **kwargs):
        degraded = noisy_img.copy().astype(np.float32)
        noise = np.random.normal(0, 50, degraded.shape)
        return np.clip(degraded + noise, 0, 255).astype(np.uint8)

    with (
        patch.dict(os.environ, clear=True),
        patch("src.agent.graph.execute_plan", side_effect=mock_execute),
    ):
        state = run_pipeline(image=noisy_img, is_synthetic=False, max_iterations=2)
        assert state["decision"] == "STOP_BEST_EFFORT"
