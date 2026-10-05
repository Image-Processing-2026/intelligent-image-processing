import os

import cv2
import numpy as np
import pytest

from src.agent.executor import execute_plan
from src.agent.state import RegionOperation, TreatmentPlan
from src.analyzer_evaluator.analyzer import analyze_image

DATA_DIR = os.path.join("data", "real")


@pytest.fixture
def dark_img():
    img = cv2.imread(os.path.join(DATA_DIR, "coffee_underexposed.jpg"))
    return (
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is not None
        else np.zeros((100, 100, 3), dtype=np.uint8)
    )


@pytest.fixture
def noisy_img():
    img = cv2.imread(os.path.join(DATA_DIR, "astro_noisy.jpg"))
    return (
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is not None
        else np.zeros((100, 100, 3), dtype=np.uint8)
    )


def test_execute_multi_action(dark_img):
    plan = TreatmentPlan(
        iteration=1,
        reasoning="Test multi action",
        actions=[
            RegionOperation(
                region_id="full_image",
                target_prompt="full",
                detected_issue="noise",
                operation="denoise",
                parameters={"method": "bilateral", "strength": 1.0},
            ),
            RegionOperation(
                region_id="full_image",
                target_prompt="full",
                detected_issue="dark",
                operation="gamma_correct",
                parameters={"gamma": 1.5},
            ),
            RegionOperation(
                region_id="full_image",
                target_prompt="full",
                detected_issue="contrast",
                operation="clahe",
                parameters={"clip_limit": 2.0},
            ),
        ],
    )
    out = execute_plan(dark_img, plan)
    out_metrics = analyze_image(out)
    in_metrics = analyze_image(dark_img)
    assert out_metrics.brightness_mean > in_metrics.brightness_mean


def test_execute_invalid_op_robustness(noisy_img):
    plan = TreatmentPlan(
        iteration=1,
        reasoning="Test invalid op",
        actions=[
            RegionOperation(
                region_id="full_image",
                target_prompt="full",
                detected_issue="blur",
                operation="super_resolve",
                parameters={},
            ),
            RegionOperation(
                region_id="full_image",
                target_prompt="full",
                detected_issue="noise",
                operation="denoise",
                parameters={},
            ),
        ],
    )
    out = execute_plan(noisy_img, plan)
    assert out.shape == noisy_img.shape


def test_execute_empty_plan(noisy_img):
    plan = TreatmentPlan(iteration=1, reasoning="Empty", actions=[])
    out = execute_plan(noisy_img, plan)
    assert np.array_equal(out, noisy_img)


def test_execute_denoise_then_sharpen(noisy_img):
    plan = TreatmentPlan(
        iteration=1,
        reasoning="Denoise then sharpen",
        actions=[
            RegionOperation(
                region_id="full_image",
                target_prompt="full",
                detected_issue="noise",
                operation="denoise",
                parameters={"method": "nlm", "strength": 1.0},
            ),
            RegionOperation(
                region_id="full_image",
                target_prompt="full",
                detected_issue="blur",
                operation="sharpen",
                parameters={"method": "unsharp_mask", "amount": 0.8},
            ),
        ],
    )
    out_full = execute_plan(noisy_img, plan)
    full_metrics = analyze_image(out_full)
    assert full_metrics.sharpness_laplacian_var is not None
