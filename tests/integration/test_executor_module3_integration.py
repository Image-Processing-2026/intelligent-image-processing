"""
Integration test: xác nhận Module 4's execute_plan() gọi đúng
các hàm Module 3 (apply_denoise, apply_gamma, ...) với format
tham số khớp nhau, và pipeline không vỡ khi ghép Agent <-> Processing Engine.
"""
import numpy as np

from src.agent.executor import execute_plan
from src.agent.state import RegionOperation, TreatmentPlan


def _make_test_image():
    np.random.seed(42)
    h, w = 300, 400
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[: h // 2, :] = 200 + np.random.randint(-10, 10, (h // 2, w, 3))
    img[h // 2 :, :] = 35 + np.random.randint(-8, 8, (h // 2, w, 3))
    return np.clip(img, 0, 255).astype(np.uint8), h


def test_execute_plan_denoise_then_gamma_on_region():
    img, h = _make_test_image()

    plan = TreatmentPlan(
        iteration=1,
        reasoning="Vùng ground bị thiếu sáng và nhiễu nhẹ.",
        actions=[
            RegionOperation(
                region_id="ground_area",
                target_prompt="ground",
                region_type="semantic",
                detected_issue="noise",
                operation="denoise",
                parameters={"method": "bilateral", "strength": 1.0},
                order=10,
            ),
            RegionOperation(
                region_id="ground_area",
                target_prompt="ground",
                region_type="semantic",
                detected_issue="underexposed",
                operation="gamma_correct",
                parameters={"gamma": 2.2},
                order=20,
            ),
        ],
    )

    result = execute_plan(img, plan)

    assert result.shape == img.shape
    assert result.dtype == img.dtype
    assert result[h // 2 :].mean() > img[h // 2 :].mean() + 30
    assert abs(result[: h // 2].mean() - img[: h // 2].mean()) < 5