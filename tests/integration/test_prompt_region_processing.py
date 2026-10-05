"""Integration checks for semantic masks, Module 3 blending, and executor flow."""

from __future__ import annotations

import numpy as np

from src.agent.executor import execute_plan
from src.agent.state import RegionOperation, TreatmentPlan
from src.region_engine.detector import PromptSegmentationResult
from src.region_engine.mask_utils import blend_regions


def _action(target: str, order: int = 0) -> RegionOperation:
    return RegionOperation(
        region_id=target,
        target_prompt=target,
        detected_issue="test",
        operation="gamma_correct",
        parameters={"gamma": 1.2},
        order=order,
    )


def _plan(*actions: RegionOperation) -> TreatmentPlan:
    return TreatmentPlan(reasoning="test", actions=list(actions))


def test_semantic_union_mask_blends_module_3_once() -> None:
    original = np.zeros((4, 6, 3), dtype=np.uint8)
    processed = np.full_like(original, [200, 100, 50])
    union = np.zeros((4, 6), dtype=np.float32)
    union[0:2, 1:3] = 1.0
    union[1:3, 2:5] = 1.0

    actual = blend_regions(original, processed, union)

    expected = np.zeros_like(original)
    expected[0, 1:3] = [200, 100, 50]
    expected[1, 1:5] = [200, 100, 50]
    expected[2, 2:5] = [200, 100, 50]
    np.testing.assert_array_equal(actual, expected)


def test_executor_skips_zero_semantic_mask_without_full_image_fallback(monkeypatch) -> None:
    import src.agent.executor as executor
    import src.region_engine.controller as controller

    calls: list[np.ndarray | None] = []

    def no_detections(_image, prompt, *, config=None):
        # Suy luận thành công nhưng không tìm thấy đối tượng nào → mask rỗng
        return PromptSegmentationResult(prompt, f"{prompt}.", (), ())

    def fake_gamma(image, mask=None, gamma=1.2):
        calls.append(mask)
        return image + 1

    monkeypatch.setattr(controller, "resolve_prompt_instances", no_detections)
    monkeypatch.setattr(executor, "apply_gamma", fake_gamma)
    image = np.zeros((3, 4, 3), dtype=np.uint8)

    actual = execute_plan(image, _plan(_action("person"), _action("full", order=1)))

    assert len(calls) == 1
    assert calls[0] is None
    np.testing.assert_array_equal(actual, np.ones_like(image))


def test_executor_skips_unknown_prompt_when_semantic_backend_fails(monkeypatch) -> None:
    import src.region_engine.controller as controller
    from src.region_engine.segmentation_backend import SegmentationUnavailableError

    def failing_segmenter(*_args, **_kwargs):
        raise SegmentationUnavailableError("semantic backend unavailable")

    monkeypatch.setattr(controller, "resolve_prompt_instances", failing_segmenter)
    image = np.zeros((3, 4, 3), dtype=np.uint8)

    # Prompt không có heuristic (quyết định D1) → bỏ qua, không xử lý toàn ảnh
    result = execute_plan(image, _plan(_action("person")))

    np.testing.assert_array_equal(result, image)
