"""Integration checks for face masks, Module 3 blending, and the executor caller."""

from __future__ import annotations

import numpy as np

from src.agent.executor import execute_plan
from src.agent.state import RegionOperation, TreatmentPlan
from src.region_engine.face_detector import FaceDetectionRecord, detect_faces
from src.region_engine.mask_utils import blend_regions


def test_detected_face_masks_merge_once_before_region_blending(monkeypatch) -> None:
    import src.region_engine.face_detector as face_detector

    class Backend:
        def detect(self, _image):
            return [
                FaceDetectionRecord(1, 1, 3, 2, 0.9),
                FaceDetectionRecord(3, 2, 3, 2, 0.8),
            ]

        def close(self):
            pass

    backend = Backend()
    face_detector.reset_face_detector()
    monkeypatch.setattr(face_detector, "_DETECTOR_FACTORY", lambda _path: backend)
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    processed = np.full_like(image, [200, 100, 50])

    masks = detect_faces(image, feather_radius=0, expand_ratio=0)
    merged = np.maximum.reduce(masks)
    actual = blend_regions(image, processed, merged)

    expected = np.zeros_like(image)
    expected[1:3, 1:4] = [200, 100, 50]
    expected[2:4, 3:6] = [200, 100, 50]
    np.testing.assert_array_equal(actual, expected)
    face_detector.reset_face_detector()


def _plan(*actions: RegionOperation) -> TreatmentPlan:
    return TreatmentPlan(reasoning="test", actions=list(actions))


def _action(target: str, order: int = 0) -> RegionOperation:
    return RegionOperation(
        region_id=target,
        target_prompt=target,
        detected_issue="test",
        operation="gamma_correct",
        parameters={"gamma": 1.2},
        order=order,
    )


def test_executor_skips_face_action_without_falling_back_to_full_image(monkeypatch) -> None:
    import src.agent.executor as executor

    calls: list[np.ndarray | None] = []

    def fake_gamma(image, mask=None, gamma=1.2):
        calls.append(mask)
        return image + 1

    monkeypatch.setattr(executor, "detect_faces", lambda _image: [])
    monkeypatch.setattr(executor, "apply_gamma", fake_gamma)
    image = np.zeros((3, 4, 3), dtype=np.uint8)

    result = execute_plan(image, _plan(_action("face"), _action("full", order=1)))

    assert len(calls) == 1
    assert calls[0] is None
    np.testing.assert_array_equal(result, np.ones_like(image))


def test_executor_merges_multiple_face_masks_before_operation(monkeypatch) -> None:
    import src.agent.executor as executor

    first = np.zeros((3, 4), dtype=np.float32)
    first[0, 0] = 1.0
    second = np.zeros((3, 4), dtype=np.float32)
    second[1, 1] = 0.75
    seen: list[np.ndarray | None] = []

    def fake_gamma(image, mask=None, gamma=1.2):
        seen.append(mask)
        return image

    monkeypatch.setattr(executor, "detect_faces", lambda _image: [first, second])
    monkeypatch.setattr(executor, "apply_gamma", fake_gamma)
    image = np.zeros((3, 4, 3), dtype=np.uint8)

    execute_plan(image, _plan(_action("face")))

    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], np.maximum(first, second))


def test_executor_propagates_face_detector_errors(monkeypatch) -> None:
    import src.agent.executor as executor

    error = RuntimeError("detector unavailable")
    monkeypatch.setattr(executor, "detect_faces", lambda _image: (_ for _ in ()).throw(error))
    image = np.zeros((3, 4, 3), dtype=np.uint8)

    try:
        execute_plan(image, _plan(_action("face")))
    except RuntimeError as raised:
        assert raised is error
    else:
        raise AssertionError("the detector error was swallowed")
