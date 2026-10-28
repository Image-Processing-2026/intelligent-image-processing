"""
Test truy vết mask và thực thi vùng (R-02, R-03, R-04): feather khai báo = feather thực thi,
báo cáo ExecutedAction đủ trường, mặt oval/QA/fallback giảm cường độ, overlay khớp mask_hash.
Offline: face detector và landmarker được patch.
"""

from unittest.mock import patch

import cv2
import numpy as np
import pytest

from src.agent import executor
from src.agent.executor import (
    FALLBACK_STRENGTH_SCALE,
    execute_plan,
    face_mask_qa,
    mask_fingerprint,
)
from src.agent.region_overlay import executed_region_overlay
from src.agent.state import ExecutedAction, RegionOperation, TreatmentPlan
from src.region_engine.face_landmarker import FaceLandmarkerUnavailableError, FaceOval
from src.region_engine.subject import clear_subject_mask_cache, resolve_subject_mask

_SIZE = 96
_BOX = (24, 24, 72, 72)  # x0, y0, x1, y1 của khuôn mặt giả


@pytest.fixture(autouse=True)
def _clean_caches():
    clear_subject_mask_cache()
    yield
    clear_subject_mask_cache()


@pytest.fixture
def dark_image() -> np.ndarray:
    return np.full((_SIZE, _SIZE, 3), 60, dtype=np.uint8)


def _box_mask() -> np.ndarray:
    mask = np.zeros((_SIZE, _SIZE), dtype=np.float32)
    x0, y0, x1, y1 = _BOX
    mask[y0:y1, x0:x1] = 1.0
    return mask


def _oval(center: tuple[int, int] = (48, 48), axes: tuple[int, int] = (18, 22)) -> FaceOval:
    hard = np.zeros((_SIZE, _SIZE), dtype=np.uint8)
    cv2.ellipse(hard, center, axes, 0, 0, 360, 1, -1)
    contour = np.array(
        [
            [center[0] + axes[0] * np.cos(t), center[1] + axes[1] * np.sin(t)]
            for t in range(0, 360, 20)
        ],
        dtype=np.float32,
    )
    return FaceOval(mask=hard.astype(np.float32), contour=contour)


def _face_plan(operation: str = "gamma_correct", **overrides) -> TreatmentPlan:
    action = RegionOperation(
        region_id="face",
        target_prompt="face",
        region_type="face",
        detected_issue="underexposed",
        operation=operation,
        parameters={"gamma": 1.8} if operation == "gamma_correct" else {"clip_limit": 3.0},
        **overrides,
    )
    return TreatmentPlan(iteration=1, reasoning="lift face", actions=[action])


def _run(image: np.ndarray, plan: TreatmentPlan, oval=None, oval_error=None):
    report: list[ExecutedAction] = []
    oval_patch = patch(
        "src.region_engine.controller.detect_face_ovals",
        side_effect=oval_error,
        return_value=[] if oval is None else [oval],
    )
    with (
        oval_patch,
        patch("src.agent.executor.detect_faces", return_value=[_box_mask()]),
    ):
        result = execute_plan(image, plan, report=report)
    return result, report


# ---------------- R-03: feather ----------------
def test_subject_feather_reaches_the_executed_mask(dark_image):
    action = RegionOperation(
        region_id="sky",
        target_prompt="sky",
        feather_radius=30,
        detected_issue="x",
        operation="gamma_correct",
    )
    box = (0, 0, 500, 1000)
    with patch(
        "src.agent.executor.resolve_subject_mask", wraps=executor.resolve_subject_mask
    ) as spy:
        resolution = executor._resolve_subject_region(dark_image, action, box, "k")
    assert spy.call_args.kwargs["feather_radius"] == 30
    assert resolution.feather_actual == 30


def test_feather_30_differs_from_15_and_caches_are_separate(dark_image):
    kwargs = {"use_semantic": False, "cache_key": "same-image"}
    mask_15 = resolve_subject_mask(dark_image, "sky", None, feather_radius=15, **kwargs)
    mask_30 = resolve_subject_mask(dark_image, "sky", None, feather_radius=30, **kwargs)
    assert mask_15 is not mask_30
    assert not np.array_equal(mask_15.mask, mask_30.mask)
    # Cùng feather thì dùng lại mục cache; feather mặc định trùng với không khai báo
    assert resolve_subject_mask(dark_image, "sky", None, feather_radius=30, **kwargs) is mask_30
    assert resolve_subject_mask(dark_image, "sky", None, **kwargs) is mask_15


# ---------------- R-02: báo cáo ----------------
def test_report_for_full_region_and_unsupported_operation(dark_image):
    plan = TreatmentPlan(
        iteration=1,
        reasoning="",
        actions=[
            RegionOperation(
                region_id="full",
                target_prompt="full",
                detected_issue="x",
                operation="gamma_correct",
                parameters={"gamma": 1.3},
            ),
            RegionOperation(
                region_id="full", target_prompt="full", detected_issue="x", operation="warp"
            ),
        ],
    )
    report: list[ExecutedAction] = []
    execute_plan(dark_image, plan, report=report)
    ok, bad = report
    assert (ok.backend, ok.skipped, ok.mask_hash, ok.mask_area) == ("geometry", False, None, 1.0)
    assert ok.feather_requested == 15
    assert bad.skipped and bad.skip_reason == "unsupported_operation"


def test_report_for_skipped_action_has_reason(dark_image):
    with patch("src.agent.executor.detect_faces", return_value=[]):
        report: list[ExecutedAction] = []
        result = execute_plan(dark_image, _face_plan(face_mode="bbox"), report=report)
    assert np.array_equal(result, dark_image)
    assert len(report) == 1
    assert report[0].skipped and report[0].skip_reason == "no_face"
    assert report[0].requested_mode == "bbox"


def test_mask_fingerprint_is_stable_and_sensitive():
    mask = _box_mask()
    assert mask_fingerprint(mask) == mask_fingerprint(mask.copy() + 1e-5)
    shifted = np.roll(mask, 3, axis=1)
    assert mask_fingerprint(mask)[0] != mask_fingerprint(shifted)[0]
    assert mask_fingerprint(None) == (None, 1.0)


# ---------------- R-04: oval, QA, fallback ----------------
def test_exposure_on_face_defaults_to_oval_mask_that_is_not_a_rectangle(dark_image):
    result, report = _run(dark_image, _face_plan(), oval=_oval())
    item = report[0]
    assert (item.requested_mode, item.actual_mode) == ("oval", "oval")
    assert item.strength_scale == 1.0 and item.fallback_reason is None
    mask = executor.executed_mask(item.mask_hash)
    assert mask is not None
    box_mask = _box_mask()
    # Góc hộp không nằm trong oval, nên vùng đổi màu không phải hình chữ nhật
    assert mask[25, 25] < 0.05 < mask[48, 48]
    assert not np.array_equal(mask > 0.5, box_mask > 0.5)
    assert result[48, 48].mean() > 60 and result[25, 25].mean() == 60


def test_explicit_bbox_mode_is_respected_for_exposure(dark_image):
    result, report = _run(dark_image, _face_plan(face_mode="bbox"), oval=_oval())
    assert report[0].actual_mode == "bbox" and report[0].strength_scale == 1.0
    assert result[30, 30].mean() > 60  # góc hộp cũng được chỉnh


def test_oval_failure_falls_back_to_bbox_with_reduced_strength(dark_image, caplog):
    full_strength, full_report = _run(dark_image, _face_plan(face_mode="bbox"))
    with caplog.at_level("WARNING", logger="src.agent.executor"):
        reduced, report = _run(
            dark_image, _face_plan(), oval_error=FaceLandmarkerUnavailableError("no model")
        )
    item = report[0]
    assert item.actual_mode == "bbox" and item.strength_scale == FALLBACK_STRENGTH_SCALE
    assert "no model" in (item.fallback_reason or "")
    assert "Falling back to 'bbox'" in caplog.text
    # Trung tính 60 < kết quả giảm < kết quả nguyên cường độ
    assert 60 < reduced[48, 48].mean() < full_strength[48, 48].mean()


def test_qa_failure_is_a_fail_and_does_not_apply_full_strength(dark_image):
    leaking = _oval(center=(20, 20), axes=(18, 22))  # lệch ra ngoài hộp mặt
    full_strength, _ = _run(dark_image, _face_plan(face_mode="bbox"))
    result, report = _run(dark_image, _face_plan(), oval=leaking)
    item = report[0]
    assert item.actual_mode == "bbox"
    assert "qa_outside_box" in (item.fallback_reason or "")
    assert item.strength_scale < 1.0
    assert result[48, 48].mean() < full_strength[48, 48].mean()


def test_face_mask_qa_rules():
    boxes = [_BOX]
    shape = (_SIZE, _SIZE)
    assert face_mask_qa(np.zeros(shape, np.float32), boxes, shape) == "qa_empty"
    assert face_mask_qa(_oval().mask, boxes, shape) is None
    tiny = np.zeros(shape, np.float32)
    tiny[46:50, 46:50] = 1.0
    assert face_mask_qa(tiny, boxes, shape).startswith("qa_too_small")
    assert face_mask_qa(np.ones(shape, np.float32), boxes, shape).startswith("qa_outside_box")
    pieces = np.zeros(shape, np.float32)
    for index in range(6):  # 6 mảng rời trong cùng một hộp
        pieces[26 + index * 7 : 30 + index * 7, 26:70] = 1.0
        pieces[28 + index * 7 : 29 + index * 7, :] *= 1.0
    # các dải cách nhau → nhiều thành phần
    assert face_mask_qa(pieces, boxes, shape).startswith("qa_fragmented")


def test_sam_roi_is_tried_first_for_exposure_when_models_exist(dark_image):
    sam = np.zeros((_SIZE, _SIZE), dtype=np.float32)
    cv2.ellipse(sam, (48, 48), (17, 21), 0, 0, 360, 1.0, -1)
    with (
        patch.object(executor, "subject_capabilities", return_value={"semantic_models": True}),
        patch.object(executor, "segment_box", return_value=sam) as fake_sam,
    ):
        _, report = _run(dark_image, _face_plan(), oval=_oval())
    fake_sam.assert_called_once()
    assert report[0].actual_mode == "sam_refined" and report[0].backend == "face_sam_roi"


def test_sam_failure_falls_through_to_oval(dark_image):
    from src.region_engine.segmentation_backend import SegmentationUnavailableError

    with (
        patch.object(executor, "subject_capabilities", return_value={"semantic_models": True}),
        patch.object(executor, "segment_box", side_effect=SegmentationUnavailableError("x")),
    ):
        _, report = _run(dark_image, _face_plan("clahe"), oval=_oval())
    assert report[0].actual_mode == "oval"
    assert report[0].strength_scale == 1.0


# ---------------- R-04: overlay ----------------
def test_overlay_draws_the_executed_mask(dark_image):
    _, report = _run(dark_image, _face_plan(), oval=_oval())
    overlay, drawn = executed_region_overlay(dark_image, report)
    assert drawn == {"face:gamma_correct": report[0].mask_hash}
    stored = executor.executed_mask(report[0].mask_hash)
    assert mask_fingerprint(stored)[0] == report[0].mask_hash
    assert overlay.shape == dark_image.shape and not np.array_equal(overlay, dark_image)


def test_overlay_skips_actions_without_cached_mask(dark_image):
    ghost = ExecutedAction(region_id="x", operation="clahe", mask_hash="0" * 64)
    skipped = ExecutedAction(region_id="y", operation="clahe", skipped=True)
    overlay, drawn = executed_region_overlay(dark_image, [ghost, skipped])
    assert drawn == {}
    assert np.array_equal(overlay, dark_image)
