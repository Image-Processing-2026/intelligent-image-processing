"""
Kiểm thử quyết định theo lỗi đã chẩn đoán (ảnh thực) của decide_node: thành công là lỗi được
sửa trên đúng vùng của nó, không phải điểm toàn ảnh tăng. Chạy offline, không cần model.
"""

import logging
from typing import Any, Dict, Optional

import cv2
import numpy as np
import pytest

from src.agent import graph
from src.agent.executor import execute_plan, image_cache_key
from src.agent.graph import _decide_tracked, decide_node
from src.agent.state import (
    Defect,
    DiagnosisReport,
    DoctorState,
    RegionOperation,
    TreatmentPlan,
)
from src.analyzer_evaluator.measurement.diff import DefectChange

HEIGHT, WIDTH = 256, 384
# Hình chữ nhật "chủ thể" (y0, y1, x0, x1) trên nền sáng
SUBJECT = (70, 200, 130, 260)
BOX_2D = [
    round(SUBJECT[0] / HEIGHT * 1000),
    round(SUBJECT[2] / WIDTH * 1000),
    round(SUBJECT[1] / HEIGHT * 1000),
    round(SUBJECT[3] / WIDTH * 1000),
]


def _backlit_image() -> np.ndarray:
    """
    Nền sáng (L* ≈ 61), chủ thể tối (L* ≈ 18), độ xám trung bình toàn ảnh ≈ 130. Biên chủ thể
    được làm mờ để gamma theo vùng không tạo halo giả trên cạnh cứng.
    """
    rng = np.random.default_rng(0)
    y0, y1, x0, x1 = SUBJECT
    alpha = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    alpha[y0:y1, x0:x1] = 1.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), 10.0)[..., None]
    background = np.array([148.0, 150.0, 155.0], dtype=np.float32)
    subject = np.array([45.0, 43.0, 40.0], dtype=np.float32)
    image = background * (1.0 - alpha) + subject * alpha
    image += rng.normal(0.0, 2.0, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def _diagnosis(severity: int = 3) -> DiagnosisReport:
    return DiagnosisReport(
        scene_type="portrait",
        subjects=["person"],
        subject_boxes={"person": BOX_2D},
        defects=[
            Defect(type="backlit_subject", region="person", severity=severity, evidence="test")
        ],
    )


def _plan(gamma: float) -> TreatmentPlan:
    return TreatmentPlan(
        iteration=1,
        reasoning="light the subject",
        actions=[
            RegionOperation(
                region_id="1",
                target_prompt="person",
                detected_issue="backlit_subject",
                operation="gamma_correct",
                parameters={"gamma": gamma},
            )
        ],
    )


def _state(
    gamma: float,
    *,
    eval_result: Optional[Dict[str, Any]] = None,
    iteration: int = 1,
    max_iterations: int = 3,
) -> DoctorState:
    original = _backlit_image()
    diagnosis = _diagnosis()
    plan = _plan(gamma)
    processed = execute_plan(
        original,
        plan,
        subject_boxes=diagnosis.subject_boxes,
        cache_key=image_cache_key(original),
    )
    assert not np.array_equal(processed, original)
    return {
        "original_image": original,
        "current_image": processed,
        "previous_image": original,
        "is_synthetic": False,
        "iteration": iteration,
        "max_iterations": max_iterations,
        "technical_metrics": {},
        "diagnosis": diagnosis,
        "treatment_plan": plan,
        # Điểm toàn ảnh xấu đi (ca ngược sáng thật): trước đây đủ để rollback
        "evaluation_result": eval_result
        if eval_result is not None
        else {"quality_improved": False},
        "history": [],
        "intermediate_images": [],
        "decision": "PENDING",
        "error_message": None,
    }


def test_brightening_subject_is_not_rolled_back_despite_global_score():
    """Chủ thể sáng lên đúng vùng → không rollback dù quality_improved=False."""
    result = decide_node(_state(gamma=1.6))
    assert result["rolled_back"] is False
    assert result["decision"] in {"SHIP", "RE_PROCESS"}
    entry = result["history"][-1]
    assert entry.defect_status
    change = entry.defect_status[0]
    assert (change["type"], change["region"]) == ("backlit_subject", "person")
    assert change["status"] in {"resolved", "improved"}
    assert isinstance(change, dict)


def test_darkening_subject_rolls_back():
    """Chủ thể tối thêm → lỗi mục tiêu worse → rollback về ảnh trước."""
    state = _state(gamma=0.6, eval_result={})
    result = decide_node(state)
    assert result["rolled_back"] is True
    assert result["decision"] == "STOP_BEST_EFFORT"
    assert np.array_equal(result["current_image"], state["previous_image"])
    assert result["history"][-1].defect_status[0]["status"] == "worse"


def test_unresolved_improvement_on_last_iteration_keeps_image():
    """Lỗi improved mà hết vòng lặp → STOP_BEST_EFFORT nhưng không rollback."""
    state = _state(gamma=1.6, iteration=3, max_iterations=3)
    changes = [
        DefectChange(type="backlit_subject", region="person", before=3, after=2, status="improved")
    ]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(graph, "_track_defects", lambda *_: changes)
        result = decide_node(state)
    assert result["decision"] == "STOP_BEST_EFFORT"
    assert result["rolled_back"] is False
    assert np.array_equal(result["current_image"], state["current_image"])


def test_tracking_failure_falls_back_to_legacy_logic(monkeypatch, caplog):
    """compare() lỗi → logic cũ: quality_improved=False vẫn rollback."""

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("compare failed")

    monkeypatch.setattr(graph, "compare", boom)
    state = _state(gamma=1.6)
    with caplog.at_level(logging.WARNING, logger="src.agent.graph"):
        result = decide_node(state)
    assert "Defect tracking failed" in caplog.text
    assert result["history"][-1].defect_status == []
    assert result["decision"] == "STOP_BEST_EFFORT"
    assert result["rolled_back"] is True


def test_synthetic_path_is_untouched(monkeypatch):
    """Ảnh synthetic không theo dõi lỗi."""
    monkeypatch.setattr(graph, "_track_defects", lambda *_: pytest.fail("must not track"))
    state = _state(gamma=1.6, eval_result={"psnr": 30.0, "ssim": 0.9})
    state["is_synthetic"] = True
    state["ground_truth_image"] = state["original_image"]
    result = decide_node(state)
    assert result["decision"] == "SHIP"
    assert result["history"][-1].defect_status == []


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (["resolved"], "SHIP"),
        (["resolved", "resolved"], "SHIP"),
        (["improved"], "RE_PROCESS"),
        (["resolved", "unchanged"], "RE_PROCESS"),
        (["unchanged"], "STOP_BEST_EFFORT"),
        (["unchanged", "unchanged"], "STOP_BEST_EFFORT"),
    ],
)
def test_decide_tracked_table(statuses, expected):
    changes = [
        DefectChange(type="noise", region="full", before=2, after=0, status=status)
        for status in statuses
    ]
    assert _decide_tracked(changes) == expected
