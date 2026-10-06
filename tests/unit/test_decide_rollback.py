"""
Kiểm thử cơ chế rollback của decide_node khi một vòng xử lý làm suy thoái chất lượng.
Chạy offline, không cần API key.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from src.agent.graph import PSNR_DEGRADATION, REAL_MIN_GAIN, REAL_TARGET_SCORE, decide_node
from src.agent.state import (
    Defect,
    DiagnosisReport,
    DoctorState,
    HistoryItem,
    RegionOperation,
    TreatmentPlan,
)

CURRENT = np.full((16, 16, 3), 200, dtype=np.uint8)
PREVIOUS = np.full((16, 16, 3), 100, dtype=np.uint8)


def _plan(iteration: int) -> TreatmentPlan:
    return TreatmentPlan(
        iteration=iteration,
        reasoning="work",
        actions=[
            RegionOperation(
                region_id="1", target_prompt="full", detected_issue="blur", operation="sharpen"
            )
        ],
    )


def _history(eval_score: Dict[str, Any]) -> List[HistoryItem]:
    return [
        HistoryItem(
            iteration=1,
            plan=_plan(1),
            metrics_before={},
            metrics_after={},
            eval_score=eval_score,
            decision="RE_PROCESS",
        )
    ]


def _state(
    *,
    is_synthetic: bool,
    eval_result: Dict[str, Any],
    history: Optional[List[HistoryItem]] = None,
    iteration: int = 2,
    max_iterations: int = 5,
) -> DoctorState:
    return {
        "original_image": PREVIOUS,
        "current_image": CURRENT,
        "previous_image": PREVIOUS,
        "ground_truth_image": PREVIOUS if is_synthetic else None,
        "is_synthetic": is_synthetic,
        "iteration": iteration,
        "max_iterations": max_iterations,
        "technical_metrics": {},
        "treatment_plan": _plan(iteration),
        "evaluation_result": eval_result,
        "history": history or [],
        "intermediate_images": [],
        "decision": "PENDING",
        "error_message": None,
    }


def test_real_quality_not_improved_rolls_back(caplog):
    """(a) Ảnh thực: quality_improved=False → trả về previous_image."""
    state = _state(is_synthetic=False, eval_result={"quality_improved": False})
    with caplog.at_level(logging.WARNING, logger="src.agent.graph"):
        res = decide_node(state)
    assert res["decision"] == "STOP_BEST_EFFORT"
    assert np.array_equal(res["current_image"], PREVIOUS)
    assert res["rolled_back"] is True
    assert res["history"][-1].rolled_back is True
    assert "rolling back" in caplog.text
    # Thumbnail vẫn là ảnh thực tế được tạo ra ở vòng này (bằng chứng)
    assert np.array_equal(res["intermediate_images"][-1], CURRENT)


def test_synthetic_psnr_drop_rolls_back():
    """(b) Ảnh synthetic: PSNR giảm quá PSNR_DEGRADATION → rollback."""
    history = _history({"psnr": 25.0, "ssim": 0.8})
    eval_result = {"psnr": 25.0 - PSNR_DEGRADATION - 1.0, "ssim": 0.7}
    res = decide_node(_state(is_synthetic=True, eval_result=eval_result, history=history))
    assert res["decision"] == "STOP_BEST_EFFORT"
    assert np.array_equal(res["current_image"], PREVIOUS)
    assert res["rolled_back"] is True


def test_degradation_on_final_iteration_rolls_back():
    """(c) Suy thoái ở vòng cuối (iteration == max_iterations) vẫn phải rollback."""
    history = _history({"psnr": 25.0, "ssim": 0.8})
    syn = _state(
        is_synthetic=True,
        eval_result={"psnr": 20.0, "ssim": 0.7},
        history=history,
        iteration=3,
        max_iterations=3,
    )
    res_syn = decide_node(syn)
    assert res_syn["decision"] == "STOP_BEST_EFFORT"
    assert np.array_equal(res_syn["current_image"], PREVIOUS)
    assert res_syn["rolled_back"] is True

    real = _state(
        is_synthetic=False,
        eval_result={"quality_improved": False},
        iteration=3,
        max_iterations=3,
    )
    res_real = decide_node(real)
    assert np.array_equal(res_real["current_image"], PREVIOUS)
    assert res_real["rolled_back"] is True


def test_max_iterations_without_degradation_keeps_current():
    """(d) Hết vòng lặp nhưng không suy thoái → giữ ảnh hiện tại, không rollback."""
    history = _history({"psnr": 20.0, "ssim": 0.7})
    syn = _state(
        is_synthetic=True,
        eval_result={"psnr": 22.0, "ssim": 0.75},
        history=history,
        iteration=3,
        max_iterations=3,
    )
    res_syn = decide_node(syn)
    assert res_syn["decision"] == "STOP_BEST_EFFORT"
    assert np.array_equal(res_syn["current_image"], CURRENT)
    assert res_syn["rolled_back"] is False
    assert res_syn["history"][-1].rolled_back is False

    real = _state(
        is_synthetic=False,
        eval_result={"quality_improved": True, "estimated_quality_score": 60.0},
        iteration=3,
        max_iterations=3,
    )
    res_real = decide_node(real)
    assert res_real["decision"] == "STOP_BEST_EFFORT"
    assert np.array_equal(res_real["current_image"], CURRENT)
    assert res_real["rolled_back"] is False


# ---------------------------------------------------------
# Phase 0: tiêu chí SHIP / bão hòa cho ảnh thực và thứ tự xét giới hạn vòng
# ---------------------------------------------------------
def test_real_image_ships_when_target_score_reached():
    """Ảnh thực đạt REAL_TARGET_SCORE → SHIP sớm, không xử lý thêm."""
    eval_result = {"quality_improved": True, "estimated_quality_score": REAL_TARGET_SCORE}
    res = decide_node(_state(is_synthetic=False, eval_result=eval_result, iteration=1))
    assert res["decision"] == "SHIP"
    assert res["rolled_back"] is False


def test_real_image_stops_when_gain_plateaus():
    """Điểm tăng chưa tới REAL_MIN_GAIN → bão hòa: STOP_BEST_EFFORT, giữ ảnh hiện tại."""
    history = _history({"estimated_quality_score": 70.0})
    eval_result = {
        "quality_improved": True,
        "estimated_quality_score": 70.0 + REAL_MIN_GAIN / 2,
    }
    res = decide_node(_state(is_synthetic=False, eval_result=eval_result, history=history))
    assert res["decision"] == "STOP_BEST_EFFORT"
    assert np.array_equal(res["current_image"], CURRENT)
    assert res["rolled_back"] is False


def test_real_image_continues_on_meaningful_gain():
    history = _history({"estimated_quality_score": 70.0})
    eval_result = {
        "quality_improved": True,
        "estimated_quality_score": 70.0 + REAL_MIN_GAIN + 1.0,
    }
    res = decide_node(_state(is_synthetic=False, eval_result=eval_result, history=history))
    assert res["decision"] == "RE_PROCESS"


def test_target_reached_on_final_iteration_ships():
    """Vòng cuối đạt mục tiêu → SHIP (trước đây bị báo STOP_BEST_EFFORT do xét giới hạn trước)."""
    syn = _state(
        is_synthetic=True,
        eval_result={"psnr": 30.0, "ssim": 0.9},
        iteration=3,
        max_iterations=3,
    )
    assert decide_node(syn)["decision"] == "SHIP"

    real = _state(
        is_synthetic=False,
        eval_result={"quality_improved": True, "estimated_quality_score": 90.0},
        iteration=3,
        max_iterations=3,
    )
    assert decide_node(real)["decision"] == "SHIP"


def test_empty_plan_on_final_iteration_ships():
    """Vòng cuối VLM trả plan rỗng (ảnh đã tốt) → SHIP, không phải STOP_BEST_EFFORT."""
    state = _state(
        is_synthetic=False,
        eval_result={"estimated_quality_score": 60.0},
        iteration=3,
        max_iterations=3,
    )
    state["treatment_plan"] = TreatmentPlan(iteration=3, reasoning="good", actions=[])
    assert decide_node(state)["decision"] == "SHIP"


# ---------------------------------------------------------
# Phase 1: SHIP ảnh thực cần chẩn đoán xác nhận lỗi rõ (severity >= 2) đã hết
# ---------------------------------------------------------
def _with_diagnosis(state: DoctorState, severity: int) -> DoctorState:
    state["diagnosis"] = DiagnosisReport(
        defects=[Defect(type="backlit_subject", region="face", severity=severity)]
    )
    return state


def test_target_score_after_major_defect_requires_verification():
    """Điểm toàn cục đạt nhưng vòng này xử lý lỗi rõ → RE_PROCESS để Perceive lại."""
    eval_result = {"quality_improved": True, "estimated_quality_score": 95.0}
    state = _with_diagnosis(_state(is_synthetic=False, eval_result=eval_result), severity=2)
    res = decide_node(state)
    assert res["decision"] == "RE_PROCESS"
    assert res["history"][-1].diagnosis.defects[0].type == "backlit_subject"


def test_minor_defects_ship_on_target_score():
    eval_result = {"quality_improved": True, "estimated_quality_score": 95.0}
    state = _with_diagnosis(_state(is_synthetic=False, eval_result=eval_result), severity=1)
    assert decide_node(state)["decision"] == "SHIP"


def test_verification_on_final_iteration_stops_best_effort():
    eval_result = {"quality_improved": True, "estimated_quality_score": 95.0}
    state = _with_diagnosis(
        _state(is_synthetic=False, eval_result=eval_result, iteration=3, max_iterations=3),
        severity=3,
    )
    assert decide_node(state)["decision"] == "STOP_BEST_EFFORT"


def test_synthetic_target_ignores_verification():
    """Synthetic có ground truth: PSNR/SSIM là thước đo khách quan, không cần xác nhận lại."""
    state = _with_diagnosis(
        _state(is_synthetic=True, eval_result={"psnr": 30.0, "ssim": 0.9}), severity=3
    )
    assert decide_node(state)["decision"] == "SHIP"


def test_iteration_that_changed_no_pixels_stops_without_rollback(caplog):
    """Mọi action bị bỏ qua (ảnh không đổi) → không phải suy thoái, dừng thay vì lặp vô ích."""
    state = _state(is_synthetic=False, eval_result={"quality_improved": False})
    state["current_image"] = PREVIOUS.copy()
    with caplog.at_level(logging.WARNING, logger="src.agent.graph"):
        res = decide_node(state)
    assert res["decision"] == "STOP_BEST_EFFORT"
    assert res["rolled_back"] is False
    assert "changed no pixels" in caplog.text
