"""
Kiểm thử cơ chế rollback của decide_node khi một vòng xử lý làm suy thoái chất lượng.
Chạy offline, không cần API key.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from src.agent.graph import PSNR_DEGRADATION, decide_node
from src.agent.state import DoctorState, HistoryItem, RegionOperation, TreatmentPlan

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
