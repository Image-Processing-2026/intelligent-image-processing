"""
Kiểm thử reviewer (LLM) tham gia quyết định của decide_node cho ảnh thực: ACCEPT → SHIP,
REVISE → RE_PROCESS, REJECT → rollback; không có kết luận thì dùng luật đo lường như cũ.
Chạy offline: `_judge_iteration` được thay bằng kết quả giả.
"""

from typing import Any, Dict, Optional
from unittest.mock import patch

import numpy as np
import pytest

from src.agent import graph
from src.agent.graph import decide_node, review_node
from src.agent.review import image_sha256
from src.agent.state import RegionOperation, ReviewResult, SideEffectCheck, TreatmentPlan

ORIGINAL = np.full((32, 32, 3), 40, dtype=np.uint8)
EARLIER = np.full((32, 32, 3), 60, dtype=np.uint8)
EDITED = np.full((32, 32, 3), 90, dtype=np.uint8)


def _plan() -> TreatmentPlan:
    return TreatmentPlan(
        iteration=1,
        reasoning="denoise",
        actions=[
            RegionOperation(
                region_id="full",
                target_prompt="full",
                detected_issue="noise",
                operation="denoise",
                parameters={"strength": 1.5},
            )
        ],
    )


def _state(quality_improved: bool = True, iteration: int = 1, **overrides: Any) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "original_image": ORIGINAL,
        "current_image": EDITED,
        "previous_image": EARLIER,
        "ground_truth_image": None,
        "is_synthetic": False,
        "iteration": iteration,
        "max_iterations": 2,
        "technical_metrics": {},
        "treatment_plan": _plan(),
        "evaluation_result": {
            "estimated_quality_score": 93.0,
            "quality_improved": quality_improved,
            "technical_metrics": {},
        },
        "history": [],
        "intermediate_images": [],
        "decision": "INITIALIZING",
        "rolled_back": False,
        "intent": None,
    }
    state.update(overrides)
    return state


def _verdict(verdict: Optional[str], status: str = "completed") -> ReviewResult:
    return ReviewResult(
        status=status,
        verdict=verdict,
        candidate_sha256=image_sha256(EDITED),
    )


@pytest.fixture
def judged(monkeypatch):
    """Cho decide_node chạy với reviewer giả; không đo lỗi/tác dụng phụ thật."""
    monkeypatch.setattr(graph, "_track_defects", lambda state, plan: [])
    monkeypatch.setattr(graph, "_side_effects_check", lambda state: SideEffectCheck())
    monkeypatch.setattr(graph, "_evaluate_returned", lambda state, image, iteration: {})

    def install(result: Optional[ReviewResult]):
        monkeypatch.setattr(graph, "_judge_iteration", lambda state, check: result)

    return install


class TestLlmJudgesTheIteration:
    def test_without_a_verdict_rules_still_roll_back(self, judged):
        judged(None)
        update = decide_node(_state(quality_improved=False))
        assert update["rolled_back"] is True
        assert np.array_equal(update["current_image"], EARLIER)

    def test_accept_overrides_a_soft_rule_regression_and_ships(self, judged):
        # Ca noisy_lena: luật phạt khử nhiễu mạnh (quality_improved=False) nhưng reviewer thấy tốt
        judged(_verdict("ACCEPT"))
        update = decide_node(_state(quality_improved=False))
        assert update["rolled_back"] is False
        assert update["decision"] == "SHIP"
        assert np.array_equal(update["current_image"], EDITED)
        assert update["history"][-1].review.verdict == "ACCEPT"

    def test_revise_keeps_the_image_and_goes_another_round(self, judged):
        judged(_verdict("REVISE"))
        update = decide_node(_state(quality_improved=False))
        assert update["rolled_back"] is False
        assert update["decision"] == "RE_PROCESS"

    def test_revise_on_the_last_round_stops_best_effort(self, judged):
        judged(_verdict("REVISE"))
        update = decide_node(_state(iteration=2))
        assert update["decision"] == "STOP_BEST_EFFORT"
        assert update["rolled_back"] is False

    def test_reject_rolls_back_even_when_the_score_is_good(self, judged):
        judged(_verdict("REJECT"))
        update = decide_node(_state(quality_improved=True))
        assert update["rolled_back"] is True
        assert update["decision"] == "STOP_BEST_EFFORT"
        assert np.array_equal(update["current_image"], EARLIER)

    @pytest.mark.parametrize("result", [_verdict("UNCERTAIN"), _verdict(None, "unavailable")])
    def test_no_conclusion_falls_back_to_the_rules(self, judged, result):
        judged(result)
        assert decide_node(_state(quality_improved=False))["rolled_back"] is True

    def test_synthetic_runs_are_not_judged(self, judged, monkeypatch):
        calls = []
        monkeypatch.setattr(
            graph, "_judge_iteration", lambda state, check: calls.append(1) or _verdict("REJECT")
        )
        decide_node(_state(is_synthetic=True, ground_truth_image=ORIGINAL))
        assert calls == []


class TestFinalReviewIsReused:
    def test_review_of_the_same_image_is_not_requested_again(self):
        review = _verdict("ACCEPT")
        with patch.object(graph, "review_candidate") as call:
            update = review_node(_state(review=review))
        call.assert_not_called()
        assert update["review"] is review

    def test_a_review_of_another_image_is_redone(self):
        stale = ReviewResult(status="completed", verdict="ACCEPT", candidate_sha256="other")
        fresh = _verdict("ACCEPT")
        with patch.object(graph, "review_candidate", return_value=fresh) as call:
            update = review_node(_state(review=stale))
        call.assert_called_once()
        assert update["review"] is fresh
