"""
Kiểm thử Outcome trung thực (R-01) và cổng review (R-08). Chạy offline: Gemini được mock,
không có API key thật.
"""

import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.agent import review as review_module
from src.agent.graph import decide_node, finalize_node, review_node, run_pipeline
from src.agent.review import (
    REVIEW_RESPONSE_SCHEMA,
    ReviewContext,
    derive_outcome,
    image_sha256,
    parse_review,
    review_candidate,
    select_review_crops,
)
from src.agent.state import (
    DiagnosisReport,
    HistoryItem,
    IntentProfile,
    RegionOperation,
    ReviewResult,
    SideEffectCheck,
    TreatmentPlan,
)

ORIGINAL = np.full((64, 64, 3), 60, dtype=np.uint8)
EDITED = np.full((64, 64, 3), 120, dtype=np.uint8)
EARLIER = np.full((64, 64, 3), 90, dtype=np.uint8)


def _plan(with_action: bool = True) -> TreatmentPlan:
    actions = (
        [
            RegionOperation(
                region_id="1",
                target_prompt="full",
                detected_issue="underexposed",
                operation="gamma_correct",
            )
        ]
        if with_action
        else []
    )
    return TreatmentPlan(iteration=1, reasoning="r", actions=actions)


def _item(with_action: bool = True, rolled_back: bool = False) -> HistoryItem:
    return HistoryItem(
        iteration=1,
        plan=_plan(with_action),
        metrics_before={},
        metrics_after={},
        eval_score={},
        decision="STOP_BEST_EFFORT",
        rolled_back=rolled_back,
    )


def _review_json(verdict: str, findings: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    return {
        "verdict": verdict,
        "goal": "brighten the face",
        "preserve": ["backlight mood"],
        "findings": findings or [],
        "confidence": 0.8,
        "reason": f"reviewer says {verdict}",
    }


FINDING_FULL = {"crop_id": "FULL", "issue": "black_lift", "severity": 2, "text": "hair is grey"}


def _mock_gemini(responses: List[Any]) -> MagicMock:
    """genai giả: mỗi lời gọi trả (hoặc ném) phần tử kế tiếp của `responses`."""
    mock_genai = MagicMock()
    queue = list(responses)

    def generate_content(**kwargs: Any) -> MagicMock:
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        response = MagicMock()
        response.text = json.dumps(item)
        return response

    mock_genai.Client.return_value.models.generate_content.side_effect = generate_content
    return mock_genai


@pytest.fixture
def review_on(monkeypatch: pytest.MonkeyPatch):
    """Bật review, có key giả; trả hàm dựng mock genai."""
    monkeypatch.setattr(review_module, "REVIEW_ENABLED", True)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr("src.agent.vlm_diagnostician.time.sleep", lambda _s: None)

    def install(responses: List[Any]):
        mock_genai = _mock_gemini(responses)
        return mock_genai, patch("src.agent.vlm_diagnostician.genai", mock_genai)

    return install


class _Retryable(Exception):
    code = 503


class TestReviewCandidate:
    @pytest.mark.parametrize(
        ("verdict", "findings"),
        [
            ("ACCEPT", []),
            ("REVISE", [FINDING_FULL]),
            ("REJECT", [FINDING_FULL]),
            ("UNCERTAIN", []),
        ],
    )
    def test_each_verdict_is_returned(self, review_on, verdict, findings):
        mock_genai, patcher = review_on([_review_json(verdict, findings)])
        with patcher:
            result = review_candidate(ORIGINAL, EARLIER, EDITED, ReviewContext())
        assert result.status == "completed"
        assert result.verdict == verdict
        assert result.candidate_sha256 == image_sha256(EDITED)
        assert result.prompt_version == review_module.PROMPT_VERSION
        assert result.latency_ms is not None
        config = mock_genai.Client.return_value.models.generate_content.call_args.kwargs["config"]
        assert config.response_json_schema == REVIEW_RESPONSE_SCHEMA

    def test_disabled_makes_no_call(self, monkeypatch):
        monkeypatch.setattr(review_module, "REVIEW_ENABLED", False)
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        mock_genai = _mock_gemini([])
        with patch("src.agent.vlm_diagnostician.genai", mock_genai):
            result = review_candidate(ORIGINAL, None, EDITED)
        assert result.status == "not_run"
        assert result.verdict is None
        mock_genai.Client.assert_not_called()

    def test_no_key_is_unavailable(self, monkeypatch):
        monkeypatch.setattr(review_module, "REVIEW_ENABLED", True)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        result = review_candidate(ORIGINAL, None, EDITED)
        assert result.status == "unavailable"
        assert result.verdict is None

    def test_transient_error_retries_once_within_budget(self, review_on, monkeypatch):
        # call_gemini_json tự gọi lại lỗi 5xx: thu về một lần để đếm đúng lời gọi của review
        monkeypatch.setattr("src.agent.vlm_diagnostician.GEMINI_MAX_ATTEMPTS", 1)
        mock_genai, patcher = review_on([_Retryable("busy"), _review_json("ACCEPT")])
        with patcher:
            result = review_candidate(ORIGINAL, None, EDITED)
        assert result.verdict == "ACCEPT"
        calls = mock_genai.Client.return_value.models.generate_content.call_count
        assert calls == 2 <= review_module.REVIEW_MAX_CALLS

    def test_retry_budget_is_not_exceeded(self, review_on, monkeypatch):
        monkeypatch.setattr("src.agent.vlm_diagnostician.GEMINI_MAX_ATTEMPTS", 1)
        mock_genai, patcher = review_on([_Retryable("busy")] * 5)
        with patcher:
            result = review_candidate(ORIGINAL, None, EDITED)
        assert result.status == "unavailable"
        assert result.verdict is None
        calls = mock_genai.Client.return_value.models.generate_content.call_count
        assert calls == review_module.REVIEW_MAX_CALLS

    def test_non_transient_error_is_not_retried(self, review_on):
        mock_genai, patcher = review_on([ValueError("bad"), _review_json("ACCEPT")])
        with patcher:
            result = review_candidate(ORIGINAL, None, EDITED)
        assert result.status == "unavailable"
        assert result.verdict is None
        assert mock_genai.Client.return_value.models.generate_content.call_count == 1

    def test_unknown_verdict_is_unavailable(self, review_on):
        _, patcher = review_on([_review_json("PERFECT")])
        with patcher:
            result = review_candidate(ORIGINAL, None, EDITED)
        assert result.status == "unavailable"
        assert result.verdict is None

    def test_images_sent_include_returned_image_and_crops(self, review_on):
        mock_genai, patcher = review_on([_review_json("ACCEPT")])
        with patcher:
            review_candidate(ORIGINAL, EARLIER, EDITED)
        contents = mock_genai.Client.return_value.models.generate_content.call_args.kwargs[
            "contents"
        ]
        labelled = [part for part in contents if isinstance(part, str)]
        assert any("ẢNH SẼ TRẢ" in text for text in labelled)
        assert any(text.startswith("Crop A") for text in labelled)


class TestParseReview:
    def test_revise_without_grounded_finding_becomes_uncertain(self):
        stray = {"crop_id": "Z", "issue": "other", "severity": 1, "text": "?"}
        result, downgrade = parse_review(_review_json("REVISE", [stray]), ["FULL", "A"])
        assert result.verdict == "UNCERTAIN"
        assert downgrade is not None
        assert result.findings == []

    def test_accept_keeps_grounded_findings_only(self):
        stray = {"crop_id": "Z", "issue": "other", "severity": 1, "text": "?"}
        result, downgrade = parse_review(_review_json("ACCEPT", [FINDING_FULL, stray]), ["FULL"])
        assert result.verdict == "ACCEPT"
        assert downgrade is None
        assert [f.crop_id for f in result.findings] == ["FULL"]

    def test_confidence_is_clamped(self):
        data = _review_json("ACCEPT")
        data["confidence"] = 7
        result, _ = parse_review(data, ["FULL"])
        assert result.confidence == 1.0


class TestReviewCrops:
    def test_changed_region_gets_paired_crop(self):
        edited = ORIGINAL.copy()
        edited[10:50, 10:50] = 200
        crops = select_review_crops(ORIGINAL, edited)
        assert crops and crops[0].crop_id == "A"
        x0, y0, x1, y1 = crops[0].box
        assert crops[0].image.shape == (y1 - y0, 2 * (x1 - x0), 3)

    def test_identical_images_have_no_crop(self):
        assert select_review_crops(ORIGINAL, ORIGINAL.copy()) == []

    def test_suspect_crop_only_when_black_lift_flagged(self):
        original = np.full((64, 64, 3), 200, dtype=np.uint8)
        original[:20, :20] = 10
        edited = original.copy()
        edited[:20, :20] = 60
        quiet = select_review_crops(original, edited, SideEffectCheck(status="ok"))
        flagged = select_review_crops(
            original,
            edited,
            SideEffectCheck(status="ok", original=[{"kind": "black_lift", "severity": 1}]),
        )
        assert len(flagged) == len(quiet) + 1
        assert "black-lift" in flagged[-1].description

    def test_unavailable_side_effect_check_is_skipped(self):
        edited = ORIGINAL.copy()
        edited[10:50, 10:50] = 200
        crops = select_review_crops(ORIGINAL, edited, SideEffectCheck(status="unavailable"))
        assert all("black-lift" not in crop.description for crop in crops)


class TestDeriveOutcome:
    def _outcome(self, returned=EDITED, review=None, **kwargs):
        params: Dict[str, Any] = {
            "original": ORIGINAL,
            "returned": returned,
            "is_synthetic": False,
            "reference_met": False,
            "history": [_item()],
            "intent": None,
            "review": review,
        }
        params.update(kwargs)
        return derive_outcome(**params)

    def _review(self, verdict: Optional[str], image=EDITED, status: str = "completed"):
        return ReviewResult(
            status=status, verdict=verdict, candidate_sha256=image_sha256(image), reason="because"
        )

    def test_real_image_without_review_is_unverified(self):
        assert self._outcome().status == "UNVERIFIED"
        assert self._outcome(review=ReviewResult()).status == "UNVERIFIED"

    def test_accept_review_accepts(self):
        outcome = self._outcome(review=self._review("ACCEPT"))
        assert outcome.status == "ACCEPTED"
        assert outcome.accepted_by == "review"
        assert outcome.candidate_sha256 == image_sha256(EDITED)

    def test_accept_of_a_different_image_is_not_accepted(self):
        outcome = self._outcome(review=self._review("ACCEPT", image=EARLIER))
        assert outcome.status == "UNVERIFIED"

    def test_revise_is_not_met(self):
        assert self._outcome(review=self._review("REVISE")).status == "NOT_MET"

    def test_reject_and_uncertain_are_unverified(self):
        assert self._outcome(review=self._review("REJECT")).status == "UNVERIFIED"
        assert self._outcome(review=self._review("UNCERTAIN")).status == "UNVERIFIED"

    def test_unavailable_review_never_accepts(self):
        outcome = self._outcome(review=self._review(None, status="unavailable"))
        assert outcome.status == "UNVERIFIED"

    def test_identical_to_original_with_fix_is_kept_original(self):
        outcome = self._outcome(returned=ORIGINAL.copy())
        assert outcome.status == "KEPT_ORIGINAL"

    def test_identical_to_original_with_user_fix_intent(self):
        outcome = self._outcome(
            returned=ORIGINAL.copy(),
            history=[_item(with_action=False)],
            intent=IntentProfile(fix=["noise"]),
        )
        assert outcome.status == "KEPT_ORIGINAL"

    def test_nothing_to_fix_is_no_change(self):
        outcome = self._outcome(returned=ORIGINAL.copy(), history=[_item(with_action=False)])
        assert outcome.status == "NO_CHANGE"

    def test_actionable_defect_without_actions_counts_as_wanted_fix(self):
        diagnosis = DiagnosisReport.model_validate(
            {
                "defects": [
                    {"type": "underexposed", "region": "full", "severity": 2, "evidence": "x"}
                ]
            }
        )
        item = _item(with_action=False)
        item.diagnosis = diagnosis
        outcome = self._outcome(returned=ORIGINAL.copy(), history=[item])
        assert outcome.status == "KEPT_ORIGINAL"

    def test_synthetic_reference(self):
        accepted = self._outcome(is_synthetic=True, reference_met=True)
        assert (accepted.status, accepted.accepted_by) == ("ACCEPTED", "reference")
        assert self._outcome(is_synthetic=True, reference_met=False).status == "NOT_MET"


def _state(**overrides: Any) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "original_image": ORIGINAL,
        "current_image": EDITED,
        "previous_image": EARLIER,
        "ground_truth_image": None,
        "is_synthetic": False,
        "iteration": 2,
        "max_iterations": 3,
        "technical_metrics": {},
        "treatment_plan": _plan(),
        "evaluation_result": {"estimated_quality_score": 90.0},
        "history": [_item()],
        "intermediate_images": [],
        "decision": "SHIP",
        "rolled_back": False,
        "intent": None,
    }
    state.update(overrides)
    return state


class TestReviewNode:
    def test_review_receives_hash_of_returned_image(self, review_on):
        _, patcher = review_on([_review_json("ACCEPT")])
        with patcher:
            update = review_node(_state())
        assert update["review"].candidate_sha256 == image_sha256(EDITED)
        assert update["history"][-1].review is update["review"]

    def test_review_after_rollback_hashes_the_rolled_back_image(self, review_on):
        # decide_node đã đặt current_image = ảnh trước vòng; review phải xem đúng ảnh đó
        _, patcher = review_on([_review_json("ACCEPT")])
        with patcher:
            update = review_node(_state(current_image=EARLIER, rolled_back=True))
        assert update["review"].candidate_sha256 == image_sha256(EARLIER)

    def test_reject_returns_previous_image(self, review_on):
        _, patcher = review_on([_review_json("REJECT", [FINDING_FULL])])
        with patcher:
            update = review_node(_state())
        assert np.array_equal(update["current_image"], EARLIER)
        assert update["rolled_back"] is True
        assert update["evaluation_result"] is not None

    def test_reject_with_no_previous_returns_original(self, review_on):
        _, patcher = review_on([_review_json("REJECT", [FINDING_FULL])])
        with patcher:
            update = review_node(_state(previous_image=None))
        assert np.array_equal(update["current_image"], ORIGINAL)

    def test_failure_keeps_image_and_never_accepts(self, review_on):
        _, patcher = review_on([ValueError("boom")])
        with patcher:
            update = review_node(_state())
        assert "current_image" not in update
        finalize = finalize_node({**_state(), **update})
        assert finalize["outcome"].status == "UNVERIFIED"
        assert finalize["decision"] == "STOP_BEST_EFFORT"

    def test_unchanged_image_is_not_reviewed(self, review_on):
        mock_genai, patcher = review_on([])
        with patcher:
            update = review_node(_state(current_image=ORIGINAL.copy()))
        assert update["review"].status == "not_run"
        mock_genai.Client.assert_not_called()

    def test_reference_run_is_not_reviewed(self, review_on):
        mock_genai, patcher = review_on([])
        with patcher:
            update = review_node(_state(is_synthetic=True, ground_truth_image=ORIGINAL))
        assert update["review"].status == "not_run"
        mock_genai.Client.assert_not_called()


class TestFinalize:
    def test_only_accepted_ships(self):
        accepted = ReviewResult(
            status="completed", verdict="ACCEPT", candidate_sha256=image_sha256(EDITED)
        )
        done = finalize_node(_state(review=accepted))
        assert (done["outcome"].status, done["decision"]) == ("ACCEPTED", "SHIP")

        for review in (None, ReviewResult()):
            done = finalize_node(_state(review=review))
            assert done["outcome"].status == "UNVERIFIED"
            assert done["decision"] == "STOP_BEST_EFFORT"

    def test_rejected_candidate_that_equals_original_is_kept_original(self):
        done = finalize_node(
            _state(
                current_image=ORIGINAL.copy(),
                review=ReviewResult(status="completed", verdict="REJECT"),
            )
        )
        assert done["outcome"].status == "KEPT_ORIGINAL"


class TestRollbackReevaluation:
    def test_evaluation_describes_the_returned_image(self):
        """Sau rollback, evaluation_result phải của ảnh được trả, không phải ảnh bị loại."""
        bad = np.full((32, 32, 3), 250, dtype=np.uint8)
        good = np.full((32, 32, 3), 128, dtype=np.uint8)
        stale = {"quality_improved": False, "estimated_quality_score": 1.0}
        state = _state(
            original_image=good,
            current_image=bad,
            previous_image=good,
            evaluation_result=stale,
            history=[],
            iteration=1,
        )
        result = decide_node(state)
        assert result["rolled_back"] is True
        assert np.array_equal(result["current_image"], good)
        assert result["evaluation_result"] != stale
        assert result["history"][-1].eval_score == stale

    def test_no_rollback_leaves_evaluation_untouched(self):
        state = _state(
            decision="PENDING",
            evaluation_result={"estimated_quality_score": 95.0, "quality_improved": True},
            history=[],
            iteration=1,
        )
        assert "evaluation_result" not in decide_node(state)


class TestPipelineOutcome:
    def _dark(self) -> np.ndarray:
        return np.full((64, 64, 3), 40, dtype=np.uint8)

    def test_real_image_without_review_is_unverified(self):
        state = run_pipeline(self._dark(), max_iterations=1)
        assert not np.array_equal(state["current_image"], self._dark())
        assert state["outcome"].status == "UNVERIFIED"
        assert state["decision"] == "STOP_BEST_EFFORT"
        assert state["review"].status == "not_run"

    def test_accepting_review_ships(self):
        accept = ReviewResult(status="completed", verdict="ACCEPT", reason="looks good")

        def fake_review(original, previous, candidate, context=None):
            return accept.model_copy(update={"candidate_sha256": image_sha256(candidate)})

        with patch("src.agent.graph.review_candidate", side_effect=fake_review):
            state = run_pipeline(self._dark(), max_iterations=1)
        assert not np.array_equal(state["current_image"], self._dark())
        assert state["outcome"].status == "ACCEPTED"
        assert state["outcome"].accepted_by == "review"
        assert state["decision"] == "SHIP"
        assert state["outcome"].candidate_sha256 == image_sha256(state["current_image"])

    def test_synthetic_with_reference_is_accepted_by_reference(self):
        truth = np.full((64, 64, 3), 128, dtype=np.uint8)
        state = run_pipeline(truth.copy(), ground_truth=truth, is_synthetic=True, max_iterations=1)
        assert state["outcome"].status == "ACCEPTED"
        assert state["outcome"].accepted_by == "reference"
        assert state["decision"] == "SHIP"
