"""
Unit tests cho hội thoại ý định (Phase 4): câu hỏi tất định từ chẩn đoán, ghi chú tự do,
áp ý định lên chẩn đoán, node clarify với interrupt/resume và phiên tương tác.
Chạy offline; Gemini được mock.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.agent import session as session_module
from src.agent.graph import run_pipeline
from src.agent.intent import (
    MAX_QUESTIONS,
    apply_intent,
    build_questions,
    intent_from_answers,
    intent_prompt,
    parse_notes,
)
from src.agent.session import (
    SessionNotFoundError,
    answer_session,
    end_session,
    start_session,
)
from src.agent.state import Defect, DiagnosisReport, IntentProfile, PreserveItem

FAKE_KEY = {"GEMINI_API_KEY": "fake-key-for-test"}


def _diagnosis(*defects: Defect, preserve=()) -> DiagnosisReport:
    return DiagnosisReport(defects=list(defects), preserve=list(preserve))


def _dark_image() -> np.ndarray:
    gradient = np.tile(np.linspace(10, 60, 48), (48, 1)).astype(np.uint8)
    return np.repeat(gradient[:, :, None], 3, axis=2)


# ============================================================
# Câu hỏi
# ============================================================
class TestQuestions:
    def test_clean_image_only_asks_style(self):
        assert [q.id for q in build_questions(_diagnosis())] == ["style"]
        assert [q.id for q in build_questions(None)] == ["style"]

    def test_ambiguities_are_prioritised_and_capped(self):
        diagnosis = _diagnosis(
            Defect(type="underexposed", severity=2),
            Defect(type="color_cast_warm", severity=1),
            Defect(type="noise", severity=2),
            Defect(type="color_cast_cool", severity=1),
        )
        questions = build_questions(diagnosis)
        assert [q.id for q in questions] == ["mood", "warm", "style"]
        assert len(questions) == MAX_QUESTIONS

    def test_defaults_follow_the_diagnosis(self):
        intentional = _diagnosis(preserve=[PreserveItem(aspect="warm_tone")])
        accidental = _diagnosis(Defect(type="color_cast_warm", severity=1))
        assert build_questions(intentional)[0].default == "keep"
        assert build_questions(accidental)[0].default == "fix"

    def test_backlit_subject_brighten_fixes_backlight(self):
        diagnosis = _diagnosis(Defect(type="backlit_subject", region="face", severity=2))
        [mood, _] = build_questions(diagnosis)
        assert "ngược sáng" in mood.text
        brighten = next(c for c in mood.choices if c.value == "brighten")
        assert brighten.fix == "backlit_subject"

    def test_questions_are_deterministic(self):
        diagnosis = _diagnosis(Defect(type="noise", severity=2))
        assert build_questions(diagnosis) == build_questions(diagnosis)


# ============================================================
# Câu trả lời và ghi chú
# ============================================================
class TestAnswers:
    @pytest.mark.parametrize(
        "notes,expected",
        [
            ("khử vàng đi", {"warm": "fix"}),
            ("giữ tông ấm nhé", {"warm": "keep"}),
            ("muốn ảnh tâm trạng, giữ tối", {"mood": "keep_dark"}),
            ("làm sáng rõ mặt", {"mood": "brighten"}),
            ("cho mịn hơn", {"grain": "smooth"}),
            ("giữ hạt film", {"grain": "keep"}),
            ("rực rỡ lên", {"style": "vivid"}),
            ("tự nhiên thôi", {"style": "natural"}),
            ("", {}),
        ],
    )
    def test_parse_notes(self, notes, expected):
        assert parse_notes(notes) == expected

    def test_explicit_answers_win_over_notes(self):
        questions = build_questions(_diagnosis(Defect(type="color_cast_warm", severity=1)))
        intent = intent_from_answers(questions, {"warm": "keep"}, "khử vàng đi")
        assert intent.answers["warm"] == "keep"
        assert [p.aspect for p in intent.keep] == ["warm_tone"]
        assert intent.fix == []

    def test_unknown_values_are_ignored(self):
        intent = intent_from_answers(build_questions(None), {"style": "sepia", "ghost": "x"})
        assert intent.style is None and intent.answers == {}

    def test_notes_about_unasked_questions_still_apply(self):
        intent = intent_from_answers(build_questions(None), {"style": "vivid"}, "giữ hạt film")
        assert intent.style == "vivid"
        assert [p.aspect for p in intent.keep] == ["film_grain"]
        assert intent.notes == "giữ hạt film"


# ============================================================
# Áp ý định lên chẩn đoán
# ============================================================
class TestApplyIntent:
    def test_fix_removes_conflicting_preserve_and_adds_user_defect(self):
        diagnosis = _diagnosis(preserve=[PreserveItem(aspect="warm_tone")])
        updated = apply_intent(diagnosis, IntentProfile(fix=["color_cast_warm"]))
        assert updated.preserve == []
        [defect] = updated.defects
        assert (defect.type, defect.severity, defect.origin) == ("color_cast_warm", 2, "user")
        assert diagnosis.preserve  # bản gốc không đổi

    def test_fix_raises_existing_severity(self):
        diagnosis = _diagnosis(Defect(type="noise", severity=1))
        [defect] = apply_intent(diagnosis, IntentProfile(fix=["noise"])).defects
        assert (defect.severity, defect.origin) == (2, "vlm")

    def test_keep_drops_conflicting_defects(self):
        diagnosis = _diagnosis(
            Defect(type="underexposed", severity=3),
            Defect(type="backlit_subject", region="face", severity=2),
            Defect(type="noise", severity=2),
        )
        intent = IntentProfile(keep=[PreserveItem(aspect="low_key")])
        updated = apply_intent(diagnosis, intent)
        assert [d.type for d in updated.defects] == ["noise"]
        assert [p.aspect for p in updated.preserve] == ["low_key"]

    def test_no_intent_is_a_no_op(self):
        diagnosis = _diagnosis(Defect(type="noise", severity=2))
        assert apply_intent(diagnosis, None) is diagnosis

    def test_intent_prompt(self):
        intent = IntentProfile(
            style="vivid",
            keep=[PreserveItem(aspect="warm_tone")],
            fix=["noise"],
            notes="làm rõ mặt",
        )
        text = intent_prompt(intent)
        assert text.startswith("Ý ĐỊNH NGƯỜI DÙNG")
        assert "Đậm nét" in text and "warm_tone@full" in text and "noise" in text
        assert "làm rõ mặt" in text
        assert intent_prompt(None) == "" and intent_prompt(IntentProfile()) == ""


# ============================================================
# Graph và phiên tương tác
# ============================================================
class TestInteractivePipeline:
    def test_non_interactive_pipeline_does_not_pause(self):
        state = run_pipeline(_dark_image(), max_iterations=1)
        assert state["intent"] is None
        assert "__interrupt__" not in state

    def test_session_asks_then_applies_the_intent(self):
        started = start_session(_dark_image(), max_iterations=2, num_variants=3)
        assert started.status == "needs_input"
        assert started.questions[-1].id == "style"
        assert started.diagnosis is not None

        done = answer_session(started.session_id, {"style": "natural"}, "giữ tông ấm")
        state = done.state
        assert done.status == "done"
        assert state["intent"].style == "natural"
        assert state["recommended_variant"] == "natural"
        # Ý định áp vào chẩn đoán của mọi vòng
        assert all(
            "warm_tone" in [p.aspect for p in item.diagnosis.preserve] for item in state["history"]
        )
        end_session(started.session_id)

    def test_preset_style_is_rendered_even_with_one_variant(self):
        state = run_pipeline(
            _dark_image(), max_iterations=1, num_variants=1, intent=IntentProfile(style="vivid")
        )
        assert sorted(v.id for v in state["variants"]) == ["balanced", "vivid"]
        assert state["recommended_variant"] == "vivid"

    def test_answering_unknown_or_finished_sessions_fails(self):
        with pytest.raises(SessionNotFoundError):
            answer_session("no-such-session", {})
        started = start_session(_dark_image(), max_iterations=1, num_variants=1)
        answer_session(started.session_id, {})
        with pytest.raises(ValueError, match="not waiting"):
            answer_session(started.session_id, {})
        end_session(started.session_id)

    def test_old_sessions_are_evicted(self, monkeypatch):
        monkeypatch.setattr(session_module, "MAX_SESSIONS", 1)
        first = start_session(_dark_image(), max_iterations=1, num_variants=1)
        second = start_session(_dark_image(), max_iterations=1, num_variants=1)
        with pytest.raises(SessionNotFoundError):
            answer_session(first.session_id, {})
        assert answer_session(second.session_id, {}).status == "done"
        end_session(second.session_id)

    @patch.dict("os.environ", FAKE_KEY)
    @patch("src.agent.vlm_diagnostician.genai")
    def test_plan_prompt_carries_the_intent(self, mock_genai):
        from src.agent.perception import PERCEIVE_RESPONSE_SCHEMA

        diagnosis = {
            "scene_type": "food",
            "defects": [{"type": "underexposed", "region": "full", "severity": 2}],
            "preserve": [],
            "summary": "Tối.",
        }
        plan = {"reasoning": "ok", "actions": []}

        def generate_content(*, model, contents, config):
            is_perceive = config.response_json_schema == PERCEIVE_RESPONSE_SCHEMA
            return MagicMock(text=json.dumps(diagnosis if is_perceive else plan))

        models = MagicMock()
        models.generate_content.side_effect = generate_content
        mock_genai.Client.return_value.models = models

        started = start_session(_dark_image(), max_iterations=1, num_variants=1)
        answer_session(started.session_id, {"style": "vivid"}, "giữ tông ấm")
        end_session(started.session_id)

        prompts = [
            call.kwargs["contents"][0]
            for call in models.generate_content.call_args_list
            if call.kwargs["config"].response_json_schema != PERCEIVE_RESPONSE_SCHEMA
        ]
        assert prompts and "Ý ĐỊNH NGƯỜI DÙNG" in prompts[0]
        assert "warm_tone@full" in prompts[0]
