"""
Unit tests cho chat bổ sung phân tích (T3): parse luật tiếng Việt, fallback Gemini (mock),
áp DiagnosisEdit lên chẩn đoán và hiệu lực qua nhiều vòng. Chạy offline.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.agent.intent import apply_intent
from src.agent.state import Defect, DiagnosisEdit, DiagnosisReport, IntentProfile, PreserveItem
from src.agent.user_analysis import apply_edits, parse_edits, parse_edits_rules

FAKE_KEY = {"GEMINI_API_KEY": "fake-key-for-test"}


def _diagnosis(*defects: Defect, preserve=()) -> DiagnosisReport:
    return DiagnosisReport(defects=list(defects), preserve=list(preserve))


def _brief(edits):
    return [(e.op, e.type or e.aspect, e.region) for e in edits]


# ============================================================
# Luật tiếng Việt
# ============================================================
class TestRules:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("mặt còn tối", [("add_defect", "underexposed", "face")]),
            ("người bị thiếu sáng", [("add_defect", "underexposed", "person")]),
            ("trời bị cháy", [("add_defect", "overexposed", "sky")]),
            ("ảnh có nhiều nhiễu", [("add_defect", "noise", "full")]),
            ("ảnh bị mờ", [("add_defect", "blur", "full")]),
            ("ảnh không nét", [("add_defect", "blur", "full")]),
            ("ám vàng", [("add_defect", "color_cast_warm", "full")]),
            ("ám xanh lá", [("add_defect", "color_cast_green", "full")]),
            ("màu nhạt quá", [("add_defect", "undersaturated", "full")]),
            ("ảnh bị bệt", [("add_defect", "low_contrast", "full")]),
            ("nền không bị cháy", [("remove_defect", "overexposed", "background")]),
            ("đừng sửa nhiễu", [("remove_defect", "noise", "full")]),
            (
                "nền cháy là cố ý",
                [
                    ("remove_defect", "overexposed", "background"),
                    ("keep", "high_key", "background"),
                ],
            ),
            ("giữ tông ấm", [("keep", "warm_tone", "full")]),
            ("giữ hạt film", [("keep", "film_grain", "full")]),
            ("giữ không khí tối", [("keep", "low_key", "full")]),
            (
                "mặt còn tối, nền không bị cháy đâu, giữ tông ấm",
                [
                    ("add_defect", "underexposed", "face"),
                    ("remove_defect", "overexposed", "background"),
                    ("keep", "warm_tone", "full"),
                ],
            ),
            ("hôm nay trời đẹp", []),
            ("", []),
        ],
    )
    def test_table(self, text, expected):
        assert _brief(parse_edits_rules(text)) == expected

    def test_severity_follows_the_wording(self):
        light, plain, strong = (
            parse_edits_rules(text)[0].severity
            for text in ("mặt hơi tối", "mặt tối", "mặt tối quá")
        )
        assert (light, plain, strong) == (1, 2, 3)

    def test_source_text_keeps_the_clause(self):
        [edit] = parse_edits_rules("mặt còn tối")
        assert edit.source_text == "mặt còn tối"

    def test_edit_model_validates_its_target(self):
        with pytest.raises(ValidationError):
            DiagnosisEdit(op="add_defect")
        with pytest.raises(ValidationError):
            DiagnosisEdit(op="keep", aspect="sepia")
        assert DiagnosisEdit(op="add_defect", type="noise", region="Khuôn mặt").region == "face"


# ============================================================
# Gemini cho câu luật không hiểu
# ============================================================
class TestParseEdits:
    def test_rules_win_without_calling_the_vlm(self):
        with patch("src.agent.user_analysis.call_gemini_json") as call:
            edits, source = parse_edits("mặt còn tối", None)
        assert source == "rule" and len(edits) == 1
        call.assert_not_called()

    def test_no_key_means_no_call(self):
        with patch("src.agent.user_analysis.call_gemini_json") as call:
            assert parse_edits("khuôn mặt hơi thiếu ánh", None) == ([], "none")
        call.assert_not_called()

    @patch.dict("os.environ", FAKE_KEY)
    @patch("src.agent.vlm_diagnostician.genai")
    def test_unknown_text_goes_to_the_vlm(self, mock_genai):
        payload = {"edits": [{"op": "add_defect", "type": "underexposed", "region": "Face"}]}
        models = MagicMock()
        models.generate_content.return_value = MagicMock(text=json.dumps(payload))
        mock_genai.Client.return_value.models = models
        diagnosis = _diagnosis(Defect(type="noise", severity=2))

        edits, source = parse_edits("khuôn mặt thiếu ánh", diagnosis)

        assert source == "vlm"
        assert _brief(edits) == [("add_defect", "underexposed", "face")]
        prompt = models.generate_content.call_args.kwargs["config"].system_instruction
        assert "noise@full" in prompt  # prompt có tóm tắt chẩn đoán

    @patch.dict("os.environ", FAKE_KEY)
    @patch("src.agent.vlm_diagnostician.genai")
    def test_broken_vlm_json_falls_back(self, mock_genai):
        models = MagicMock()
        models.generate_content.return_value = MagicMock(text="{not json")
        mock_genai.Client.return_value.models = models
        assert parse_edits("khuôn mặt thiếu ánh", None) == ([], "vlm_fallback")

    @patch.dict("os.environ", FAKE_KEY)
    @patch("src.agent.vlm_diagnostician.genai")
    def test_invalid_vlm_edits_are_dropped(self, mock_genai):
        payload = {"edits": [{"op": "keep", "aspect": "sepia"}, {"op": "add_defect"}]}
        models = MagicMock()
        models.generate_content.return_value = MagicMock(text=json.dumps(payload))
        mock_genai.Client.return_value.models = models
        assert parse_edits("khuôn mặt thiếu ánh", None) == ([], "none")


# ============================================================
# apply_edits
# ============================================================
class TestApplyEdits:
    def test_add_creates_a_user_defect(self):
        diagnosis = _diagnosis()
        edit = DiagnosisEdit(op="add_defect", type="underexposed", region="face", severity=3)
        [defect] = apply_edits(diagnosis, [edit]).defects
        assert (defect.type, defect.region, defect.severity) == ("underexposed", "face", 3)
        assert defect.origin == "user" and defect.confidence == 1.0 and defect.fixable
        assert diagnosis.defects == []  # bản gốc không đổi

    def test_add_raises_the_severity_of_an_existing_defect(self):
        diagnosis = _diagnosis(Defect(type="noise", severity=1, origin="vlm"))
        edit = DiagnosisEdit(op="add_defect", type="noise", severity=3)
        [defect] = apply_edits(diagnosis, [edit]).defects
        assert (defect.severity, defect.origin) == (3, "user")

    def test_add_on_a_backlit_subject_upgrades_it_instead(self):
        diagnosis = _diagnosis(Defect(type="backlit_subject", region="person", severity=1))
        edit = DiagnosisEdit(op="add_defect", type="underexposed", region="face", severity=3)
        [defect] = apply_edits(diagnosis, [edit]).defects
        assert (defect.type, defect.region, defect.severity) == ("backlit_subject", "person", 3)

    def test_unfixable_types_stay_unfixable(self):
        edit = DiagnosisEdit(op="add_defect", type="haze", severity=3)
        [defect] = apply_edits(_diagnosis(), [edit]).defects
        assert defect.fixable is False

    def test_add_drops_a_conflicting_preserve(self):
        diagnosis = _diagnosis(preserve=[PreserveItem(aspect="low_key")])
        edit = DiagnosisEdit(op="add_defect", type="underexposed", region="face")
        updated = apply_edits(diagnosis, [edit])
        assert updated.preserve == [] and [d.type for d in updated.defects] == ["underexposed"]

    def test_remove_moves_the_defect_to_rejected(self):
        diagnosis = _diagnosis(
            Defect(type="overexposed", region="background", severity=2),
            Defect(type="noise", severity=2),
        )
        edit = DiagnosisEdit(op="remove_defect", type="overexposed", region="background")
        updated = apply_edits(diagnosis, [edit])
        assert [d.type for d in updated.defects] == ["noise"]
        assert [d.type for d in updated.rejected] == ["overexposed"]
        assert any("user removed" in note for note in updated.fusion_notes)

    def test_keep_drops_conflicting_defects(self):
        diagnosis = _diagnosis(
            Defect(type="underexposed", severity=2), Defect(type="noise", severity=2)
        )
        updated = apply_edits(diagnosis, [DiagnosisEdit(op="keep", aspect="low_key")])
        assert [d.type for d in updated.defects] == ["noise"]
        assert [p.aspect for p in updated.preserve] == ["low_key"]

    def test_later_edit_wins(self):
        add = DiagnosisEdit(op="add_defect", type="noise", severity=3)
        remove = DiagnosisEdit(op="remove_defect", type="noise")
        assert apply_edits(_diagnosis(), [add, remove]).defects == []
        assert [d.type for d in apply_edits(_diagnosis(), [remove, add]).defects] == ["noise"]

    def test_no_edits_is_a_no_op(self):
        diagnosis = _diagnosis(Defect(type="noise", severity=2))
        assert apply_edits(diagnosis, []) is diagnosis


# ============================================================
# apply_intent với edits
# ============================================================
class TestIntentEdits:
    def test_edits_beat_a_conflicting_answer(self):
        # Câu trả lời trắc nghiệm giữ không khí tối, nhưng người dùng nói mặt còn tối
        intent = IntentProfile(
            keep=[PreserveItem(aspect="low_key")],
            edits=[DiagnosisEdit(op="add_defect", type="underexposed", region="face")],
        )
        updated = apply_intent(_diagnosis(), intent)
        assert [(d.type, d.region, d.origin) for d in updated.defects] == [
            ("underexposed", "face", "user")
        ]

    def test_removed_defect_stays_removed_in_the_next_iteration(self):
        intent = IntentProfile(
            edits=[DiagnosisEdit(op="remove_defect", type="overexposed", region="background")]
        )
        first = _diagnosis(Defect(type="overexposed", region="background", severity=3))
        second = _diagnosis(Defect(type="overexposed", region="background", severity=2))
        for diagnosis in (first, second):
            assert apply_intent(diagnosis, intent).defects == []

    def test_user_defect_survives_every_iteration(self):
        intent = IntentProfile(edits=[DiagnosisEdit(op="add_defect", type="noise", severity=3)])
        for iteration in (1, 2):
            diagnosis = DiagnosisReport(iteration=iteration)
            [defect] = apply_intent(diagnosis, intent).defects
            assert defect.origin == "user" and defect.severity == 3

    def test_intent_serialises_edits(self):
        intent = IntentProfile(edits=[DiagnosisEdit(op="keep", aspect="warm_tone")])
        restored = IntentProfile.model_validate(intent.model_dump(mode="json"))
        assert restored.edits == intent.edits
