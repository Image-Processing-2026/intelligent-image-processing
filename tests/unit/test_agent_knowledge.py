"""
Unit tests cho Knowledge Base (Phase 2): nạp và kiểm tra card, đồng bộ với toolbox,
truy xuất (lọc cấu trúc + BM25), rule engine offline và tích hợp vào prompt giai đoạn Plan.
Chạy offline; Gemini được mock.
"""

import json
import textwrap
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.agent.knowledge import (
    KnowledgeBaseError,
    diagnose_from_metrics,
    format_context,
    get_knowledge_base,
    load_knowledge_base,
    match_cards,
    plan_actions_from_knowledge,
    retrieve,
    with_metric_defects,
)
from src.agent.knowledge.engine import instantiate, param_value
from src.agent.knowledge.retriever import BM25, tokenize
from src.agent.planner import clamp_parameters
from src.agent.state import Defect, DiagnosisReport, PreserveItem
from src.agent.vlm_diagnostician import diagnose_and_plan


# ============================================================
# Helpers
# ============================================================
def _write_kb(root: Path, cards_yaml: str, principles_md: str = "") -> Path:
    (root / "cards").mkdir(parents=True)
    (root / "principles").mkdir()
    (root / "cards" / "test.yaml").write_text(textwrap.dedent(cards_yaml), encoding="utf-8")
    if principles_md:
        (root / "principles" / "p.md").write_text(textwrap.dedent(principles_md), "utf-8")
    return root


def _diagnosis(*defects: Defect, scene: str = "other", preserve=()) -> DiagnosisReport:
    return DiagnosisReport(scene_type=scene, defects=list(defects), preserve=list(preserve))


def _ids(matches) -> List[str]:
    return [match.card.id for match in matches]


VALID_CARD = """
- id: lift
  title: Lift
  auto_apply: true
  defects:
    - {types: [underexposed], region: full, min_severity: 2}
  recipe:
    - operation: gamma_correct
      region: full
      params: {gamma: {2: 1.3, 3: 1.5}}
"""


# ============================================================
# Nạp KB và đồng bộ với toolbox
# ============================================================
class TestKnowledgeBaseContent:
    def test_default_kb_loads(self):
        kb = get_knowledge_base()
        assert len(kb.cards) >= 15
        assert len(kb.principles) >= 10
        assert all(p.tags for p in kb.principles)

    @pytest.mark.parametrize("severity", [1, 2, 3])
    def test_every_recipe_is_within_parameter_bounds(self, severity: int):
        """Tham số mọi card (mọi mức severity) đi qua planner mà không cần kẹp."""
        for card in get_knowledge_base().cards:
            for step in card.recipe:
                params = {name: param_value(spec, severity) for name, spec in step.params.items()}
                clamped = clamp_parameters(step.operation, params)
                assert {k: clamped[k] for k in params} == params, (card.id, step.operation)

    def test_auto_apply_cards_avoid_measured_regressions(self):
        """Phase 0 đo được: sharpen và chỉnh màu tự động làm giảm chất lượng ảnh thật."""
        for card in get_knowledge_base().cards:
            if card.auto_apply:
                ops = {step.operation for step in card.recipe}
                assert not ops & {"sharpen", "color_correct"}, card.id


class TestKnowledgeBaseValidation:
    def test_valid_custom_kb(self, tmp_path: Path):
        kb = load_knowledge_base(
            _write_kb(
                tmp_path,
                VALID_CARD,
                """
                # Nguyên lý
                ## Gamma và đèn
                tags: gamma, exposure
                sources: Sách A; Sách B
                Dòng một
                dòng hai.
                """,
            )
        )
        assert [card.id for card in kb.cards] == ["lift"]
        [principle] = kb.principles
        assert principle.id == "p/gamma-va-den"
        assert principle.tags == ["gamma", "exposure"]
        assert principle.sources == ["Sách A", "Sách B"]
        assert principle.text == "Dòng một dòng hai."

    @pytest.mark.parametrize(
        "original,broken,message",
        [
            ("operation: gamma_correct", "operation: face_beautify", "not in the toolbox"),
            ("{gamma: {2: 1.3, 3: 1.5}}", "{gamma: {2: 9.0}}", "outside"),
            ("{gamma: {2: 1.3, 3: 1.5}}", "{strength: 1.0}", "has no parameter"),
            ("{gamma: {2: 1.3, 3: 1.5}}", "{gamma: {5: 1.3}}", "severity keys"),
            ("types: [underexposed]", "types: [haze]", "unknown defect types"),
        ],
    )
    def test_invalid_card_is_rejected_with_location(
        self, tmp_path: Path, original: str, broken: str, message: str
    ):
        _write_kb(tmp_path, VALID_CARD.replace(original, broken))
        with pytest.raises(KnowledgeBaseError, match=message) as excinfo:
            load_knowledge_base(tmp_path)
        assert "test.yaml" in str(excinfo.value)
        assert "lift" in str(excinfo.value)

    def test_auto_apply_card_needs_recipe(self, tmp_path: Path):
        card = VALID_CARD.split("  recipe:")[0] + "  recipe: []\n"
        _write_kb(tmp_path, card)
        with pytest.raises(KnowledgeBaseError, match="needs a recipe"):
            load_knowledge_base(tmp_path)

    def test_duplicate_ids_are_rejected(self, tmp_path: Path):
        _write_kb(tmp_path, VALID_CARD + VALID_CARD)
        with pytest.raises(KnowledgeBaseError, match="duplicate card ids"):
            load_knowledge_base(tmp_path)

    def test_invalid_yaml_is_rejected(self, tmp_path: Path):
        _write_kb(tmp_path, "- id: [unclosed")
        with pytest.raises(KnowledgeBaseError, match="invalid YAML"):
            load_knowledge_base(tmp_path)


# ============================================================
# BM25
# ============================================================
class TestBM25:
    def test_tokenize_splits_identifiers(self):
        assert tokenize("color_cast_warm Ấm") == ["color_cast_warm", "color", "cast", "warm", "ấm"]

    def test_relevant_document_ranks_first(self):
        bm25 = BM25([tokenize("gamma exposure dark"), tokenize("noise denoise median")])
        scores = bm25.scores(tokenize("noise"))
        assert scores[1] > scores[0] == 0.0


# ============================================================
# Truy xuất card
# ============================================================
class TestMatchCards:
    def test_backlit_face_matches_dark_region(self):
        diagnosis = _diagnosis(Defect(type="backlit_subject", region="face", severity=3))
        matches = match_cards(diagnosis)
        assert _ids(matches)[0] == "dark-region"
        assert matches[0].bindings[0].region == "face"

    def test_unless_blocks_region_lift_when_globally_dark(self):
        diagnosis = _diagnosis(
            Defect(type="underexposed", severity=2),
            Defect(type="underexposed", region="face", severity=2),
        )
        ids = _ids(match_cards(diagnosis))
        assert "global-underexposed" in ids
        assert "dark-region" not in ids

    def test_preserve_blocks_cards(self):
        cast = Defect(type="color_cast_warm", severity=2)
        assert "warm-cast-correction" in _ids(match_cards(_diagnosis(cast)))
        preserved = _diagnosis(cast, preserve=[PreserveItem(aspect="warm_tone")])
        assert "warm-cast-correction" not in _ids(match_cards(preserved))

    def test_region_preserve_only_blocks_that_region(self):
        face_dark = Defect(type="underexposed", region="face", severity=2)
        silhouette_person = [PreserveItem(aspect="silhouette", region="person")]
        assert "dark-region" in _ids(match_cards(_diagnosis(face_dark, preserve=silhouette_person)))
        silhouette_face = [PreserveItem(aspect="silhouette", region="face")]
        assert "dark-region" not in _ids(
            match_cards(_diagnosis(face_dark, preserve=silhouette_face))
        )

    def test_match_all_requires_every_condition(self):
        dark = Defect(type="underexposed", severity=2)
        noisy = Defect(type="noise", severity=2)
        assert "low-light-noisy" not in _ids(match_cards(_diagnosis(dark)))
        assert "low-light-noisy" in _ids(match_cards(_diagnosis(dark, noisy)))

    def test_scene_cards_need_matching_scene(self):
        assert "scene-food" in _ids(match_cards(_diagnosis(scene="food")))
        assert "scene-food" not in _ids(match_cards(_diagnosis(scene="portrait")))

    def test_severity_below_minimum_does_not_match(self):
        diagnosis = _diagnosis(Defect(type="noise", severity=1))
        assert "noise-reduction" not in _ids(match_cards(diagnosis))

    def test_metric_conditions(self, tmp_path: Path):
        card = VALID_CARD + "  metrics: {noise_level: {in: [clean, low]}, contrast_std: {lt: 30}}\n"
        kb = load_knowledge_base(_write_kb(tmp_path, card))
        diagnosis = _diagnosis(Defect(type="underexposed", severity=2))
        assert _ids(match_cards(diagnosis, {"noise_level": "low", "contrast_std": 20}, kb)) == [
            "lift"
        ]
        assert match_cards(diagnosis, {"noise_level": "severe", "contrast_std": 20}, kb) == []
        assert match_cards(diagnosis, {"noise_level": "low"}, kb) == []


class TestRetrieve:
    def test_food_scene_keeps_warmth_knowledge(self):
        diagnosis = _diagnosis(
            Defect(type="underexposed", severity=3),
            Defect(type="color_cast_warm", severity=1),
            scene="food",
            preserve=[PreserveItem(aspect="warm_tone")],
        )
        context = retrieve(diagnosis)
        card_ids = [match.card.id for match in context.cards]
        assert "global-underexposed" in card_ids
        assert "scene-food" in card_ids
        assert "warm-cast-correction" not in card_ids
        assert "color_workflow/tong-am-co-chu-y" in [p.id for p in context.principles]

    def test_limits_are_respected(self):
        diagnosis = _diagnosis(
            Defect(type="noise", severity=3),
            Defect(type="blur", severity=3),
            Defect(type="undersaturated", severity=2),
            Defect(type="low_contrast", severity=2),
            Defect(type="color_cast_cool", severity=2),
            scene="landscape",
        )
        context = retrieve(diagnosis, max_cards=2, max_principles=1)
        assert len(context.cards) == 2
        assert len(context.principles) == 1

    def test_format_context(self):
        diagnosis = _diagnosis(Defect(type="backlit_subject", region="face", severity=2))
        text = format_context(retrieve(diagnosis))
        assert text.startswith("TRI THỨC CHUYÊN MÔN")
        assert "[dark-region]" in text
        assert "backlit_subject@face (mức 2)" in text
        assert (
            "gamma_correct trên vùng 'face' (mức 2): gamma=1.4 [mức 2→1.4, mức 3→1.6], "
            "feather_radius=30"
        ) in text
        assert "Tránh:" in text

    def test_format_context_picks_values_per_binding(self):
        """Card nhiều bước: mỗi bước dùng severity của lỗi khớp; nối các bước theo thứ tự."""
        diagnosis = _diagnosis(
            Defect(type="underexposed", severity=2), Defect(type="noise", severity=3)
        )
        text = format_context(retrieve(diagnosis))
        assert (
            "Công thức: denoise trên toàn ảnh (mức 3): method=bilateral, "
            "strength=2.0 [mức 2→1.5, mức 3→2.0]; rồi gamma_correct trên toàn ảnh (mức 3)"
        ) in text

    def test_empty_context_formats_to_empty_string(self):
        assert format_context(retrieve(_diagnosis(), max_principles=0)) == ""


# ============================================================
# Rule engine offline
# ============================================================
class TestEngine:
    def test_param_value_by_severity(self):
        assert param_value(1.2, 3) == 1.2
        assert param_value({2: 1.3, 3: 1.5}, 3) == 1.5
        assert param_value({2: 1.3, 3: 1.5}, 1) == 1.3  # thấp hơn mọi khóa → khóa nhỏ nhất
        assert param_value({1: 0.5, 3: 1.0}, 2) == 0.5  # mức thấp hơn gần nhất

    def test_defect_binding_per_region_and_order(self):
        diagnosis = _diagnosis(
            Defect(type="backlit_subject", region="face", severity=2),
            Defect(type="underexposed", region="face", severity=3),
            Defect(type="overexposed", region="sky", severity=2),
            Defect(type="noise", severity=3),
        )
        actions, used = instantiate(match_cards(diagnosis))
        summary = [(a["operation"], a["target_prompt"], a["parameters"]) for a in actions]
        assert summary == [
            ("denoise", "full", {"method": "bilateral", "strength": 1.5}),
            ("gamma_correct", "face", {"gamma": 1.6}),
            ("gamma_correct", "sky", {"gamma": 0.8}),
        ]
        assert [a["order"] for a in actions] == [1, 2, 3]
        assert all(a["feather_radius"] == 30 for a in actions[1:])
        assert set(used) == {"noise-reduction", "dark-region", "bright-region"}

    def test_advisory_cards_are_not_auto_applied(self):
        diagnosis = _diagnosis(
            Defect(type="blur", severity=2), Defect(type="color_cast_warm", severity=2)
        )
        assert instantiate(match_cards(diagnosis)) == ([], [])
        actions, _ = instantiate(match_cards(diagnosis), auto_only=False)
        assert {a["operation"] for a in actions} == {"sharpen", "color_correct"}

    def test_higher_priority_card_wins_duplicate_operation(self):
        """low-light-noisy (priority 70) thắng noise-reduction và global-underexposed (50)."""
        diagnosis = _diagnosis(
            Defect(type="underexposed", severity=2), Defect(type="noise", severity=2)
        )
        actions, used = instantiate(match_cards(diagnosis))
        assert [(a["operation"], a["parameters"]) for a in actions] == [
            ("denoise", {"method": "bilateral", "strength": 1.5}),
            ("gamma_correct", {"gamma": 1.3}),
        ]
        assert used == ["low-light-noisy"]

    def test_with_metric_defects_adds_missing_global_defects(self):
        diagnosis = _diagnosis(
            Defect(type="noise", severity=1),
            Defect(type="underexposed", region="face", severity=2),
        )
        merged = with_metric_defects(
            diagnosis, {"noise_level": "severe", "brightness_level": "underexposed"}
        )
        assert [(d.type, d.region, d.origin) for d in merged.defects] == [
            ("noise", "full", "vlm"),
            ("underexposed", "face", "vlm"),
            ("underexposed", "full", "rule"),
        ]
        assert len(diagnosis.defects) == 2  # bản gốc không đổi

    def test_preserve_still_blocks_metric_defects(self):
        diagnosis = _diagnosis(preserve=[PreserveItem(aspect="low_key")])
        actions, _ = plan_actions_from_knowledge(diagnosis, {"brightness_level": "underexposed"})
        assert actions == []

    def test_diagnose_from_metrics_without_diagnosis(self):
        actions, used = plan_actions_from_knowledge(None, {"noise_level": "medium"})
        assert [a["operation"] for a in actions] == ["denoise"]
        assert used == ["noise-reduction"]
        assert diagnose_from_metrics({}).subjects == ["face"]


# ============================================================
# Tích hợp với giai đoạn Plan
# ============================================================
FAKE_KEY = {"GEMINI_API_KEY": "fake-key-for-test"}


class TestPlanIntegration:
    def test_offline_plan_records_playbook(self):
        img = np.ones((32, 32, 3), dtype=np.uint8) * 128
        plan = diagnose_and_plan(img, {"noise_level": "severe", "brightness_level": "underexposed"})
        assert plan.knowledge == ["low-light-noisy"]
        assert [a.operation for a in plan.actions] == ["denoise", "gamma_correct"]
        assert "Playbook:" in plan.reasoning

    @patch.dict("os.environ", FAKE_KEY)
    @patch("src.agent.vlm_diagnostician.genai")
    def test_plan_prompt_carries_knowledge(self, mock_genai):
        response = json.dumps({"reasoning": "[dark-region] nâng mặt", "actions": []})
        models = MagicMock()
        models.generate_content.return_value = MagicMock(text=response)
        mock_genai.Client.return_value.models = models
        diagnosis = _diagnosis(
            Defect(type="backlit_subject", region="face", severity=2), scene="portrait"
        )

        img = np.ones((32, 32, 3), dtype=np.uint8) * 128
        plan = diagnose_and_plan(img, {}, diagnosis=diagnosis)

        prompt = models.generate_content.call_args.kwargs["contents"][0]
        assert "TRI THỨC CHUYÊN MÔN" in prompt
        assert "[dark-region]" in prompt
        assert "[scene-portrait]" in prompt
        assert "dark-region" in plan.knowledge
        assert "scene-portrait" in plan.knowledge

    @patch.dict("os.environ", FAKE_KEY)
    @patch("src.agent.vlm_diagnostician.genai")
    def test_plan_without_diagnosis_uses_metric_diagnosis_for_retrieval(self, mock_genai):
        models = MagicMock()
        models.generate_content.return_value = MagicMock(
            text=json.dumps({"reasoning": "ok", "actions": []})
        )
        mock_genai.Client.return_value.models = models

        img = np.ones((32, 32, 3), dtype=np.uint8) * 128
        plan = diagnose_and_plan(img, {"noise_level": "severe"})

        assert "noise-reduction" in plan.knowledge
