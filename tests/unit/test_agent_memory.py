"""
Unit tests cho bộ nhớ ca bệnh (Phase 5): ghi ca, lựa chọn và góp ý; truy xuất ca tương tự;
sở thích phong cách; tích hợp vào prompt Plan và phiên bản đề xuất. Chạy offline với SQLite
trong thư mục tạm; Gemini được mock.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.agent import memory as memory_module
from src.agent.graph import run_pipeline
from src.agent.memory import (
    MAX_DISTANCE,
    CaseMemory,
    CaseRecord,
    configure_case_memory,
    experience_prompt,
    metric_signature,
    remember_run,
)
from src.agent.refine import Adjustment
from src.agent.state import Defect, DiagnosisReport, IntentProfile

FAKE_KEY = {"GEMINI_API_KEY": "fake-key-for-test"}


def _dark_image(low: float = 10, high: float = 60) -> np.ndarray:
    gradient = np.tile(np.linspace(low, high, 48), (48, 1)).astype(np.uint8)
    return np.repeat(gradient[:, :, None], 3, axis=2)


def _case(case_id: str, **fields) -> CaseRecord:
    defaults = {"created_at": f"2026-10-0{len(case_id) % 9 + 1}T00:00:00", "signature": [0.2] * 6}
    return CaseRecord(id=case_id, **{**defaults, **fields})


@pytest.fixture
def memory(tmp_path: Path) -> CaseMemory:
    """Bộ nhớ bật trên file tạm cho test này (fixture chung tắt nó sau test)."""
    return configure_case_memory(tmp_path / "memory" / "cases.sqlite")


# ============================================================
# Ghi và đọc
# ============================================================
class TestRecording:
    def test_database_is_created_lazily(self, tmp_path: Path):
        path = tmp_path / "lazy" / "cases.sqlite"
        store = CaseMemory(path)
        assert not path.exists()
        assert store.cases() == []
        assert path.exists()

    def test_record_run_from_pipeline_state(self, memory: CaseMemory):
        state = run_pipeline(_dark_image(), max_iterations=2, num_variants=2)
        case = memory.get(remember_run(state))

        assert case is not None and case.user_id == "default"
        assert len(case.signature) == 6 and all(0.0 <= v <= 1.0 for v in case.signature)
        assert any(item.startswith("underexposed@full") for item in case.defects)
        assert [a["operation"] for a in case.treatment][0] == "gamma_correct"
        assert case.final_score is not None
        assert case.recommended_variant == state["recommended_variant"]

    def test_choice_and_feedback_are_appended(self, memory: CaseMemory):
        memory._save(_case("abc"))
        assert memory.record_choice("abc", "vivid")
        assert memory.record_feedback("abc", "tối quá", [Adjustment(kind="brighter", strength=2)])
        assert memory.record_feedback(
            "abc", "da hơi vàng", [Adjustment(kind="cooler", region="face")]
        )
        case = memory.get("abc")
        assert case.chosen_variant == "vivid"
        assert case.feedback == ["brighter@full:2", "cooler@face:1"]
        assert case.feedback_texts == ["tối quá", "da hơi vàng"]

    def test_unknown_case_is_reported(self, memory: CaseMemory):
        assert memory.get("nope") is None
        assert not memory.record_choice("nope", "vivid")
        assert not memory.record_feedback("nope", "x", [])

    def test_records_persist_across_instances(self, tmp_path: Path):
        path = tmp_path / "cases.sqlite"
        CaseMemory(path)._save(_case("persist"))
        assert CaseMemory(path).get("persist") is not None

    def test_remember_run_is_a_no_op_when_disabled(self):
        assert remember_run({"history": []}) is None

    def test_remember_run_swallows_storage_errors(self, memory: CaseMemory):
        with patch.object(memory, "record_run", side_effect=RuntimeError("disk full")):
            assert remember_run({"history": []}) is None


# ============================================================
# Truy xuất và sở thích
# ============================================================
class TestRetrieval:
    def test_signature_is_normalised(self):
        signature = metric_signature(
            {
                "brightness_mean": 255.0,
                "contrast_std": 300.0,
                "noise_variance": 15.0,
                "sharpness_laplacian_var": 0.0,
                "histogram_stats": {"highlight_clip_ratio": 0.2},
            }
        )
        assert signature == [1.0, 1.0, 0.5, 0.0, 0.2, 0.0]
        assert metric_signature({}) == [0.0] * 6

    def test_similar_ranks_by_metrics_defects_and_scene(self, memory: CaseMemory):
        metrics = {"brightness_mean": 40.0, "contrast_std": 20.0}
        here = metric_signature(metrics)
        memory._save(
            _case("same", signature=here, defects=["underexposed@full:3"], scene_type="food")
        )
        memory._save(_case("other-scene", signature=here, defects=["underexposed@full:3"]))
        memory._save(_case("far", signature=[1.0] * 6, defects=["noise@full:2"]))
        diagnosis = DiagnosisReport(
            scene_type="food", defects=[Defect(type="underexposed", severity=3)]
        )

        found = memory.similar(diagnosis, metrics)
        assert [case.id for case, _ in found] == ["same", "other-scene"]
        assert all(distance <= MAX_DISTANCE for _, distance in found)
        assert [c.id for c, _ in memory.similar(diagnosis, metrics, exclude="same")] == [
            "other-scene"
        ]

    def test_cases_are_separated_by_user(self, memory: CaseMemory):
        memory._save(_case("mine", user_id="an"))
        memory._save(_case("theirs", user_id="binh"))
        assert [c.id for c in memory.cases("an")] == ["mine"]
        assert {c.id for c in memory.cases()} == {"mine", "theirs"}

    @pytest.mark.parametrize(
        "choices,expected",
        [
            (["vivid", "vivid"], None),  # chưa đủ MIN_CHOICES
            (["vivid", "vivid", "natural"], "vivid"),
            (["vivid", "natural", "balanced"], None),  # không phong cách nào đủ tỉ lệ
        ],
    )
    def test_preferred_style(self, memory: CaseMemory, choices, expected):
        for index, choice in enumerate(choices):
            memory._save(_case(f"c{index}", chosen_variant=choice))
        assert memory.preferred_style() == expected

    def test_stats(self, memory: CaseMemory):
        memory._save(_case("a", chosen_variant="vivid", feedback=["brighter@full:2"]))
        memory._save(_case("b", feedback=["brighter@face:1", "cooler@full:1"]))
        stats = memory.stats()
        assert stats["cases"] == 2
        assert stats["choices"] == {"vivid": 1}
        assert stats["feedback"] == {"brighter": 2, "cooler": 1}

    def test_experience_prompt(self):
        case = _case(
            "abcdef123",
            scene_type="food",
            defects=["underexposed@full:3"],
            treatment=[
                {
                    "operation": "gamma_correct",
                    "target_prompt": "full",
                    "parameters": {"gamma": 1.5},
                },
                {
                    "operation": "gamma_correct",
                    "target_prompt": "face",
                    "parameters": {"gamma": 1.4},
                },
            ],
            final_score=85.8,
            chosen_variant="vivid",
            feedback=["brighter@full:2"],
            feedback_texts=["tối quá"],
        )
        text = experience_prompt([(case, 0.12)])
        assert text.startswith("KINH NGHIỆM TỪ CA TƯƠNG TỰ")
        assert "[abcdef12]" in text
        assert "gamma_correct(gamma=1.5) → gamma_correct@face(gamma=1.4)" in text
        assert "chọn bản 'vivid'" in text and '"tối quá"' in text
        assert experience_prompt([]) == ""


# ============================================================
# Tích hợp vào pipeline
# ============================================================
class TestPipelineIntegration:
    def test_preference_drives_the_recommendation(self, memory: CaseMemory):
        for index in range(3):
            memory._save(_case(f"p{index}", chosen_variant="natural"))
        state = run_pipeline(_dark_image(), max_iterations=1, num_variants=3)
        assert state["recommended_variant"] == "natural"
        assert state["recommended_reason"] == "preference"

    def test_intent_style_beats_preference(self, memory: CaseMemory):
        for index in range(3):
            memory._save(_case(f"p{index}", chosen_variant="natural"))
        state = run_pipeline(
            _dark_image(), max_iterations=1, num_variants=3, intent=IntentProfile(style="vivid")
        )
        assert (state["recommended_variant"], state["recommended_reason"]) == ("vivid", "intent")

    def test_without_memory_reason_is_the_ranking_source(self):
        state = run_pipeline(_dark_image(), max_iterations=1, num_variants=2)
        assert state["recommended_reason"] == "score"

    @patch.dict("os.environ", FAKE_KEY)
    @patch("src.agent.vlm_diagnostician.genai")
    def test_similar_case_reaches_the_plan_prompt(self, mock_genai, memory: CaseMemory):
        from src.agent.perception import PERCEIVE_RESPONSE_SCHEMA

        diagnosis = {
            "scene_type": "food",
            "defects": [{"type": "underexposed", "region": "full", "severity": 3}],
            "preserve": [],
            "summary": "Tối.",
        }
        plan = {
            "reasoning": "ok",
            "actions": [
                {
                    "region_id": "full",
                    "target_prompt": "full",
                    "detected_issue": "underexposed",
                    "operation": "gamma_correct",
                    "parameters": {"gamma": 1.5},
                }
            ],
        }

        def generate_content(*, model, contents, config):
            is_perceive = config.response_json_schema == PERCEIVE_RESPONSE_SCHEMA
            return MagicMock(text=json.dumps(diagnosis if is_perceive else plan))

        models = MagicMock()
        models.generate_content.side_effect = generate_content
        mock_genai.Client.return_value.models = models

        first = run_pipeline(_dark_image(), max_iterations=1)
        case_id = remember_run(first)
        memory.record_feedback(case_id, "vẫn tối quá", [Adjustment(kind="brighter", strength=2)])
        models.generate_content.reset_mock()

        second = run_pipeline(_dark_image(12, 62), max_iterations=1)
        prompts = [
            call.kwargs["contents"][0]
            for call in models.generate_content.call_args_list
            if call.kwargs["config"].response_json_schema != PERCEIVE_RESPONSE_SCHEMA
        ]
        assert "KINH NGHIỆM TỪ CA TƯƠNG TỰ" in prompts[0]
        assert '"vẫn tối quá"' in prompts[0]
        assert f"case:{case_id[:8]}" in second["history"][0].plan.knowledge


def test_memory_module_default_is_disabled():
    """Fixture chung tắt bộ nhớ: test không bao giờ ghi vào data/memory."""
    assert memory_module.get_case_memory() is None
