"""
Unit tests cho sinh phiên bản (Phase 3): phong cách tất định trên phác đồ, preserve guard,
render song song bằng Send, bỏ trùng, xếp hạng (điểm Module 1 / VLM critic mock).
Chạy offline; Gemini được mock.
"""

import json
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.agent.graph import run_pipeline
from src.agent.imaging import PREVIEW_MAX_SIDE
from src.agent.state import (
    Defect,
    DiagnosisReport,
    HistoryItem,
    PreserveItem,
    RegionOperation,
    TreatmentPlan,
    Variant,
)
from src.agent.variants import (
    STYLES,
    _deduplicate,
    _scaled_parameters,
    _score_ranking,
    is_noisy,
    merged_preserve,
    rank_variants,
    styles_for,
    stylize,
    treatment_recipe,
)

FAKE_KEY = {"GEMINI_API_KEY": "fake-key-for-test"}


# ============================================================
# Helpers
# ============================================================
def _action(operation: str, target: str = "full", **parameters) -> RegionOperation:
    return RegionOperation(
        region_id=target,
        target_prompt=target,
        region_type="full" if target == "full" else None,
        detected_issue="test",
        operation=operation,
        parameters=parameters,
    )


def _history_item(
    iteration: int, actions: List[RegionOperation], rolled_back: bool = False, **diagnosis
) -> HistoryItem:
    return HistoryItem(
        iteration=iteration,
        plan=TreatmentPlan(iteration=iteration, reasoning="", actions=actions),
        metrics_before={},
        metrics_after={},
        eval_score={},
        decision="RE_PROCESS",
        rolled_back=rolled_back,
        diagnosis=DiagnosisReport(**diagnosis) if diagnosis else None,
    )


def _variant(variant_id: str, score: float, improved: bool = True, fill: int = 0) -> Variant:
    return Variant(
        id=variant_id,
        label=variant_id,
        image=np.full((4, 4, 3), fill, dtype=np.uint8),
        quality_score=score,
        quality_improved=improved,
    )


def _ops(actions: List[RegionOperation]) -> List[tuple]:
    return [(a.operation, a.target_prompt, a.parameters) for a in actions]


RECIPE = [
    _action("denoise", method="bilateral", strength=1.0),
    _action("gamma_correct", gamma=1.5),
    _action("gamma_correct", target="face", gamma=1.4),
    _action("sharpen", method="unsharp_mask", amount=0.8),
]


# ============================================================
# Phong cách
# ============================================================
class TestStyles:
    @pytest.mark.parametrize(
        "operation,parameters,k,expected",
        [
            ("gamma_correct", {"gamma": 1.5}, 0.7, {"gamma": 1.35}),
            ("gamma_correct", {"gamma": 0.8}, 0.5, {"gamma": 0.9}),
            ("clahe", {"clip_limit": 3.0}, 0.5, {"clip_limit": 2.0}),
            (
                "denoise",
                {"method": "nlm", "strength": 1.0},
                0.8,
                {"method": "nlm", "strength": 0.8},
            ),
            (
                "color_correct",
                {"saturation_scale": 1.2, "temperature_shift": -0.4},
                0.5,
                {"saturation_scale": 1.1, "temperature_shift": -0.2},
            ),
            ("gamma_correct", {"gamma": 2.4}, 1.5, {"gamma": 2.5}),  # kẹp theo PARAMETER_BOUNDS
        ],
    )
    def test_scaled_parameters(self, operation, parameters, k, expected):
        assert _scaled_parameters(operation, parameters, k) == pytest.approx(expected)

    def test_balanced_keeps_the_recipe(self):
        assert _ops(stylize(RECIPE, STYLES["balanced"])) == _ops(RECIPE)

    def test_natural_softens_and_drops_sharpening(self):
        actions = stylize(RECIPE, STYLES["natural"])
        assert [a.operation for a in actions] == ["denoise", "gamma_correct", "gamma_correct"]
        assert actions[1].parameters["gamma"] == pytest.approx(1.35)
        assert actions[2].target_prompt == "face"

    def test_vivid_adds_contrast_and_saturation(self):
        actions = stylize(RECIPE, STYLES["vivid"])
        added = _ops(actions[len(RECIPE) :])
        assert added == [
            ("clahe", "full", {"clip_limit": 1.5}),
            ("color_correct", "full", {"saturation_scale": 1.12, "temperature_shift": 0.0}),
        ]
        assert actions[3].parameters["amount"] == pytest.approx(0.96)

    def test_vivid_skips_clahe_on_noisy_images(self):
        actions = stylize(RECIPE, STYLES["vivid"], noisy=True)
        assert "clahe" not in [a.operation for a in actions]

    def test_vivid_scales_existing_operations_instead_of_adding(self):
        recipe = [_action("clahe", clip_limit=2.0)]
        actions = stylize(recipe, STYLES["vivid"])
        assert [a.operation for a in actions].count("clahe") == 1
        assert actions[0].parameters["clip_limit"] == pytest.approx(2.4)

    def test_preserve_blocks_style_additions(self):
        preserve = [PreserveItem(aspect="muted_colors"), PreserveItem(aspect="low_key")]
        actions = stylize([], STYLES["vivid"], preserve)
        assert actions == []

    def test_recipe_is_not_mutated(self):
        recipe = [_action("gamma_correct", gamma=1.5)]
        stylize(recipe, STYLES["natural"])
        assert recipe[0].parameters == {"gamma": 1.5}

    @pytest.mark.parametrize(
        "n,expected",
        [
            (0, []),
            (1, ["balanced"]),
            (2, ["balanced", "natural"]),
            (3, ["balanced", "natural", "vivid"]),
            (9, ["balanced", "natural", "vivid"]),  # tối đa MAX_VARIANTS
        ],
    )
    def test_styles_for(self, n, expected):
        assert styles_for(n) == expected


class TestRecipe:
    def test_rolled_back_iterations_are_excluded(self):
        history = [
            _history_item(1, [_action("gamma_correct", gamma=1.3)]),
            _history_item(2, [_action("clahe", clip_limit=2.0)], rolled_back=True),
            _history_item(3, [_action("denoise", method="bilateral", strength=1.0)]),
        ]
        assert [a.operation for a in treatment_recipe(history)] == ["gamma_correct", "denoise"]

    def test_preserve_is_merged_across_iterations(self):
        history = [
            _history_item(1, [], preserve=[PreserveItem(aspect="warm_tone")]),
            _history_item(
                2,
                [],
                preserve=[PreserveItem(aspect="warm_tone"), PreserveItem(aspect="film_grain")],
            ),
        ]
        assert [p.aspect for p in merged_preserve(history)] == ["warm_tone", "film_grain"]

    def test_is_noisy(self):
        noisy = [_history_item(1, [], defects=[Defect(type="noise", severity=2)])]
        clean = [_history_item(1, [], defects=[Defect(type="noise", severity=1)])]
        assert is_noisy(noisy) and not is_noisy(clean)


# ============================================================
# Bỏ trùng và xếp hạng
# ============================================================
class TestRanking:
    def test_identical_images_are_deduplicated_in_style_order(self):
        unique = _deduplicate(
            [_variant("vivid", 80, fill=5), _variant("natural", 80), _variant("balanced", 80)]
        )
        assert [v.id for v in unique] == ["balanced", "vivid"]

    def test_score_ranking_prefers_balanced_within_tolerance(self):
        ranking = _score_ranking(
            [_variant("vivid", 86.3), _variant("balanced", 86.0), _variant("natural", 80.0)]
        )
        assert ranking == ["balanced", "vivid", "natural"]

    def test_clearly_higher_score_wins(self):
        assert _score_ranking([_variant("vivid", 90), _variant("balanced", 85)])[0] == "vivid"

    def test_variants_that_did_not_improve_rank_last(self):
        ranking = _score_ranking(
            [_variant("balanced", 95, improved=False), _variant("natural", 70, fill=1)]
        )
        assert ranking == ["natural", "balanced"]

    def test_rank_without_key_uses_scores(self):
        variants = [_variant("balanced", 85), _variant("vivid", 95, fill=1)]
        ranked, recommended, source = rank_variants(np.zeros((4, 4, 3), np.uint8), variants)
        assert (recommended, source) == ("vivid", "score")
        assert [(v.id, v.rank) for v in ranked] == [("vivid", 1), ("balanced", 2)]

    def test_no_variants(self):
        assert rank_variants(np.zeros((4, 4, 3), np.uint8), []) == ([], None, "score")


def _mock_critic(mock_genai: MagicMock, response: dict) -> MagicMock:
    models = MagicMock()
    models.generate_content.return_value = MagicMock(text=json.dumps(response))
    mock_genai.Client.return_value.models = models
    return models


class TestCritic:
    VARIANTS = [
        _variant("balanced", 90),
        _variant("natural", 85, fill=1),
        _variant("vivid", 88, fill=2),
    ]

    @patch.dict("os.environ", FAKE_KEY)
    @patch("src.agent.vlm_diagnostician.genai")
    def test_critic_ranking_and_notes(self, mock_genai):
        models = _mock_critic(
            mock_genai,
            {
                "ranking": ["vivid", "balanced", "natural"],
                "recommended": "vivid",
                "notes": [{"id": "vivid", "note": "Màu nổi, không cháy."}],
            },
        )
        diagnosis = DiagnosisReport(summary="Ảnh tối.", preserve=[PreserveItem(aspect="warm_tone")])
        ranked, recommended, source = rank_variants(
            np.zeros((4, 4, 3), np.uint8), self.VARIANTS, diagnosis
        )
        assert source == "critic" and recommended == "vivid"
        assert [v.id for v in ranked] == ["vivid", "balanced", "natural"]
        assert ranked[0].critic_note == "Màu nổi, không cháy."
        contents = models.generate_content.call_args.kwargs["contents"]
        assert "warm_tone@full" in contents[0]
        assert len([c for c in contents if not isinstance(c, str)]) == 4  # gốc + 3 phiên bản

    @patch.dict("os.environ", FAKE_KEY)
    @patch("src.agent.vlm_diagnostician.genai")
    def test_partial_or_unknown_ids_are_completed_by_score(self, mock_genai):
        _mock_critic(
            mock_genai,
            {"ranking": ["natural", "ghost", "natural"], "recommended": "ghost", "notes": []},
        )
        ranked, recommended, source = rank_variants(np.zeros((4, 4, 3), np.uint8), self.VARIANTS)
        assert source == "critic"
        assert [v.id for v in ranked] == ["natural", "balanced", "vivid"]
        assert recommended == "natural"

    @patch.dict("os.environ", FAKE_KEY)
    @patch("src.agent.vlm_diagnostician.genai")
    def test_critic_failure_falls_back_to_scores(self, mock_genai):
        mock_genai.Client.return_value.models.generate_content.side_effect = RuntimeError("503")
        ranked, recommended, source = rank_variants(np.zeros((4, 4, 3), np.uint8), self.VARIANTS)
        assert (source, recommended) == ("score", "balanced")

    @patch.dict("os.environ", FAKE_KEY)
    @patch("src.agent.vlm_diagnostician.genai")
    def test_single_variant_skips_the_critic(self, mock_genai):
        rank_variants(np.zeros((4, 4, 3), np.uint8), [_variant("balanced", 90)])
        mock_genai.Client.assert_not_called()


# ============================================================
# Pipeline
# ============================================================
class TestPipelineVariants:
    @staticmethod
    def _dark_image(size: tuple = (64, 64)) -> np.ndarray:
        gradient = np.tile(np.linspace(10, 60, size[1], dtype=np.float64), (size[0], 1))
        return np.repeat(gradient[:, :, None], 3, axis=2).astype(np.uint8)

    def test_default_pipeline_has_no_variants(self):
        state = run_pipeline(self._dark_image(), max_iterations=2)
        assert state["variants"] == []
        assert state["recommended_variant"] is None

    def test_three_variants_ranked_once(self):
        image = self._dark_image()
        state = run_pipeline(image, max_iterations=2, num_variants=3)

        ids = [v.id for v in state["variants"]]
        assert sorted(ids) == ["balanced", "natural", "vivid"]
        assert [v.rank for v in state["variants"]] == [1, 2, 3]
        assert len(state["variant_candidates"]) == 3  # mỗi phong cách đúng một nhánh Send
        assert state["recommended_variant"] in ids
        assert state["variant_ranking_source"] == "score"
        balanced = next(v for v in state["variants"] if v.id == "balanced")
        # Ảnh nhỏ hơn preview: phiên bản cân bằng chính là kết quả của vòng lặp
        assert np.array_equal(balanced.image, state["current_image"])

    def test_variants_are_rendered_on_a_preview(self):
        image = self._dark_image((300, 2048))
        state = run_pipeline(image, max_iterations=1, num_variants=2)
        assert {v.image.shape for v in state["variants"]} == {(150, PREVIEW_MAX_SIDE, 3)}
        assert state["current_image"].shape == image.shape
