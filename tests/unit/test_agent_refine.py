"""
Unit tests cho chỉnh theo góp ý (Phase 4): phân tích góp ý tiếng Việt bằng luật, VLM dự phòng
(mock), áp điều chỉnh vào phác đồ và render lại từ ảnh gốc. Chạy offline.
"""

import json
from typing import List
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.agent.refine import (
    Adjustment,
    apply_adjustments,
    parse_feedback,
    parse_feedback_rules,
    refine,
)
from src.agent.state import RegionOperation

FAKE_KEY = {"GEMINI_API_KEY": "fake-key-for-test"}


def _action(operation: str, target: str = "full", **parameters) -> RegionOperation:
    return RegionOperation(
        region_id=target,
        target_prompt=target,
        region_type="full" if target == "full" else None,
        detected_issue="test",
        operation=operation,
        parameters=parameters,
    )


def _parsed(text: str) -> List[tuple]:
    return [(a.kind, a.region, a.strength) for a in parse_feedback_rules(text)]


def _gradient_image() -> np.ndarray:
    gradient = np.tile(np.linspace(40, 200, 64), (64, 1)).astype(np.uint8)
    image = np.repeat(gradient[:, :, None], 3, axis=2)
    image[:, :, 0] = np.clip(image[:, :, 0].astype(int) + 30, 0, 255)  # hơi ám đỏ/vàng
    return image


# ============================================================
# Phân tích góp ý
# ============================================================
class TestFeedbackRules:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("sáng quá", [("darker", "full", 2)]),
            ("hơi tối", [("brighter", "full", 1)]),
            ("tối hơn chút", [("darker", "full", 1)]),
            ("mặt tối quá", [("brighter", "face", 2)]),
            ("bầu trời bị cháy", [("darker", "sky", 1)]),
            ("xanh quá", [("warmer", "full", 2)]),
            ("ảnh hơi xanh", [("warmer", "full", 1)]),
            ("gắt quá", [("less_contrast", "full", 2)]),
            ("ảnh bệt quá", [("more_contrast", "full", 2)]),
            ("màu nhạt quá", [("more_saturated", "full", 2)]),
            ("đẹp rồi", []),
        ],
    )
    def test_single_requests(self, text, expected):
        assert _parsed(text) == expected

    def test_multiple_clauses_and_regions(self):
        assert _parsed("da hơi vàng, trời gắt quá") == [("cooler", "face", 1), ("darker", "sky", 2)]
        assert _parsed("màu rực quá nhưng mặt tối quá") == [
            ("less_saturated", "full", 2),
            ("brighter", "face", 2),
        ]
        assert _parsed("ảnh còn nhiễu và hơi mờ") == [
            ("less_noise", "full", 1),
            ("sharper", "full", 1),
        ]

    def test_repeated_request_keeps_the_stronger_strength(self):
        assert _parsed("nét quá, có viền sáng") == [("softer", "full", 2)]

    def test_rules_do_not_call_the_vlm(self):
        with patch("src.agent.refine.call_gemini_json") as fake:
            assert parse_feedback("tối quá") == ([Adjustment(kind="brighter", strength=2)], "rules")
        fake.assert_not_called()

    def test_unknown_feedback_without_key(self):
        assert parse_feedback("thêm chút hồn cho ảnh") == ([], "none")

    @patch.dict("os.environ", FAKE_KEY)
    @patch("src.agent.vlm_diagnostician.genai")
    def test_vlm_parses_what_rules_miss(self, mock_genai):
        response = {
            "adjustments": [
                {"kind": "more_contrast", "region": "Khuôn mặt", "strength": 5},
                {"kind": "teleport"},
            ]
        }
        mock_genai.Client.return_value.models.generate_content.return_value = MagicMock(
            text=json.dumps(response)
        )
        adjustments, source = parse_feedback("thêm chút hồn cho ảnh")
        assert source == "vlm"
        assert adjustments == [Adjustment(kind="more_contrast", region="face", strength=2)]

    @patch.dict("os.environ", FAKE_KEY)
    @patch("src.agent.vlm_diagnostician.genai")
    def test_vlm_failure_returns_nothing(self, mock_genai):
        mock_genai.Client.return_value.models.generate_content.side_effect = RuntimeError("503")
        assert parse_feedback("thêm chút hồn cho ảnh") == ([], "none")


# ============================================================
# Áp điều chỉnh vào phác đồ
# ============================================================
class TestApplyAdjustments:
    RECIPE = [_action("gamma_correct", gamma=1.4), _action("clahe", clip_limit=3.0)]

    def test_added_operations_and_order(self):
        actions, notes = apply_adjustments(
            self.RECIPE,
            [
                Adjustment(kind="less_noise", strength=2),
                Adjustment(kind="brighter", region="face"),
                Adjustment(kind="cooler", strength=2),
            ],
        )
        ops = [(a.operation, a.target_prompt, a.parameters) for a in actions]
        assert ops[0] == ("denoise", "full", {"method": "bilateral", "strength": 1.5})
        assert ops[1:3] == [(a.operation, a.target_prompt, a.parameters) for a in self.RECIPE]
        assert ops[3] == ("gamma_correct", "face", {"gamma": 1.15})
        assert ops[4][0] == "color_correct" and ops[4][2]["temperature_shift"] == -0.3
        assert actions[3].feather_radius == 30
        assert notes == [
            "khử nhiễu trên toàn ảnh",
            "sáng hơn trên vùng 'face'",
            "lạnh hơn trên toàn ảnh",
        ]

    def test_reducing_scales_existing_operation(self):
        actions, notes = apply_adjustments(self.RECIPE, [Adjustment(kind="less_contrast")])
        assert actions[1].parameters["clip_limit"] == pytest.approx(1 + 0.6 * 2.0)
        assert notes == ["giảm tương phản trên toàn ảnh"]

    def test_strong_softer_removes_sharpening(self):
        recipe = [*self.RECIPE, _action("sharpen", method="unsharp_mask", amount=1.0)]
        actions, _ = apply_adjustments(recipe, [Adjustment(kind="softer", strength=2)])
        assert "sharpen" not in [a.operation for a in actions]

    def test_nothing_to_reduce_is_reported(self):
        actions, notes = apply_adjustments(
            [_action("gamma_correct", gamma=1.2)], [Adjustment(kind="softer")]
        )
        assert len(actions) == 1
        assert notes[0].startswith("không thể bớt nét")

    def test_recipe_is_not_mutated(self):
        recipe = [_action("clahe", clip_limit=3.0)]
        apply_adjustments(recipe, [Adjustment(kind="less_contrast")])
        assert recipe[0].parameters == {"clip_limit": 3.0}


# ============================================================
# Render lại theo góp ý
# ============================================================
class TestRefine:
    @staticmethod
    def _gray_mean(image: np.ndarray) -> float:
        return float(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).mean())

    @staticmethod
    def _red_blue(image: np.ndarray) -> float:
        return float(image[:, :, 0].mean() - image[:, :, 2].mean())

    def test_feedback_moves_the_image_in_the_requested_direction(self):
        image = _gradient_image()
        recipe = [_action("gamma_correct", gamma=1.2)]
        base = refine(image, recipe, "đẹp rồi")
        brighter = refine(image, recipe, "tối quá")
        warmer_down = refine(image, recipe, "vàng quá")

        assert base.adjustments == [] and base.source == "none"
        assert self._gray_mean(brighter.image) > self._gray_mean(base.image) + 5
        assert self._red_blue(warmer_down.image) < self._red_blue(base.image) - 3

    def test_refine_renders_from_the_original(self):
        """Góp ý hai lần liên tiếp = áp cả hai lên phác đồ, render một lần từ ảnh gốc."""
        image = _gradient_image()
        first = refine(image, [], "hơi tối")
        second = refine(image, first.actions, "hơi tối")
        assert [a.parameters["gamma"] for a in second.actions] == [1.15, 1.15]
        assert second.image.shape == image.shape
        assert second.quality_score is not None
