"""
Unit tests cho giai đoạn Perceive (Phase 1): chẩn đoán có cấu trúc, đo số liệu theo vùng,
lỗi suy ra từ số đo, preserve guard và lập kế hoạch từ chẩn đoán.
Chạy offline: Gemini được mock, Module 2 được patch khi cần mask vùng.
"""

import json
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.agent.graph import plan_treatment, run_pipeline
from src.agent.knowledge import diagnose_from_metrics
from src.agent.perception import (
    MAX_SUBJECTS,
    PERCEIVE_RESPONSE_SCHEMA,
    _drop_preserved_defects,
    _measured_defects,
    _report_from_vlm_json,
    compute_region_metrics,
    perceive,
)
from src.agent.planner import apply_preserve_guard
from src.agent.state import (
    Defect,
    DiagnosisReport,
    PreserveItem,
    RegionMetrics,
    RegionOperation,
    TreatmentPlan,
)
from src.agent.vlm_diagnostician import diagnose_and_plan

FAKE_KEY = {"GEMINI_API_KEY": "fake-key-for-test"}


# ============================================================
# Helpers
# ============================================================
def _backlit_image() -> np.ndarray:
    """Nền sáng 210, ô giữa (chủ thể) tối 45 — mô phỏng chân dung ngược sáng."""
    image = np.full((100, 100, 3), 210, dtype=np.uint8)
    image[30:70, 30:70] = 45
    return image


def _center_mask() -> np.ndarray:
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[30:70, 30:70] = 1.0
    return mask


def _region_metrics(region: str, mean: float, vs_rest: Optional[float]) -> RegionMetrics:
    level = "underexposed" if mean < 70 else "overexposed" if mean > 185 else "normal"
    return RegionMetrics(
        region=region,
        area_ratio=0.2,
        brightness_mean=mean,
        brightness_std=5.0,
        brightness_level=level,
        highlight_clip_ratio=0.3 if mean > 240 else 0.0,
        shadow_clip_ratio=0.0,
        brightness_vs_rest=vs_rest,
    )


def _action(operation: str, target: str = "full", **parameters: Any) -> RegionOperation:
    return RegionOperation(
        region_id=target,
        target_prompt=target,
        region_type="full" if target == "full" else None,
        detected_issue="test",
        operation=operation,
        parameters=parameters,
    )


def _plan(*actions: RegionOperation) -> TreatmentPlan:
    return TreatmentPlan(iteration=1, reasoning="kế hoạch", actions=list(actions))


def _mock_models(mock_genai: MagicMock, *texts: str) -> MagicMock:
    models = MagicMock()
    models.generate_content.side_effect = [MagicMock(text=text) for text in texts]
    mock_genai.Client.return_value.models = models
    return models


VLM_DIAGNOSIS = {
    "scene_type": "portrait",
    "lighting": "ngược sáng",
    "subjects": ["face"],
    "defects": [{"type": "noise", "region": "full", "severity": 2, "evidence": "hạt nhiễu"}],
    "preserve": [{"aspect": "warm_tone", "region": "full", "reason": "nắng chiều"}],
    "summary": "Chân dung ngược sáng, nhiễu nhẹ.",
}


# ============================================================
# Đo số liệu theo vùng
# ============================================================
class TestComputeRegionMetrics:
    def test_backlit_region_is_dark_relative_to_rest(self):
        metrics = compute_region_metrics(_backlit_image(), _center_mask(), "face", "test")
        assert metrics is not None
        assert metrics.brightness_mean == pytest.approx(45, abs=1)
        assert metrics.brightness_level == "underexposed"
        assert metrics.area_ratio == pytest.approx(0.16, abs=1e-3)
        assert metrics.brightness_vs_rest == pytest.approx(45 - 210, abs=1)
        assert metrics.backend == "test"

    def test_soft_mask_weights_pixels(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        image[:, 5:] = 200
        mask = np.full((10, 10), 0.5, dtype=np.float32)
        mask[:, 5:] = 1.0  # cột sáng có trọng số gấp đôi
        metrics = compute_region_metrics(image, mask, "x")
        assert metrics.brightness_mean == pytest.approx(200 * 2 / 3, abs=1)

    def test_full_coverage_has_no_rest_comparison(self):
        mask = np.ones((100, 100), dtype=np.float32)
        metrics = compute_region_metrics(_backlit_image(), mask, "everything")
        assert metrics.brightness_vs_rest is None

    def test_empty_mask_returns_none(self):
        mask = np.zeros((100, 100), dtype=np.float32)
        assert compute_region_metrics(_backlit_image(), mask, "nothing") is None

    def test_grayscale_and_channel_mask(self):
        gray = _backlit_image()[:, :, 0]
        metrics = compute_region_metrics(gray, _center_mask()[:, :, None], "face")
        assert metrics.brightness_mean == pytest.approx(45, abs=1)

    def test_clipping_ratios(self):
        image = np.full((10, 10, 3), 255, dtype=np.uint8)
        image[:5] = 0
        metrics = compute_region_metrics(image, np.ones((10, 10), np.float32), "x")
        assert metrics.highlight_clip_ratio == pytest.approx(0.5)
        assert metrics.shadow_clip_ratio == pytest.approx(0.5)


# ============================================================
# Lỗi suy ra từ số đo
# ============================================================
class TestMeasuredDefects:
    def test_backlit_face_is_added(self):
        report = DiagnosisReport(region_metrics={"face": _region_metrics("face", 60, -120)})
        [defect] = _measured_defects(report)
        assert defect.type == "backlit_subject"
        assert defect.region == "face"
        assert defect.severity == 2
        assert defect.origin == "measured"

    def test_very_dark_region_is_severe(self):
        report = DiagnosisReport(region_metrics={"face": _region_metrics("face", 30, None)})
        [defect] = _measured_defects(report)
        assert defect.type == "underexposed"
        assert defect.severity == 3

    def test_normal_but_much_darker_than_rest_is_backlit(self):
        report = DiagnosisReport(region_metrics={"face": _region_metrics("face", 90, -100)})
        assert [d.type for d in _measured_defects(report)] == ["backlit_subject"]

    def test_existing_defect_is_not_duplicated(self):
        report = DiagnosisReport(
            defects=[Defect(type="underexposed", region="face", severity=1)],
            region_metrics={"face": _region_metrics("face", 60, -120)},
        )
        assert _measured_defects(report) == []

    @pytest.mark.parametrize("aspect,region", [("silhouette", "face"), ("low_key", "full")])
    def test_intentional_darkness_is_respected(self, aspect: str, region: str):
        report = DiagnosisReport(
            preserve=[PreserveItem(aspect=aspect, region=region)],
            region_metrics={"face": _region_metrics("face", 60, -120)},
        )
        assert _measured_defects(report) == []

    def test_blown_sky_is_added(self):
        report = DiagnosisReport(region_metrics={"sky": _region_metrics("sky", 245, 80)})
        [defect] = _measured_defects(report)
        assert (defect.type, defect.region) == ("overexposed", "sky")

    def test_naturally_bright_or_dark_regions_are_not_defects(self):
        """Trời sáng, mèo đen: không phải lỗi nếu không cháy/bệt và không phải khuôn mặt."""
        report = DiagnosisReport(
            region_metrics={
                "sky": _region_metrics("sky", 205, 60),  # sáng nhưng không cháy
                "cat": _region_metrics("cat", 50, -10),  # tối nhưng không chênh sáng
            }
        )
        assert _measured_defects(report) == []

    def test_crushed_shadows_in_any_region_are_defects(self):
        crushed = _region_metrics("cat", 30, -10).model_copy(update={"shadow_clip_ratio": 0.4})
        [defect] = _measured_defects(DiagnosisReport(region_metrics={"cat": crushed}))
        assert (defect.type, defect.region) == ("underexposed", "cat")

    def test_high_key_preserve_skips_bright_region(self):
        report = DiagnosisReport(
            preserve=[PreserveItem(aspect="high_key")],
            region_metrics={"sky": _region_metrics("sky", 245, 80)},
        )
        assert _measured_defects(report) == []


# ============================================================
# Chẩn đoán rule-based và parse JSON của VLM
# ============================================================
class TestPreservedDefects:
    def test_defect_conflicting_with_preserve_is_dropped(self):
        report = DiagnosisReport(
            defects=[
                Defect(type="color_cast_warm", severity=1),
                Defect(type="noise", region="face", severity=2),
                Defect(type="underexposed", region="sky", severity=2),
            ],
            preserve=[
                PreserveItem(aspect="warm_tone"),
                PreserveItem(aspect="film_grain", region="face"),
                PreserveItem(aspect="low_key", region="person"),
            ],
        )
        kept = _drop_preserved_defects(report)
        assert [(d.type, d.region) for d in kept] == [("underexposed", "sky")]

    @patch.dict("os.environ", FAKE_KEY)
    @patch("src.agent.vlm_diagnostician.genai")
    def test_perceive_resolves_vlm_contradiction(self, mock_genai):
        data = dict(VLM_DIAGNOSIS)
        data["defects"] = [{"type": "color_cast_warm", "region": "full", "severity": 2}]
        _mock_models(mock_genai, json.dumps(data))
        with patch("src.agent.perception.resolve_region_mask", side_effect=_fake_resolver({})):
            report = perceive(_backlit_image(), {})
        assert report.defects == []
        assert [p.aspect for p in report.preserve] == ["warm_tone"]


class TestRuleBasedReport:
    def test_defects_and_severities_from_metrics(self):
        report = diagnose_from_metrics(
            {
                "noise_level": "severe",
                "brightness_level": "underexposed",
                "brightness_mean": 30.0,
                "contrast_level": "low",
                "contrast_std": 20.0,
                "blur_level": "mild_blur",
                "color_cast": "greenish",
            },
            iteration=2,
            source="rule_based",
        )
        found = {d.type: d.severity for d in report.defects}
        assert found == {
            "noise": 3,
            "underexposed": 3,
            "low_contrast": 2,
            "blur": 1,
            "color_cast_green": 1,
        }
        assert all(d.origin == "rule" and d.region == "full" for d in report.defects)
        assert report.subjects == ["face"]
        assert report.iteration == 2
        assert report.source == "rule_based"

    def test_clean_metrics_have_no_defects(self):
        report = diagnose_from_metrics({"noise_level": "clean"}, 1, "rule_based")
        assert report.defects == []
        assert "không có lỗi" in report.summary


class TestReportFromVlmJson:
    def test_valid_diagnosis(self):
        report = _report_from_vlm_json(VLM_DIAGNOSIS, iteration=1)
        assert report.scene_type == "portrait"
        assert report.lighting == "ngược sáng"
        assert report.subjects == ["face"]
        assert [(d.type, d.severity, d.origin) for d in report.defects] == [("noise", 2, "vlm")]
        assert [(p.aspect, p.region) for p in report.preserve] == [("warm_tone", "full")]
        assert report.source == "vlm"

    def test_invalid_items_are_dropped_individually(self):
        data = {
            "scene_type": "underwater",
            "defects": [
                {"type": "haze", "region": "full", "severity": 2},
                "not an object",
                {"type": "blur", "region": " Full ", "severity": 9},
                {"type": "noise", "region": "", "severity": "x"},
            ],
            "preserve": [{"aspect": "sepia"}, {"aspect": "film_grain", "region": None}],
            "summary": 42,
        }
        report = _report_from_vlm_json(data, iteration=1)
        assert report.scene_type == "other"
        assert [(d.type, d.region, d.severity) for d in report.defects] == [("blur", "full", 3)]
        assert [(p.aspect, p.region) for p in report.preserve] == [("film_grain", "full")]
        assert report.summary == ""

    def test_subjects_include_defect_regions_deduplicated_and_capped(self):
        data = {
            "subjects": ["Face", "sky", "full", "face"],
            "defects": [
                {"type": "overexposed", "region": "sky", "severity": 2},
                {"type": "underexposed", "region": "dog", "severity": 1},
                {"type": "noise", "region": "car", "severity": 1},
                {"type": "noise", "region": "tree", "severity": 1},
            ],
            "preserve": [],
        }
        report = _report_from_vlm_json(data, iteration=1)
        assert report.subjects == ["face", "sky", "dog", "car"][:MAX_SUBJECTS]

    def test_all_invalid_defects_raise(self):
        """Mọi lỗi đều sai → không được thành chẩn đoán rỗng (= SHIP âm thầm)."""
        with pytest.raises(ValueError, match="all 2 VLM defects were invalid"):
            _report_from_vlm_json(
                {"defects": [{"type": "haze"}, {"type": "noise", "severity": "x"}]}, 1
            )

    def test_empty_defect_list_is_a_clean_diagnosis(self):
        assert _report_from_vlm_json({"defects": [], "preserve": []}, 1).defects == []

    def test_region_synonyms_are_normalized(self):
        report = _report_from_vlm_json(
            {"defects": [{"type": "underexposed", "region": "Khuôn mặt", "severity": 2}]}, 1
        )
        assert report.defects[0].region == "face"
        assert report.subjects == ["face"]

    @pytest.mark.parametrize("data", [[], {"preserve": []}, {"defects": "none"}])
    def test_missing_defects_list_raises(self, data: Any):
        with pytest.raises(ValueError):
            _report_from_vlm_json(data, iteration=1)


# ============================================================
# perceive(): VLM, fallback và đo vùng
# ============================================================
def _fake_resolver(found_regions: dict):
    """resolve_region_mask giả: trả mask cho các vùng trong found_regions."""

    def resolve(image: np.ndarray, target_prompt: str, feather_radius: int = 15):
        mask = found_regions.get(target_prompt)
        if mask is None:
            return False, None, None
        return True, mask, "test"

    return resolve


class TestPerceive:
    def test_offline_measures_face_and_adds_backlit_defect(self):
        with patch(
            "src.agent.perception.resolve_region_mask",
            side_effect=_fake_resolver({"face": _center_mask()}),
        ):
            report = perceive(_backlit_image(), {"brightness_level": "normal"})

        assert report.source == "rule_based"
        assert set(report.region_metrics) == {"face"}
        assert [(d.type, d.region, d.origin) for d in report.defects] == [
            ("backlit_subject", "face", "measured")
        ]

    def test_unresolved_region_is_skipped(self):
        with patch("src.agent.perception.resolve_region_mask", side_effect=_fake_resolver({})):
            report = perceive(_backlit_image(), {})
        assert report.region_metrics == {}

    def test_measurement_error_does_not_break_perception(self):
        with patch("src.agent.perception.resolve_region_mask", side_effect=RuntimeError("boom")):
            report = perceive(_backlit_image(), {"noise_level": "severe"})
        assert report.region_metrics == {}
        assert [d.type for d in report.defects] == ["noise"]

    @patch.dict("os.environ", FAKE_KEY)
    @patch("src.agent.vlm_diagnostician.genai")
    def test_vlm_diagnosis_uses_perceive_schema(self, mock_genai):
        models = _mock_models(mock_genai, json.dumps(VLM_DIAGNOSIS))
        with patch("src.agent.perception.resolve_region_mask", side_effect=_fake_resolver({})):
            report = perceive(_backlit_image(), {"noise_level": "medium"})

        assert report.source == "vlm"
        assert report.scene_type == "portrait"
        config = models.generate_content.call_args.kwargs["config"]
        assert config.response_json_schema == PERCEIVE_RESPONSE_SCHEMA

    @patch.dict("os.environ", FAKE_KEY)
    @patch("src.agent.vlm_diagnostician.genai")
    def test_vlm_failure_falls_back_to_rules(self, mock_genai):
        _mock_models(mock_genai, "không phải JSON")
        with patch("src.agent.perception.resolve_region_mask", side_effect=_fake_resolver({})):
            report = perceive(_backlit_image(), {"noise_level": "severe"})

        assert report.source == "vlm_fallback"
        assert [d.type for d in report.defects] == ["noise"]

    @patch.dict("os.environ", FAKE_KEY)
    @patch("src.agent.vlm_diagnostician.genai")
    def test_later_iteration_sends_original_image(self, mock_genai):
        models = _mock_models(mock_genai, json.dumps(VLM_DIAGNOSIS))
        with patch("src.agent.perception.resolve_region_mask", side_effect=_fake_resolver({})):
            perceive(
                _backlit_image(),
                {},
                iteration=2,
                original_image=np.zeros((100, 100, 3), dtype=np.uint8),
            )

        contents = models.generate_content.call_args.kwargs["contents"]
        assert len([c for c in contents if not isinstance(c, str)]) == 2


# ============================================================
# Preserve guard
# ============================================================
class TestPreserveGuard:
    def test_no_preserve_keeps_plan(self):
        plan = _plan(_action("gamma_correct", gamma=1.4))
        assert apply_preserve_guard(plan, []).actions == plan.actions

    def test_warm_tone_neutralizes_cooling_but_keeps_saturation(self):
        plan = _plan(_action("color_correct", saturation_scale=1.2, temperature_shift=-0.4))
        guarded = apply_preserve_guard(plan, [PreserveItem(aspect="warm_tone")])
        [action] = guarded.actions
        assert action.parameters == {"saturation_scale": 1.2, "temperature_shift": 0.0}
        assert "warm_tone" in guarded.reasoning

    def test_action_that_becomes_neutral_is_dropped(self):
        plan = _plan(_action("color_correct", saturation_scale=1.0, temperature_shift=-0.4))
        assert apply_preserve_guard(plan, [PreserveItem(aspect="warm_tone")]).actions == []

    def test_film_grain_blocks_denoise_on_every_region(self):
        plan = _plan(
            _action("denoise", method="bilateral", strength=1.0),
            _action("denoise", target="face", method="bilateral", strength=1.0),
            _action("gamma_correct", gamma=1.2),
        )
        guarded = apply_preserve_guard(plan, [PreserveItem(aspect="film_grain")])
        assert [a.operation for a in guarded.actions] == ["gamma_correct"]

    def test_full_image_preserve_covers_every_region(self):
        """Preserve toàn ảnh áp dụng cho mọi vùng (cùng quy ước với perception, retriever)."""
        plan = _plan(
            _action("gamma_correct", gamma=1.4),
            _action("clahe", clip_limit=2.0),
            _action("gamma_correct", target="face", gamma=1.4),
            _action("gamma_correct", gamma=0.9),
        )
        guarded = apply_preserve_guard(plan, [PreserveItem(aspect="low_key")])
        assert [(a.target_prompt, a.parameters["gamma"]) for a in guarded.actions] == [
            ("full", 0.9),
        ]

    def test_region_synonyms_are_guarded(self):
        """'khuôn mặt' và 'face' là cùng một vùng với preserve guard (như với executor)."""
        plan = _plan(
            _action("sharpen", target="khuôn mặt", method="unsharp_mask", amount=1.0),
            _action("sharpen", target="faces", method="unsharp_mask", amount=1.0),
            _action("sharpen", target="person", method="unsharp_mask", amount=1.0),
        )
        guarded = apply_preserve_guard(plan, [PreserveItem(aspect="soft_focus", region="face")])
        assert [a.target_prompt for a in guarded.actions] == ["person"]

    def test_region_preserve_also_guards_full_image_actions(self):
        plan = _plan(
            _action("color_correct", saturation_scale=1.0, temperature_shift=-0.3),
            _action("color_correct", target="sky", saturation_scale=1.0, temperature_shift=-0.3),
            _action("color_correct", target="face", saturation_scale=1.0, temperature_shift=-0.3),
        )
        guarded = apply_preserve_guard(plan, [PreserveItem(aspect="warm_tone", region="sky")])
        assert [a.target_prompt for a in guarded.actions] == ["face"]

    def test_soft_focus_and_silhouette(self):
        plan = _plan(
            _action("sharpen", target="face", method="unsharp_mask", amount=1.0),
            _action("gamma_correct", target="person", gamma=1.5),
            _action("clahe", target="person", clip_limit=2.0),
        )
        preserve = [
            PreserveItem(aspect="soft_focus"),
            PreserveItem(aspect="silhouette", region="person"),
        ]
        assert apply_preserve_guard(plan, preserve).actions == []


# ============================================================
# Lập kế hoạch từ chẩn đoán
# ============================================================
class TestPlanTreatment:
    @patch.dict("os.environ", FAKE_KEY)
    @patch("src.agent.vlm_diagnostician.genai")
    def test_clean_diagnosis_skips_the_plan_call(self, mock_genai):
        diagnosis = DiagnosisReport(
            defects=[Defect(type="noise", severity=0)], summary="Ảnh tốt.", source="vlm"
        )
        plan = plan_treatment(_backlit_image(), {}, diagnosis)

        assert plan.actions == []
        assert plan.source == "vlm"
        assert "Ảnh tốt." in plan.reasoning
        mock_genai.Client.assert_not_called()

    @patch.dict("os.environ", FAKE_KEY)
    @patch("src.agent.vlm_diagnostician.genai")
    def test_plan_receives_diagnosis_and_guard_applies(self, mock_genai):
        vlm_plan = {
            "reasoning": "Khử ám ấm.",
            "actions": [
                {
                    "region_id": "full",
                    "target_prompt": "full",
                    "detected_issue": "color_cast_warm",
                    "operation": "color_correct",
                    "parameters": {"saturation_scale": 1.0, "temperature_shift": -0.5},
                },
                {
                    "region_id": "full",
                    "target_prompt": "full",
                    "detected_issue": "noise",
                    "operation": "denoise",
                    "parameters": {"method": "bilateral", "strength": 1.0},
                },
            ],
        }
        models = _mock_models(mock_genai, json.dumps(vlm_plan))
        diagnosis = _report_from_vlm_json(VLM_DIAGNOSIS, iteration=1)

        plan = plan_treatment(_backlit_image(), {}, diagnosis)

        assert [a.operation for a in plan.actions] == ["denoise"]
        prompt = models.generate_content.call_args.kwargs["contents"][0]
        assert "CHẨN ĐOÁN" in prompt
        assert "warm_tone" in prompt

    def test_offline_plan_lifts_backlit_face(self):
        diagnosis = DiagnosisReport(
            defects=[Defect(type="backlit_subject", region="face", severity=3, origin="measured")]
        )
        plan = plan_treatment(_backlit_image(), {"brightness_level": "normal"}, diagnosis)

        [action] = plan.actions
        assert (action.operation, action.target_prompt) == ("gamma_correct", "face")
        assert action.parameters["gamma"] == 1.6
        assert action.feather_radius == 30

    def test_offline_plan_does_not_double_lift_when_globally_dark(self):
        diagnosis = DiagnosisReport(
            defects=[Defect(type="underexposed", region="face", severity=2, origin="measured")]
        )
        plan = diagnose_and_plan(
            _backlit_image(), {"brightness_level": "underexposed"}, diagnosis=diagnosis
        )
        assert [(a.operation, a.target_prompt) for a in plan.actions] == [("gamma_correct", "full")]


class TestPipelineWithPerception:
    def test_offline_pipeline_records_diagnosis(self):
        with patch(
            "src.agent.perception.resolve_region_mask",
            side_effect=_fake_resolver({"face": _center_mask()}),
        ):
            state = run_pipeline(_backlit_image(), max_iterations=1)

        first = state["history"][0]
        assert first.diagnosis is not None
        assert "face" in first.diagnosis.region_metrics
        operations: List[tuple] = [(a.operation, a.target_prompt) for a in first.plan.actions]
        assert ("gamma_correct", "face") in operations


def test_regions_are_measured_on_a_downscaled_image():
    """Đo vùng (và phát hiện khuôn mặt) chạy trên ảnh ≤ MEASURE_MAX_SIDE, không phải ảnh gốc."""
    from src.agent.perception import MEASURE_MAX_SIDE, measure_regions

    seen = []

    def resolve(image: np.ndarray, target_prompt: str, feather_radius: int = 15):
        seen.append(image.shape)
        return True, np.ones(image.shape[:2], dtype=np.float32), "test"

    big = np.full((1500, 3000, 3), 100, dtype=np.uint8)
    with patch("src.agent.perception.resolve_region_mask", side_effect=resolve):
        measured = measure_regions(big, ["face"])

    assert seen == [(MEASURE_MAX_SIDE // 2, MEASURE_MAX_SIDE, 3)]
    assert measured["face"].brightness_mean == pytest.approx(100)
