"""
Unit test cho hợp nhất tất định (Analysis v2 P6, src/agent/fusion.py): mỗi dòng bảng quyết định
một test, verify bằng câu hỏi đóng (fake verifier, không gọi Gemini), preserve, ask_user, prior
EXIF, lỗi không sửa được và contraindication trong planner.
"""

import json
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.agent import fusion, perception
from src.agent.fusion import (
    VERIFY_RESPONSE_SCHEMA,
    VerifyAnswer,
    VerifyQuestion,
    backlit_measurement,
    fuse,
    measure_hypothesis,
    parse_verify_answers,
    plan_contraindications,
)
from src.agent.intent import build_questions
from src.agent.planner import apply_contraindications
from src.agent.state import (
    Defect,
    DiagnosisReport,
    IntentProfile,
    PreserveItem,
    RegionOperation,
    TreatmentPlan,
)
from src.analyzer_evaluator.measurement import Estimate, MeasurementReport, RegionStats
from src.analyzer_evaluator.metadata import Prior
from src.region_engine import SubjectMask

FAKE_KEY = {"GEMINI_API_KEY": "fake-key-for-test"}


# ============================================================
# Helpers
# ============================================================
def _e(value: float, confidence: float = 1.0) -> Estimate:
    return Estimate(value=value, confidence=confidence, n_samples=100)


def _report(
    *,
    mean_l: float = 50.0,
    sigma: float = 1.0,
    illuminant: tuple = (1 / 3, 1 / 3, 1 / 3),
    blockiness: float = 1.0,
) -> MeasurementReport:
    """Báo cáo đo toàn ảnh 1024×768 với các giá trị kiểm soát được (đi qua calibration thật)."""
    r, g, b = illuminant
    return MeasurementReport(
        width=1024,
        height=768,
        tile_size=64,
        exposure={
            "mean_l": _e(mean_l),
            "p5_l": _e(max(0.0, mean_l - 40)),
            "p95_l": _e(min(100.0, mean_l + 40)),
            "largest_clip_blob": _e(0.0),
        },
        noise={"sigma_luma": _e(sigma), "sigma_dark_ratio": _e(1.0)},
        sharpness={"defocus_sigma": _e(0.5)},
        color={
            "illuminant_r": _e(r),
            "illuminant_g": _e(g),
            "illuminant_b": _e(b),
            "chroma_p50": _e(15.0),
        },
        compression={"blockiness": _e(blockiness)},
        haze={"dark_channel_median": _e(0.1)},
    )


def _stats(region: str, mean_l: float, area: float = 0.2, sigma: float = 1.0) -> RegionStats:
    return RegionStats(
        region=region,
        area_fraction=area,
        values={
            "exposure.mean_l": _e(mean_l),
            "exposure.clip_high": _e(0.0),
            "noise.sigma_luma": _e(sigma),
        },
        mask_source="test",
        mask_confidence=0.9,
    )


def _vlm(*defects: Defect, preserve: Optional[List[PreserveItem]] = None) -> DiagnosisReport:
    return DiagnosisReport(defects=list(defects), preserve=preserve or [], source="vlm")


def _claim(defect_type: str, severity: int = 2, region: str = "full", **kw) -> Defect:
    return Defect(
        type=defect_type, region=region, severity=severity, evidence="VLM thấy", origin="vlm", **kw
    )


class FakeVerifier:
    """Verifier giả: trả verdict cố định theo loại lỗi, ghi lại câu hỏi đã nhận."""

    def __init__(self, verdicts: Dict[str, str], confidence: float = 0.9) -> None:
        self.verdicts = verdicts
        self.confidence = confidence
        self.calls: List[List[VerifyQuestion]] = []

    def __call__(self, questions: List[VerifyQuestion]) -> Dict[str, VerifyAnswer]:
        self.calls.append(questions)
        return {
            q.hypothesis_id: VerifyAnswer(
                hypothesis_id=q.hypothesis_id,
                verdict=self.verdicts[q.defect_type],
                confidence=self.confidence,
                reason="kiểm chứng",
            )
            for q in questions
            if q.defect_type in self.verdicts
        }


def _types(defects: List[Defect]) -> List[tuple]:
    return [(d.type, d.region) for d in defects]


# ============================================================
# Đo một giả thuyết
# ============================================================
class TestMeasureHypothesis:
    def test_full_image_uses_calibration(self):
        measured = measure_hypothesis("underexposed", "full", _report(mean_l=20.0), {})
        assert measured is not None and measured.severity == 3

    def test_unmeasurable_type_is_none(self):
        assert measure_hypothesis("color_cast_green", "full", _report(), {}) is None

    def test_region_without_stats_is_none_except_global_types(self):
        report = _report(blockiness=1.6)
        assert measure_hypothesis("noise", "sky", report, {}) is None
        assert measure_hypothesis("jpeg_artifacts", "sky", report, {}).severity == 2

    def test_backlit_face_against_bright_background(self):
        stats = {"face": _stats("face", 25.0), "background": _stats("background", 80.0, 0.8)}
        measured = measure_hypothesis("backlit_subject", "face", _report(), stats)
        assert measured is not None and measured.severity == 3

    def test_dark_face_on_dark_background_is_not_backlit(self):
        """Mặt tối nhưng nền cũng tối (cảnh đêm): không phải ngược sáng."""
        measured = backlit_measurement(
            "face", _stats("face", 25.0), _stats("background", 30.0, 0.8)
        )
        assert measured is not None and measured.severity == 0

    def test_backdrop_region_is_never_backlit(self):
        assert backlit_measurement("sky", _stats("sky", 20.0), _stats("bg", 90.0)) is None


# ============================================================
# Bảng quyết định: mỗi dòng một test
# ============================================================
class TestDecisionTable:
    def test_row1_measured_and_vlm_agree_is_confirmed_with_measured_severity(self):
        """severity ≥ 1 + VLM báo lỗi → confirmed, severity = số đo, VLM kéo tối đa ±1."""
        fused = fuse(_vlm(_claim("underexposed", severity=1)), _report(mean_l=20.0), {})
        [defect] = fused.defects
        assert (defect.type, defect.severity) == ("underexposed", 2)  # đo 3, VLM 1 → 3 - 1
        assert defect.fixable
        assert any(item.metric == "calibrated_severity" for item in defect.evidence_items)

    def test_row2_measured_but_vlm_preserves_is_preserved(self):
        """Hoàng hôn + preserve warm_tone: lỗi ám ấm đo được bị chặn, không vào defects."""
        warm = (0.5, 0.33, 0.17)
        fused = fuse(
            _vlm(preserve=[PreserveItem(aspect="warm_tone", reason="hoàng hôn")]),
            _report(illuminant=warm),
            {},
        )
        assert fused.defects == []
        assert ("color_cast_warm", "full") in _types(fused.rejected)
        assert [p.aspect for p in fused.preserve] == ["warm_tone"]

    def test_row2_severe_on_subject_asks_user(self):
        stats = {"face": _stats("face", 20.0), "background": _stats("background", 85.0, 0.8)}
        fused = fuse(
            _vlm(preserve=[PreserveItem(aspect="silhouette", region="face")]),
            _report(),
            stats,
        )
        assert fused.defects == []
        assert _types(fused.ask_user) == [("backlit_subject", "face")]

    def test_row3_measured_only_is_verified_and_confirmed(self):
        verifier = FakeVerifier({"underexposed": "defect"})
        fused = fuse(_vlm(), _report(mean_l=20.0), {}, verifier=verifier)
        assert _types(fused.defects) == [("underexposed", "full")]
        assert fused.defects[0].origin == "measured"
        assert fused.defects[0].confidence <= fusion.MEDIUM_CONFIDENCE
        assert len(verifier.calls) == 1

    def test_row3_measured_only_verified_intentional_becomes_preserve(self):
        verifier = FakeVerifier({"underexposed": "intentional"})
        fused = fuse(_vlm(), _report(mean_l=20.0), {}, verifier=verifier)
        assert fused.defects == []
        assert [p.aspect for p in fused.preserve] == ["low_key"]

    def test_row3_measured_only_is_never_emitted_alone_with_vlm(self):
        """Có VLM nhưng không verify được → lỗi đo được không được phát ra một mình."""
        fused = fuse(_vlm(), _report(mean_l=20.0), {}, verifier=None)
        assert fused.defects == []
        assert ("underexposed", "full") in _types(fused.rejected)

    def test_row4_vlm_noise_on_clean_image_is_rejected(self):
        """VLM khai nhiễu, số đo 0 trên ảnh sạch, verify not_present → rejected (ảo giác)."""
        verifier = FakeVerifier({"noise": "not_present"})
        fused = fuse(_vlm(_claim("noise")), _report(sigma=0.8), {}, verifier=verifier)
        assert fused.defects == []
        assert _types(fused.rejected) == [("noise", "full")]
        [question] = verifier.calls[0]
        assert question.defect_type == "noise" and "texture" in question.question

    def test_row4_vlm_claim_confirmed_by_verify(self):
        verifier = FakeVerifier({"noise": "defect"})
        fused = fuse(_vlm(_claim("noise")), _report(sigma=0.8), {}, verifier=verifier)
        assert _types(fused.defects) == [("noise", "full")]

    def test_row4_without_verify_keeps_vlm_with_lower_confidence(self):
        fused = fuse(_vlm(_claim("noise")), _report(sigma=0.8), {}, verifier=None)
        [defect] = fused.defects
        assert defect.confidence == pytest.approx(fusion.UNVERIFIED_VLM_SCALE)

    def test_row5_unmeasurable_semantic_defect_is_confirmed_with_consensus(self):
        claim = _claim("color_cast_green", confidence=0.667)
        verifier = FakeVerifier({})
        fused = fuse(_vlm(claim), _report(), {}, verifier=verifier)
        assert _types(fused.defects) == [("color_cast_green", "full")]
        assert fused.defects[0].confidence == pytest.approx(0.667)
        assert verifier.calls == []  # bảng không yêu cầu verify

    @pytest.mark.parametrize(
        ("verdict", "expected"),
        [("defect", "confirmed"), ("not_present", "rejected"), ("texture", "rejected")],
    )
    def test_row1_low_trust_colour_cast_goes_to_verify(self, verdict: str, expected: str):
        """Ám màu đo bằng đường dự phòng (illuminant_method ≥ 1) + VLM đồng ý → verify quyết định,
        severity theo VLM chứ không theo số đo."""
        report = _report(illuminant=(0.5, 0.33, 0.17))  # ln(r/b) ≈ 1.08 → đo warm 3
        report.color["illuminant_method"] = _e(1.0)
        assert measure_hypothesis("color_cast_warm", "full", report, {}).low_trust
        verifier = FakeVerifier({"color_cast_warm": verdict})
        fused = fuse(_vlm(_claim("color_cast_warm", severity=1)), report, {}, verifier=verifier)
        assert [q.defect_type for q in verifier.calls[0]] == ["color_cast_warm"]
        if expected == "confirmed":
            [defect] = fused.defects
            assert (defect.type, defect.severity) == ("color_cast_warm", 1)
        else:
            assert fused.defects == []
            assert ("color_cast_warm", "full") in _types(fused.rejected)

    def test_row1_trusted_colour_cast_is_not_verified(self):
        """Đường chuẩn (illuminant_method = 0) giữ dòng 1: không verify, severity = số đo (±1)."""
        report = _report(illuminant=(0.5, 0.33, 0.17))
        report.color["illuminant_method"] = _e(0.0)
        verifier = FakeVerifier({"color_cast_warm": "not_present"})
        fused = fuse(_vlm(_claim("color_cast_warm", severity=1)), report, {}, verifier=verifier)
        assert verifier.calls == []
        assert [(d.type, d.severity) for d in fused.defects] == [("color_cast_warm", 2)]

    def test_row1_low_trust_without_verify_keeps_vlm_severity(self):
        report = _report(illuminant=(0.5, 0.33, 0.17))
        report.color["illuminant_method"] = _e(2.0)
        fused = fuse(_vlm(_claim("color_cast_warm", severity=1)), report, {}, verifier=None)
        [defect] = fused.defects
        assert defect.severity == 1
        assert defect.confidence == pytest.approx(fusion.UNVERIFIED_VLM_SCALE)

    def test_row6_intent_wins(self):
        intent = IntentProfile(fix=["color_cast_warm"])
        fused = fuse(
            _vlm(preserve=[PreserveItem(aspect="warm_tone")]),
            _report(illuminant=(0.5, 0.33, 0.17)),
            {},
            intent=intent,
        )
        assert ("color_cast_warm", "full") in _types(fused.defects)
        assert all(p.aspect != "warm_tone" for p in fused.preserve)

    def test_clean_image_and_silent_vlm_has_no_defects_and_no_verify(self):
        verifier = FakeVerifier({})
        fused = fuse(_vlm(), _report(), {}, verifier=verifier)
        assert fused.defects == [] and verifier.calls == []

    def test_one_verify_call_for_all_questions(self):
        verifier = FakeVerifier({"underexposed": "defect", "noise": "not_present"})
        fuse(_vlm(_claim("noise")), _report(mean_l=20.0, sigma=0.8), {}, verifier=verifier)
        assert len(verifier.calls) == 1
        assert {q.defect_type for q in verifier.calls[0]} == {"noise", "underexposed"}

    def test_failed_verify_falls_back_deterministically(self):
        def broken(questions: List[VerifyQuestion]) -> Dict[str, VerifyAnswer]:
            raise RuntimeError("network down")

        fused = fuse(_vlm(_claim("noise")), _report(mean_l=20.0, sigma=0.8), {}, verifier=broken)
        assert _types(fused.defects) == [("noise", "full")]  # lời VLM giữ, hạ confidence
        assert ("underexposed", "full") in _types(fused.rejected)  # số đo không phát một mình

    def test_fusion_is_deterministic(self):
        verifier = FakeVerifier({"underexposed": "defect", "noise": "texture"})
        args = (_vlm(_claim("noise")), _report(mean_l=20.0, sigma=0.8), {})
        first = fuse(*args, verifier=verifier).model_dump()
        second = fuse(*args, verifier=verifier).model_dump()
        assert first == second


# ============================================================
# Luật dự phòng (không VLM), prior EXIF, bằng chứng yếu
# ============================================================
class TestOfflineAndPriors:
    def test_offline_keeps_rule_defects_and_adds_confident_regional_measurement(self):
        rules = DiagnosisReport(
            defects=[Defect(type="noise", severity=2, origin="rule")], source="rule_based"
        )
        stats = {"face": _stats("face", 20.0), "background": _stats("background", 85.0, 0.8)}
        fused = fuse(rules, _report(mean_l=20.0), stats)
        assert _types(fused.defects) == [("noise", "full"), ("backlit_subject", "face")]

    def test_prior_shifts_confidence_only(self):
        noisy = Prior(kind="noise", strength=1.0, reason="ISO cao")
        base = fuse(_vlm(_claim("noise")), _report(sigma=12.0), {})
        shifted = fuse(_vlm(_claim("noise")), _report(sigma=12.0), {}, [noisy])
        assert shifted.defects[0].confidence >= base.defects[0].confidence
        assert fuse(_vlm(), _report(), {}, [noisy]).defects == []  # không tự tạo lỗi

    def test_no_sharpen_prior_adds_contraindication(self):
        prior = Prior(kind="no_sharpen", strength=0.8, reason="JPEG nén nặng")
        fused = fuse(_vlm(_claim("noise")), _report(sigma=12.0), {}, [prior])
        assert "no_sharpen" in plan_contraindications(fused)

    def test_weak_rejected_hypothesis_supports_measurement(self):
        hypotheses = _vlm()
        hypotheses.rejected = [_claim("underexposed", confidence=0.333)]
        verifier = FakeVerifier({})
        fused = fuse(hypotheses, _report(mean_l=20.0), {}, verifier=verifier, use_weak=True)
        assert ("underexposed", "full") in _types(fused.defects)
        assert verifier.calls == []
        off = fuse(hypotheses, _report(mean_l=20.0), {}, verifier=verifier, use_weak=False)
        assert ("underexposed", "full") not in _types(off.defects)


# ============================================================
# Lỗi không sửa được, contraindication, clarify
# ============================================================
class TestFixabilityAndPlanning:
    def test_jpeg_artifacts_is_not_fixable_and_contraindicates_sharpen(self):
        verifier = FakeVerifier({"jpeg_artifacts": "defect"})
        fused = fuse(_vlm(), _report(blockiness=1.6), {}, verifier=verifier)
        [defect] = fused.defects
        assert defect.type == "jpeg_artifacts" and defect.fixable is False
        assert defect.contraindications == ["no_sharpen"]
        assert fused.actionable_defects == []

    def test_minor_non_fixable_defect_is_not_reported_nor_verified(self):
        """Cháy sáng nhẹ (mức 1) toolbox không sửa được: bỏ, không tốn câu hỏi verify."""
        stats = {"face": _stats("face", 60.0), "background": _stats("background", 60.0, 0.8)}
        stats["face"].values["exposure.clip_high"] = _e(0.01)
        verifier = FakeVerifier({"highlight_clipping": "defect"})
        claim = _claim("highlight_clipping", severity=1, region="face")
        fused = fuse(_vlm(claim), _report(), stats, verifier=verifier)
        assert fused.defects == []
        assert ("highlight_clipping", "face") in _types(fused.rejected)
        assert verifier.calls == []

    def test_contraindication_drops_sharpen_from_plan(self):
        plan = TreatmentPlan(
            iteration=1,
            reasoning="r",
            actions=[
                RegionOperation(
                    region_id="full",
                    target_prompt="full",
                    region_type="full",
                    detected_issue=issue,
                    operation=operation,
                    parameters=params,
                )
                for issue, operation, params in (
                    ("noise", "denoise", {"strength": 1.0}),
                    ("blur", "sharpen", {"amount": 1.0}),
                )
            ],
        )
        guarded = apply_contraindications(plan, ["no_sharpen"])
        assert [a.operation for a in guarded.actions] == ["denoise"]
        assert "no_sharpen" in guarded.reasoning

    def test_ask_user_is_part_of_the_recommendation_text(self):
        report = DiagnosisReport(
            ask_user=[Defect(type="backlit_subject", region="face", severity=3)],
            preserve=[PreserveItem(aspect="silhouette", region="face")],
        )
        [question] = build_questions(report)
        assert question.default == "recommend"
        assert question.choices[0].label == "Giữ nguyên"
        assert "chưa chắc là lỗi" in question.text
        report.defects = [Defect(type="noise", severity=2)]
        [question] = build_questions(report)
        assert "mặc định giữ nguyên" in question.text
        assert question == build_questions(report)[0]  # tất định (chạy lại khi resume)


# ============================================================
# Verify: schema, parse, tích hợp perceive (Gemini mock)
# ============================================================
class TestVerifyIntegration:
    def test_parse_answers_drops_invalid_items(self):
        answers = parse_verify_answers(
            {
                "answers": [
                    {"hypothesis_id": "h1", "verdict": "defect", "confidence": 0.8},
                    {"hypothesis_id": "h2", "verdict": "maybe"},
                    {"hypothesis_id": "h3", "verdict": "uncertain"},
                    "rác",
                ]
            }
        )
        assert list(answers) == ["h1"]

    def test_verify_schema_has_closed_verdicts(self):
        verdict = VERIFY_RESPONSE_SCHEMA["properties"]["answers"]["items"]["properties"]["verdict"]
        assert verdict["enum"] == ["defect", "intentional", "texture", "not_present", "uncertain"]

    @patch.dict("os.environ", FAKE_KEY)
    @patch("src.agent.vlm_diagnostician.genai")
    def test_perceive_calls_verify_and_rejects_hallucinated_noise(
        self, mock_genai: MagicMock, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(perception, "VERIFY_ENABLED", True)
        perceive_json = {
            "scene_type": "landscape",
            "subjects": [],
            "defects": [{"type": "noise", "region": "full", "severity": 2, "evidence": "hạt"}],
            "preserve": [],
            "summary": "nhiễu",
        }

        def respond(**kwargs) -> MagicMock:
            schema = kwargs["config"].response_json_schema
            if schema == VERIFY_RESPONSE_SCHEMA:
                prompt = kwargs["contents"][0]
                hid = prompt.split("- ", 1)[1].split(":", 1)[0]
                data = {"answers": [{"hypothesis_id": hid, "verdict": "not_present"}]}
            else:
                data = perceive_json
            return MagicMock(text=json.dumps(data))

        models = MagicMock()
        models.generate_content.side_effect = respond
        mock_genai.Client.return_value.models = models
        image = np.full((256, 256, 3), 128, dtype=np.uint8)
        with patch(
            "src.agent.perception.resolve_subject_mask",
            return_value=SubjectMask(np.ones((256, 256), np.float32), "full", 0.1, None),
        ):
            report = perception.perceive(image, {})

        assert models.generate_content.call_count == 2
        assert report.defects == []
        assert ("noise", "full") in _types(report.rejected)
        assert any("verify not_present" in note for note in report.fusion_notes)
