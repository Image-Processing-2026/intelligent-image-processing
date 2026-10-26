"""
Unit test cho vòng ≥ 2 (Analysis v2 P7): perceive đo lại + so với ảnh gốc (không gọi VLM khi mọi
lỗi đo được), và decide_node rollback khi xử lý gây tác dụng phụ severity ≥ 2. Chạy offline;
Gemini được mock để chứng minh có/không gọi.
"""

import json
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from src.agent import perception
from src.agent.graph import decide_node
from src.agent.state import (
    Defect,
    DiagnosisReport,
    DoctorState,
    HistoryItem,
    RegionOperation,
    TreatmentPlan,
)
from src.processing_engine import apply_gamma, apply_sharpen
from src.region_engine import SubjectMask

FAKE_KEY = {"GEMINI_API_KEY": "fake-key-for-test"}


def _scene(size: int = 512) -> np.ndarray:
    """Ảnh có cạnh và chuyển sắc, đủ lớn cho bản đồ đo tile 64 px."""
    rng = np.random.default_rng(5)
    image = np.tile(np.linspace(40, 210, size, dtype=np.float32), (size, 1))
    image = np.repeat(image[..., None], 3, axis=2)
    for _ in range(10):
        x, y = rng.integers(0, size - 90, 2)
        image[y : y + 80, x : x + 80] = rng.uniform(40, 220, 3)
    return np.clip(image, 0, 255).astype(np.uint8)


def _plan(operation: str = "gamma_correct") -> TreatmentPlan:
    return TreatmentPlan(
        iteration=1,
        reasoning="r",
        actions=[
            RegionOperation(
                region_id="full",
                target_prompt="full",
                region_type="full",
                detected_issue="underexposed",
                operation=operation,
            )
        ],
    )


def _history(diagnosis: DiagnosisReport) -> list:
    return [
        HistoryItem(
            iteration=1,
            plan=_plan(),
            metrics_before={},
            metrics_after={},
            eval_score={},
            decision="RE_PROCESS",
            diagnosis=diagnosis,
        )
    ]


def _no_regions(image: np.ndarray, name: str, box_2d: Optional[tuple] = None, **_: Any):
    return SubjectMask(np.ones(image.shape[:2], np.float32), "full", 0.1, None)


def _mock_gemini(mock_genai: MagicMock, data: Dict[str, Any]) -> MagicMock:
    models = MagicMock()
    models.generate_content.return_value = MagicMock(text=json.dumps(data))
    mock_genai.Client.return_value.models = models
    return models


class TestFollowupPerceive:
    @patch.dict("os.environ", FAKE_KEY)
    @patch("src.agent.vlm_diagnostician.genai")
    def test_measured_fix_is_resolved_without_calling_vlm(self, mock_genai: MagicMock):
        """Lỗi đo được đã hết sau xử lý đúng → resolved, vòng 2 không gọi Gemini."""
        models = _mock_gemini(mock_genai, {})
        original = (_scene().astype(np.float32) * 0.25).astype(np.uint8)
        fixed = apply_gamma(original, gamma=2.2)
        previous = DiagnosisReport(
            defects=[Defect(type="underexposed", severity=3, origin="vlm")], source="vlm"
        )
        with patch("src.agent.perception.resolve_subject_mask", side_effect=_no_regions):
            report = perception.perceive(
                fixed, {}, iteration=2, history=_history(previous), original_image=original
            )
        assert models.generate_content.call_count == 0
        assert report.defects == []
        change = report.diff["defects"][0]
        assert (change["type"], change["status"]) == ("underexposed", "resolved")
        assert report.iteration == 2

    @patch.dict("os.environ", FAKE_KEY)
    @patch("src.agent.vlm_diagnostician.genai")
    def test_partial_fix_keeps_defect_with_measured_severity(self, mock_genai: MagicMock):
        models = _mock_gemini(mock_genai, {})
        original = (_scene().astype(np.float32) * 0.2).astype(np.uint8)
        partial = apply_gamma(original, gamma=1.15)
        previous = DiagnosisReport(
            defects=[Defect(type="underexposed", severity=3, origin="vlm")], source="vlm"
        )
        with patch("src.agent.perception.resolve_subject_mask", side_effect=_no_regions):
            report = perception.perceive(
                partial, {}, iteration=2, history=_history(previous), original_image=original
            )
        assert models.generate_content.call_count == 0
        [defect] = report.defects
        assert defect.type == "underexposed" and 1 <= defect.severity <= 3
        assert report.diff["defects"][0]["status"] in ("unchanged", "improved")

    @patch.dict("os.environ", FAKE_KEY)
    @patch("src.agent.vlm_diagnostician.genai")
    def test_open_semantic_defect_calls_vlm(self, mock_genai: MagicMock):
        """Lỗi không đo được (ám xanh lá) còn mở → gọi lại VLM để chẩn đoán vòng 2."""
        models = _mock_gemini(
            mock_genai,
            {"scene_type": "other", "defects": [], "preserve": [], "summary": "ổn"},
        )
        image = _scene()
        previous = DiagnosisReport(
            defects=[Defect(type="color_cast_green", severity=2, origin="vlm")], source="vlm"
        )
        with patch("src.agent.perception.resolve_subject_mask", side_effect=_no_regions):
            report = perception.perceive(
                image, {}, iteration=2, history=_history(previous), original_image=image.copy()
            )
        assert models.generate_content.call_count == 1
        assert report.defects == []
        assert report.diff is not None
        assert any("open" in note for note in report.fusion_notes)


def _decide_state(previous: np.ndarray, current: np.ndarray) -> DoctorState:
    return {
        "original_image": previous,
        "current_image": current,
        "previous_image": previous,
        "ground_truth_image": None,
        "is_synthetic": False,
        "iteration": 1,
        "max_iterations": 3,
        "technical_metrics": {},
        "diagnosis": None,
        "treatment_plan": _plan("sharpen"),
        "evaluation_result": {"estimated_quality_score": 90.0, "quality_improved": True},
        "history": [],
        "intermediate_images": [],
        "decision": "PENDING",
        "rolled_back": False,
        "error_message": None,
    }


class TestDecideSideEffects:
    def test_strong_side_effect_rolls_back(self):
        previous = cv2.GaussianBlur(_scene(), (0, 0), 1.0)
        current = apply_sharpen(apply_sharpen(previous, amount=2.0), amount=2.0)
        result = decide_node(_decide_state(previous, current))
        assert result["decision"] == "STOP_BEST_EFFORT"
        assert result["rolled_back"] is True
        assert np.array_equal(result["current_image"], previous)
        effects = result["history"][-1].side_effects
        assert max(effect["severity"] for effect in effects) >= 2

    def test_clean_processing_ships(self):
        previous = (_scene().astype(np.float32) * 0.5).astype(np.uint8)
        current = apply_gamma(previous, gamma=1.3)
        result = decide_node(_decide_state(previous, current))
        assert result["decision"] == "SHIP"
        assert result["rolled_back"] is False
        assert all(effect["severity"] < 2 for effect in result["history"][-1].side_effects)
