"""
Unit tests for Module 4: Agent & Plan Validator.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from src.agent.graph import diagnose_and_plan_node
from src.agent.planner import validate_and_sort_plan
from src.agent.state import DoctorState, HistoryItem, RegionOperation, TreatmentPlan
from src.agent.vlm_diagnostician import _build_history_feedback, diagnose_and_plan


def test_validate_and_sort_plan():
    """Kiểm tra việc lọc công cụ ngoài danh mục và sắp xếp độ ưu tiên."""
    raw_plan = TreatmentPlan(
        iteration=1,
        reasoning="Test plan",
        actions=[
            RegionOperation(
                region_id="1",
                target_prompt="full",
                detected_issue="blur",
                operation="sharpen",
                order=1,
            ),
            RegionOperation(
                region_id="2",
                target_prompt="full",
                detected_issue="noise",
                operation="denoise",
                order=2,
            ),
            RegionOperation(
                region_id="3",
                target_prompt="full",
                detected_issue="invalid",
                operation="deep_fake_beautify",  # Invalid operation
                order=3,
            ),
        ],
    )

    validated = validate_and_sort_plan(raw_plan)

    # Thao tác ngoài danh mục phải bị loại bỏ
    assert len(validated.actions) == 2
    # Thao tác khử nhiễu phải được sắp xếp trước làm nét (denoise before sharpen)
    assert validated.actions[0].operation == "denoise"
    assert validated.actions[1].operation == "sharpen"


def test_region_operation_default_region_type():
    """Kiểm tra giá trị mặc định của region_type trong RegionOperation."""
    op = RegionOperation(
        region_id="full_image",
        target_prompt="full",
        detected_issue="noise",
        operation="denoise",
    )
    assert op.region_type == "full"


def test_build_history_feedback_empty():
    """History rỗng hoặc None → feedback rỗng."""
    assert _build_history_feedback([]) == ""
    assert _build_history_feedback(None) == ""


def test_build_history_feedback_with_data():
    """History có dữ liệu → feedback chứa thông tin chi tiết của vòng trước."""
    history = [
        HistoryItem(
            iteration=1,
            plan=TreatmentPlan(
                iteration=1,
                reasoning="Khử nhiễu vùng nền",
                actions=[
                    RegionOperation(
                        region_id="background",
                        target_prompt="background",
                        region_type="semantic",
                        detected_issue="noise",
                        operation="denoise",
                        parameters={"method": "bilateral", "strength": 1.0},
                    )
                ],
            ),
            metrics_before={
                "brightness_mean": 80.0,
                "noise_variance": 25.0,
                "contrast_std": 45.0,
                "sharpness_laplacian_var": 120.0,
            },
            metrics_after={
                "brightness_mean": 80.5,
                "noise_variance": 15.0,
                "contrast_std": 44.5,
                "sharpness_laplacian_var": 115.0,
            },
            eval_score={"psnr": 25.0},
            decision="RE_PROCESS",
        )
    ]
    feedback = _build_history_feedback(history)
    assert "Vòng 1" in feedback
    assert "denoise" in feedback
    assert "bilateral" in feedback
    assert "background" in feedback
    assert "noise_variance: 25.00 → 15.00 (giảm 10.00)" in feedback
    assert "RE_PROCESS" in feedback
    assert "Khử nhiễu vùng nền" in feedback
    assert "TINH CHỈNH" in feedback


def test_build_history_feedback_window_limit():
    """Giới hạn tối đa 2 vòng gần nhất nếu history có 3 vòng trở lên."""
    history = [
        HistoryItem(
            iteration=i,
            plan=TreatmentPlan(
                iteration=i,
                reasoning=f"Reasoning {i}",
                actions=[],
            ),
            metrics_before={},
            metrics_after={},
            eval_score={},
            decision="RE_PROCESS",
        )
        for i in range(1, 4)
    ]
    feedback = _build_history_feedback(history)
    assert "Vòng 1" not in feedback
    assert "Vòng 2" in feedback
    assert "Vòng 3" in feedback


def test_diagnose_and_plan_fallback_accepts_history():
    """Hàm diagnose_and_plan ở chế độ fallback chấp nhận tham số history và gán region_type."""
    img = np.ones((32, 32, 3), dtype=np.uint8) * 128
    metrics = {
        "noise_level": "medium",
        "brightness_level": "underexposed",
        "contrast_level": "low",
    }
    plan = diagnose_and_plan(img, metrics, iteration=2, history=[])
    assert isinstance(plan, TreatmentPlan)
    assert len(plan.actions) > 0
    for action in plan.actions:
        assert action.region_type == "full"


def test_diagnose_and_plan_vlm_prompt_includes_history(monkeypatch):
    """Khi có GEMINI_API_KEY, prompt gửi VLM phải bao gồm phần history feedback."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key_for_test")

    history = [
        HistoryItem(
            iteration=1,
            plan=TreatmentPlan(
                iteration=1,
                reasoning="Khử nhiễu ban đầu",
                actions=[
                    RegionOperation(
                        region_id="full_image",
                        target_prompt="full",
                        region_type="full",
                        detected_issue="noise",
                        operation="denoise",
                        parameters={"method": "median"},
                    )
                ],
            ),
            metrics_before={"noise_variance": 50.0},
            metrics_after={"noise_variance": 30.0},
            eval_score={},
            decision="RE_PROCESS",
        )
    ]

    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"iteration": 2, "reasoning": "Tinh chỉnh tiếp", "actions": []}'
    mock_model.generate_content.return_value = mock_response

    with patch("google.generativeai.GenerativeModel", return_value=mock_model):
        img = np.ones((32, 32, 3), dtype=np.uint8) * 128
        metrics = {"noise_level": "medium"}
        plan = diagnose_and_plan(img, metrics, iteration=2, history=history)

        assert plan.iteration == 2
        mock_model.generate_content.assert_called_once()
        call_args = mock_model.generate_content.call_args[0][0]
        prompt_sent = call_args[0]
        assert "--- PHẢN HỒI TỪ CÁC VÒNG TRƯỚC ---" in prompt_sent
        assert "Vòng 1" in prompt_sent
        assert "noise_variance: 50.00 → 30.00 (giảm 20.00)" in prompt_sent


def test_diagnose_and_plan_node_passes_history():
    """Kiểm tra diagnose_and_plan_node lấy state['history'] truyền vào diagnose_and_plan."""
    state: DoctorState = {
        "original_image": np.zeros((10, 10, 3), dtype=np.uint8),
        "current_image": np.zeros((10, 10, 3), dtype=np.uint8),
        "ground_truth_image": None,
        "is_synthetic": False,
        "iteration": 2,
        "max_iterations": 3,
        "technical_metrics": {"noise_level": "clean", "brightness_level": "normal"},
        "treatment_plan": None,
        "evaluation_result": {},
        "history": [
            HistoryItem(
                iteration=1,
                plan=TreatmentPlan(iteration=1, reasoning="prev", actions=[]),
                metrics_before={},
                metrics_after={},
                eval_score={},
                decision="RE_PROCESS",
            )
        ],
        "decision": "RE_PROCESS",
        "error_message": None,
    }

    with patch("src.agent.graph.diagnose_and_plan") as mock_diag:
        mock_diag.return_value = TreatmentPlan(iteration=2, reasoning="ok", actions=[])
        res = diagnose_and_plan_node(state)
        mock_diag.assert_called_once_with(
            image=state["current_image"],
            metrics=state["technical_metrics"],
            iteration=2,
            history=state["history"],
        )
        assert res["treatment_plan"] is not None
