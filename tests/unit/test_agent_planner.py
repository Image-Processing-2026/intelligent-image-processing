"""
Unit tests for Module 4: Agent & Plan Validator.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from src.agent.graph import diagnose_and_plan_node
from src.agent.planner import clamp_parameters, validate_and_sort_plan
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


def test_clamp_parameters_over_max():
    """Tham số vượt trần → kẹp về giá trị max."""
    result = clamp_parameters("gamma_correct", {"gamma": 10.0})
    assert result["gamma"] == 2.5


def test_clamp_parameters_under_min():
    """Tham số dưới sàn → kẹp về giá trị min."""
    result = clamp_parameters("gamma_correct", {"gamma": 0.01})
    assert result["gamma"] == 0.5


def test_clamp_parameters_valid_value():
    """Tham số hợp lệ → giữ nguyên."""
    result = clamp_parameters("denoise", {"method": "bilateral", "strength": 1.5})
    assert result["strength"] == 1.5
    assert result["method"] == "bilateral"


def test_clamp_parameters_invalid_enum():
    """Enum không hợp lệ → fallback default."""
    result = clamp_parameters("denoise", {"method": "magic_filter", "strength": 1.0})
    assert result["method"] == "bilateral"


def test_clamp_parameters_missing_param():
    """Tham số bị thiếu → gán default."""
    result = clamp_parameters("gamma_correct", {})
    assert result["gamma"] == 1.2


def test_clamp_parameters_all_operations_and_edge_cases():
    """Kiểm tra đầy đủ các operation khác và các trường hợp biên/lỗi định dạng."""
    # CLAHE
    res_clahe_high = clamp_parameters("clahe", {"clip_limit": 50.0})
    assert res_clahe_high["clip_limit"] == 4.0
    res_clahe_low = clamp_parameters("clahe", {"clip_limit": 0.1})
    assert res_clahe_low["clip_limit"] == 1.0

    # Sharpen
    res_sharp = clamp_parameters("sharpen", {"amount": 5.0, "method": "invalid_method"})
    assert res_sharp["amount"] == 2.0
    assert res_sharp["method"] == "unsharp_mask"

    res_sharp_lap = clamp_parameters("sharpen", {"amount": 0.05, "method": "laplacian"})
    assert res_sharp_lap["amount"] == 0.2
    assert res_sharp_lap["method"] == "laplacian"

    # Color correct
    res_color = clamp_parameters(
        "color_correct",
        {"saturation_scale": 0.1, "temperature_shift": 5.0},
    )
    assert res_color["saturation_scale"] == 0.5
    assert res_color["temperature_shift"] == 1.0

    res_color_neg = clamp_parameters(
        "color_correct",
        {"saturation_scale": 2.0, "temperature_shift": -3.0},
    )
    assert res_color_neg["saturation_scale"] == 1.5
    assert res_color_neg["temperature_shift"] == -1.0

    # Non-numeric string input
    res_invalid_type = clamp_parameters("gamma_correct", {"gamma": "not_a_number"})
    assert res_invalid_type["gamma"] == 1.2

    # None params
    res_none = clamp_parameters("gamma_correct", None)
    assert res_none["gamma"] == 1.2

    # Unknown operation
    res_unknown = clamp_parameters("custom_tool", {"param": 42})
    assert res_unknown == {"param": 42}


def test_validate_plan_with_extreme_params():
    """Kế hoạch với tham số cực đoan phải được kẹp tự động."""
    plan = TreatmentPlan(
        iteration=1,
        reasoning="Test extreme params",
        actions=[
            RegionOperation(
                region_id="full",
                target_prompt="full",
                detected_issue="underexposed",
                operation="gamma_correct",
                parameters={"gamma": 50.0},
            ),
            RegionOperation(
                region_id="full",
                target_prompt="full",
                detected_issue="noise",
                operation="denoise",
                parameters={"strength": -5.0, "method": "unknown"},
            ),
        ],
    )
    validated = validate_and_sort_plan(plan)
    assert validated.actions[0].operation == "denoise"
    assert validated.actions[0].parameters["strength"] == 0.1
    assert validated.actions[0].parameters["method"] == "bilateral"
    assert validated.actions[1].operation == "gamma_correct"
    assert validated.actions[1].parameters["gamma"] == 2.5
