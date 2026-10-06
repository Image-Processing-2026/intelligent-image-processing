"""
Unit tests for Module 4: Agent & Plan Validator.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from src.agent.graph import (
    PSNR_THRESHOLD,
    SSIM_THRESHOLD,
    _decide_real,
    _decide_synthetic,
    decide_node,
    diagnose_and_plan_node,
    evaluate_node,
    process_node,
)
from src.agent.planner import clamp_parameters, clamp_region_fields, validate_and_sort_plan
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
    """region_type mặc định là None → executor tự suy ra từ target_prompt (D2)."""
    op = RegionOperation(
        region_id="full_image",
        target_prompt="full",
        detected_issue="noise",
        operation="denoise",
    )
    assert op.region_type is None


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

    with patch("src.agent.vlm_diagnostician.genai") as mock_genai:
        mock_genai.Client.return_value.models = mock_model
        img = np.ones((32, 32, 3), dtype=np.uint8) * 128
        metrics = {"noise_level": "medium"}
        plan = diagnose_and_plan(img, metrics, iteration=2, history=history)

        assert plan.iteration == 2
        mock_model.generate_content.assert_called_once()
        prompt_sent = mock_model.generate_content.call_args.kwargs["contents"][0]
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
            original_image=state["original_image"],
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


def test_clamp_region_fields_numeric_bounds():
    """Trường vùng số vượt biên được kẹp; không phải số → default (M2-INT-04)."""
    res = clamp_region_fields(
        {
            "feather_radius": 500,
            "expand_ratio": -2.0,
            "num_faces": 0,
            "box_threshold": 1.7,
            "text_threshold": "abc",
            "nms_iou_threshold": float("nan"),
        }
    )
    assert res["feather_radius"] == 50
    assert isinstance(res["feather_radius"], int)
    assert res["expand_ratio"] == 0.0
    assert res["num_faces"] == 1
    assert res["box_threshold"] == 1.0
    assert res["text_threshold"] == 0.25
    assert res["nms_iou_threshold"] == 0.8

    # ADR-003: không cho tắt feather; số thực được làm tròn về số nguyên
    assert clamp_region_fields({"feather_radius": 0})["feather_radius"] == 5
    assert clamp_region_fields({"feather_radius": 12.6})["feather_radius"] == 13


def test_clamp_region_fields_enums_and_defaults():
    """Enum sai → default; trường thiếu được bổ sung; khóa khác giữ nguyên."""
    res = clamp_region_fields(
        {
            "operation": "clahe",
            "region_type": "polygon",
            "quadrant": "upper-left",
            "face_mode": "sam_refined",
            "merge_policy": "min",
            "instance_selection": "random",
        }
    )
    assert res["operation"] == "clahe"
    assert res["region_type"] is None
    assert res["quadrant"] is None
    assert res["face_mode"] == "bbox"
    assert res["merge_policy"] == "max"
    assert res["instance_selection"] == "all"

    defaults = clamp_region_fields({})
    assert defaults["feather_radius"] == 15
    assert defaults["region_type"] is None
    assert defaults["instance_index"] is None

    # Chữ hoa/khoảng trắng được chuẩn hóa thay vì bị coi là sai
    assert clamp_region_fields({"region_type": " Semantic "})["region_type"] == "semantic"


def test_clamp_region_fields_drops_in_process_fields():
    """bbox/binary_mask từ VLM luôn bị đặt về None; region_type bbox → suy ra (D2)."""
    res = clamp_region_fields(
        {"region_type": "bbox", "bbox": [1.5, 2, 3, 4], "binary_mask": [[1, 0], [0, 1]]}
    )
    assert res["bbox"] is None
    assert res["binary_mask"] is None
    assert res["region_type"] is None


def test_clamp_region_fields_instance_index():
    """instance_index chỉ giữ khi instance_selection='index' và là số nguyên >= 0."""
    ok = clamp_region_fields({"instance_selection": "index", "instance_index": 2})
    assert (ok["instance_selection"], ok["instance_index"]) == ("index", 2)

    bad = clamp_region_fields({"instance_selection": "index", "instance_index": -1})
    assert (bad["instance_selection"], bad["instance_index"]) == ("all", None)

    stray = clamp_region_fields({"instance_selection": "largest", "instance_index": 3})
    assert (stray["instance_selection"], stray["instance_index"]) == ("largest", None)


def test_validate_plan_clamps_region_fields():
    """validate_and_sort_plan kẹp trường vùng của RegionOperation (model không còn ge/le)."""
    plan = TreatmentPlan(
        iteration=1,
        reasoning="Out-of-range region fields",
        actions=[
            RegionOperation(
                region_id="sky",
                target_prompt="sky",
                detected_issue="overexposed",
                operation="gamma_correct",
                parameters={"gamma": 0.8},
                feather_radius=999,
                expand_ratio=3.0,
                bbox=(0, 0, 4, 4),
                region_type="bbox",
            )
        ],
    )
    action = validate_and_sort_plan(plan).actions[0]
    assert action.feather_radius == 50
    assert action.expand_ratio == 1.0
    assert action.bbox is None
    assert action.region_type is None


def test_decide_synthetic_ship_on_good_quality():
    """Synthetic: PSNR và SSIM đạt ngưỡng → SHIP."""
    eval_result = {"psnr": 30.0, "ssim": 0.92}
    assert _decide_synthetic(eval_result, []) == "SHIP"


def test_decide_synthetic_reprocess_on_low_quality():
    """Synthetic: PSNR chưa đạt → RE_PROCESS."""
    eval_result = {"psnr": 22.0, "ssim": 0.75}
    assert _decide_synthetic(eval_result, []) == "RE_PROCESS"


def test_decide_synthetic_stop_on_degradation():
    """Synthetic: PSNR giảm > 1.5 dB → STOP_BEST_EFFORT."""
    prev_entry = HistoryItem(
        iteration=1,
        plan=TreatmentPlan(iteration=1, reasoning="test", actions=[]),
        metrics_before={},
        metrics_after={},
        eval_score={"psnr": 27.0, "ssim": 0.85},
        decision="RE_PROCESS",
    )
    eval_result = {"psnr": 24.0, "ssim": 0.80}  # giảm 3 dB
    assert _decide_synthetic(eval_result, [prev_entry]) == "STOP_BEST_EFFORT"


def test_decide_synthetic_continue_on_minor_dip():
    """Synthetic: PSNR giảm nhẹ <= 1.5 dB → RE_PROCESS (chưa phải suy thoái)."""
    prev_entry = HistoryItem(
        iteration=1,
        plan=TreatmentPlan(iteration=1, reasoning="test", actions=[]),
        metrics_before={},
        metrics_after={},
        eval_score={"psnr": 26.0, "ssim": 0.83},
        decision="RE_PROCESS",
    )
    eval_result = {"psnr": 25.0, "ssim": 0.82}  # giảm 1 dB <= 1.5 dB threshold
    assert _decide_synthetic(eval_result, [prev_entry]) == "RE_PROCESS"


def test_decide_real_stop_on_quality_drop():
    """Real: Quality score giảm → STOP_BEST_EFFORT."""
    prev_entry = HistoryItem(
        iteration=1,
        plan=TreatmentPlan(iteration=1, reasoning="test", actions=[]),
        metrics_before={},
        metrics_after={},
        eval_score={"estimated_quality_score": 75.0},
        decision="RE_PROCESS",
    )
    eval_result = {"estimated_quality_score": 60.0}
    assert _decide_real(eval_result, [prev_entry]) == "STOP_BEST_EFFORT"


def test_decide_real_continue_on_improvement():
    """Real: Quality score cải thiện hoặc giữ nguyên → RE_PROCESS."""
    prev_entry = HistoryItem(
        iteration=1,
        plan=TreatmentPlan(iteration=1, reasoning="test", actions=[]),
        metrics_before={},
        metrics_after={},
        eval_score={"estimated_quality_score": 60.0},
        decision="RE_PROCESS",
    )
    eval_result = {"estimated_quality_score": 75.0}
    assert _decide_real(eval_result, [prev_entry]) == "RE_PROCESS"


def test_decide_node_max_iterations_reached():
    """Đạt max iterations → dừng STOP_BEST_EFFORT (cả synthetic và real)."""
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    state_synthetic: DoctorState = {
        "original_image": img,
        "current_image": img,
        "ground_truth_image": img,
        "is_synthetic": True,
        "iteration": 3,
        "max_iterations": 3,
        "technical_metrics": {},
        "treatment_plan": TreatmentPlan(
            iteration=3,
            reasoning="still work",
            actions=[
                RegionOperation(
                    region_id="1", target_prompt="full", detected_issue="blur", operation="sharpen"
                )
            ],
        ),
        "evaluation_result": {"psnr": 20.0, "ssim": 0.70},
        "history": [],
        "decision": "PENDING",
        "error_message": None,
    }
    res_syn = decide_node(state_synthetic)
    assert res_syn["decision"] == "STOP_BEST_EFFORT"
    assert res_syn["iteration"] == 4
    assert res_syn["history"][-1].decision == "STOP_BEST_EFFORT"

    state_real: DoctorState = {
        **state_synthetic,
        "is_synthetic": False,
        "evaluation_result": {"estimated_quality_score": 40.0},
    }
    res_real = decide_node(state_real)
    assert res_real["decision"] == "STOP_BEST_EFFORT"
    assert res_real["iteration"] == 4


def test_decide_node_empty_plan_ships():
    """Kế hoạch điều trị rỗng hoặc None → VLM thấy ảnh đã tốt → SHIP."""
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    state_empty_plan: DoctorState = {
        "original_image": img,
        "current_image": img,
        "ground_truth_image": None,
        "is_synthetic": False,
        "iteration": 1,
        "max_iterations": 3,
        "technical_metrics": {},
        "treatment_plan": TreatmentPlan(iteration=1, reasoning="already good", actions=[]),
        "evaluation_result": {"estimated_quality_score": 80.0},
        "history": [],
        "decision": "PENDING",
        "error_message": None,
    }
    res = decide_node(state_empty_plan)
    assert res["decision"] == "SHIP"
    assert res["iteration"] == 2

    state_none_plan: DoctorState = {
        **state_empty_plan,
        "treatment_plan": None,
    }
    res_none = decide_node(state_none_plan)
    assert res_none["decision"] == "SHIP"


def test_decide_node_synthetic_ship_on_target_quality():
    """Synthetic: Chưa hết vòng nhưng đạt ngưỡng PSNR & SSIM → SHIP."""
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    state: DoctorState = {
        "original_image": img,
        "current_image": img,
        "ground_truth_image": img,
        "is_synthetic": True,
        "iteration": 1,
        "max_iterations": 3,
        "technical_metrics": {},
        "treatment_plan": TreatmentPlan(
            iteration=1,
            reasoning="some action",
            actions=[
                RegionOperation(
                    region_id="1", target_prompt="full", detected_issue="blur", operation="sharpen"
                )
            ],
        ),
        "evaluation_result": {"psnr": PSNR_THRESHOLD + 2.0, "ssim": SSIM_THRESHOLD + 0.05},
        "history": [],
        "decision": "PENDING",
        "error_message": None,
    }
    res = decide_node(state)
    assert res["decision"] == "SHIP"
    assert res["iteration"] == 2
    assert len(res["history"]) == 1
    assert res["history"][0].decision == "SHIP"


def test_process_node_tracks_previous_image_multi_turn():
    """[INT-01] Kiểm tra process_node lưu previous_image:
    - Vòng 1: previous_image là ảnh gốc.
    - Vòng 2: previous_image là ảnh kết quả của vòng 1.
    """
    img_orig = np.ones((32, 32, 3), dtype=np.uint8) * 100
    plan_round_1 = TreatmentPlan(
        iteration=1,
        reasoning="Tăng sáng vòng 1",
        actions=[
            RegionOperation(
                region_id="full",
                target_prompt="full",
                detected_issue="underexposed",
                operation="gamma_correct",
                parameters={"gamma": 1.5},
            )
        ],
    )

    state_round_1: DoctorState = {
        "original_image": img_orig.copy(),
        "current_image": img_orig.copy(),
        "previous_image": None,
        "ground_truth_image": None,
        "is_synthetic": False,
        "iteration": 1,
        "max_iterations": 3,
        "technical_metrics": {},
        "treatment_plan": plan_round_1,
        "evaluation_result": {},
        "history": [],
        "intermediate_images": [],
        "decision": "INITIALIZING",
        "error_message": None,
    }

    # Thực thi vòng 1
    res_1 = process_node(state_round_1)
    assert "previous_image" in res_1
    assert res_1["previous_image"] is not None
    # Vòng 1: previous_image là ảnh trước khi xử lý (ảnh gốc)
    np.testing.assert_array_equal(res_1["previous_image"], img_orig)
    # current_image đã bị thay đổi qua gamma
    assert not np.array_equal(res_1["current_image"], img_orig)

    # Chuẩn bị cho Vòng 2
    img_after_round_1 = res_1["current_image"]
    plan_round_2 = TreatmentPlan(
        iteration=2,
        reasoning="Khử nhiễu vòng 2",
        actions=[
            RegionOperation(
                region_id="full",
                target_prompt="full",
                detected_issue="noise",
                operation="denoise",
                parameters={"method": "bilateral", "strength": 1.0},
            )
        ],
    )
    state_round_2: DoctorState = {
        **state_round_1,
        "iteration": 2,
        "current_image": img_after_round_1,
        "previous_image": res_1["previous_image"],
        "treatment_plan": plan_round_2,
    }

    # Thực thi vòng 2
    res_2 = process_node(state_round_2)
    assert "previous_image" in res_2
    # Vòng 2: previous_image phải là ảnh của vòng liền kề trước đó (img_after_round_1)
    np.testing.assert_array_equal(res_2["previous_image"], img_after_round_1)


def test_process_node_tracks_previous_image_empty_plan():
    """[INT-01] Kế hoạch rỗng: previous_image vẫn lưu lại current_image hiện tại."""
    img = np.ones((32, 32, 3), dtype=np.uint8) * 120
    state: DoctorState = {
        "original_image": img,
        "current_image": img.copy(),
        "previous_image": None,
        "ground_truth_image": None,
        "is_synthetic": False,
        "iteration": 1,
        "max_iterations": 3,
        "technical_metrics": {},
        "treatment_plan": None,
        "evaluation_result": {},
        "history": [],
        "intermediate_images": [],
        "decision": "INITIALIZING",
        "error_message": None,
    }
    res = process_node(state)
    assert "previous_image" in res
    np.testing.assert_array_equal(res["previous_image"], img)
    np.testing.assert_array_equal(res["current_image"], img)


def test_evaluate_node_passes_params_synthetic():
    """[INT-02] evaluate_node truyền iteration và previous_image khi is_synthetic=True, sinh delta_metrics."""
    gt = np.ones((32, 32, 3), dtype=np.uint8) * 128
    prev = np.clip(gt.astype(np.int16) + 30, 0, 255).astype(np.uint8)
    curr = np.clip(gt.astype(np.int16) + 10, 0, 255).astype(np.uint8)

    state: DoctorState = {
        "original_image": prev,
        "current_image": curr,
        "previous_image": prev,
        "ground_truth_image": gt,
        "is_synthetic": True,
        "iteration": 2,
        "max_iterations": 3,
        "technical_metrics": {},
        "treatment_plan": None,
        "evaluation_result": {},
        "history": [],
        "intermediate_images": [],
        "decision": "PENDING",
        "error_message": None,
    }

    res = evaluate_node(state)
    eval_res = res["evaluation_result"]
    assert eval_res["iteration"] == 2
    assert eval_res["is_reference_eval"] is True
    assert "technical_metrics" in eval_res
    assert "delta_metrics" in eval_res
    assert eval_res["delta_metrics"] is not None
    assert "delta_psnr" in eval_res["delta_metrics"]
    assert "delta_ssim" in eval_res["delta_metrics"]


def test_evaluate_node_passes_params_real():
    """[INT-02] evaluate_node truyền iteration và previous_image khi is_synthetic=False."""
    prev = np.ones((32, 32, 3), dtype=np.uint8) * 80
    curr = np.ones((32, 32, 3), dtype=np.uint8) * 110

    state: DoctorState = {
        "original_image": prev,
        "current_image": curr,
        "previous_image": prev,
        "ground_truth_image": None,
        "is_synthetic": False,
        "iteration": 2,
        "max_iterations": 3,
        "technical_metrics": {},
        "treatment_plan": None,
        "evaluation_result": {},
        "history": [],
        "intermediate_images": [],
        "decision": "PENDING",
        "error_message": None,
    }

    res = evaluate_node(state)
    eval_res = res["evaluation_result"]
    assert eval_res["iteration"] == 2
    assert eval_res["is_reference_eval"] is False
    assert "technical_metrics" in eval_res
    assert "delta_metrics" in eval_res
    assert eval_res["delta_metrics"] is not None
    assert "delta_brightness" in eval_res["delta_metrics"]


def test_decide_node_zero_redundant_compute():
    """[INT-02] decide_node tái sử dụng technical_metrics từ evaluate_node, không gọi analyze_image."""
    img = np.ones((16, 16, 3), dtype=np.uint8) * 100
    fake_tech_metrics = {"brightness_mean": 100.0, "noise_variance": 5.0}

    state: DoctorState = {
        "original_image": img,
        "current_image": img,
        "previous_image": img,
        "ground_truth_image": None,
        "is_synthetic": False,
        "iteration": 1,
        "max_iterations": 3,
        "technical_metrics": {"brightness_mean": 90.0},
        "treatment_plan": TreatmentPlan(
            iteration=1,
            reasoning="test",
            actions=[
                RegionOperation(
                    region_id="1", target_prompt="full", detected_issue="blur", operation="sharpen"
                )
            ],
        ),
        "evaluation_result": {
            "estimated_quality_score": 75.0,
            "technical_metrics": fake_tech_metrics,
        },
        "history": [],
        "intermediate_images": [],
        "decision": "PENDING",
        "error_message": None,
    }

    with patch("src.agent.graph.analyze_image") as mock_analyze:
        res = decide_node(state)
        # analyze_image KHÔNG được phép gọi lại
        mock_analyze.assert_not_called()
        # metrics_after phải lấy chính xác fake_tech_metrics
        assert res["history"][-1].metrics_after == fake_tech_metrics


def test_decide_node_fallback_when_no_technical_metrics():
    """[INT-02] decide_node fallback gọi analyze_image nếu evaluation_result thiếu technical_metrics."""
    img = np.ones((16, 16, 3), dtype=np.uint8) * 100
    state: DoctorState = {
        "original_image": img,
        "current_image": img,
        "previous_image": img,
        "ground_truth_image": None,
        "is_synthetic": False,
        "iteration": 1,
        "max_iterations": 3,
        "technical_metrics": {},
        "treatment_plan": TreatmentPlan(
            iteration=1,
            reasoning="test",
            actions=[
                RegionOperation(
                    region_id="1", target_prompt="full", detected_issue="blur", operation="sharpen"
                )
            ],
        ),
        "evaluation_result": {"estimated_quality_score": 75.0},  # Thiếu technical_metrics
        "history": [],
        "intermediate_images": [],
        "decision": "PENDING",
        "error_message": None,
    }

    with patch("src.agent.graph.analyze_image") as mock_analyze:
        mock_analyze.return_value.model_dump.return_value = {"fallback": True}
        res = decide_node(state)
        mock_analyze.assert_called_once()
        assert res["history"][-1].metrics_after == {"fallback": True}


def test_decide_real_stop_on_quality_improved_false():
    """[INT-03] Khi Module 1 báo quality_improved=False (cháy sáng / nổ nhiễu) → STOP_BEST_EFFORT ngay lập tức."""
    eval_result = {
        "estimated_quality_score": 85.0,
        "quality_improved": False,  # Bị Module 1 chặn suy thoái
    }
    assert _decide_real(eval_result, []) == "STOP_BEST_EFFORT"


def test_decide_real_continue_when_quality_improved_true():
    """[INT-03] Khi quality_improved=True và điểm chưa đạt mục tiêu → RE_PROCESS."""
    eval_result = {
        "estimated_quality_score": 70.0,
        "quality_improved": True,
    }
    assert _decide_real(eval_result, []) == "RE_PROCESS"


def test_decide_node_stops_immediately_on_degradation_guard():
    """[INT-03] decide_node dừng sớm STOP_BEST_EFFORT ngay tại vòng 1 nếu phát hiện làm hỏng ảnh."""
    img = np.ones((16, 16, 3), dtype=np.uint8) * 100
    state: DoctorState = {
        "original_image": img,
        "current_image": img,
        "previous_image": img,
        "ground_truth_image": None,
        "is_synthetic": False,
        "iteration": 1,
        "max_iterations": 3,
        "technical_metrics": {},
        "treatment_plan": TreatmentPlan(
            iteration=1,
            reasoning="thử tăng sáng",
            actions=[
                RegionOperation(
                    region_id="1",
                    target_prompt="full",
                    detected_issue="dark",
                    operation="gamma_correct",
                )
            ],
        ),
        "evaluation_result": {
            "estimated_quality_score": 80.0,
            "quality_improved": False,  # Báo động suy thoái
            "technical_metrics": {},
        },
        "history": [],
        "intermediate_images": [],
        "decision": "PENDING",
        "error_message": None,
    }

    res = decide_node(state)
    assert res["decision"] == "STOP_BEST_EFFORT"
    assert res["history"][-1].decision == "STOP_BEST_EFFORT"
