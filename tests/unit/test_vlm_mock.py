"""
Unit tests cho VLM Diagnostician với mock phản hồi Gemini.
Chạy offline, không cần GEMINI_API_KEY, không tốn quota API.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.agent.graph import run_pipeline
from src.agent.planner import validate_and_sort_plan
from src.agent.state import TreatmentPlan
from src.agent.vlm_diagnostician import diagnose_and_plan

# ============================================================
# Constants & Fixtures: Các mẫu phản hồi mock từ Gemini VLM
# ============================================================

RAW_VALID_RESPONSE = {
    "iteration": 1,
    "reasoning": "Ảnh bị thiếu sáng vùng trung tâm, nhiễu Gaussian mức trung bình ở nền.",
    "actions": [
        {
            "region_id": "background",
            "target_prompt": "background",
            "region_type": "semantic",
            "detected_issue": "high_noise",
            "operation": "denoise",
            "parameters": {"method": "bilateral", "strength": 1.2},
            "order": 10,
        },
        {
            "region_id": "full_image",
            "target_prompt": "full",
            "region_type": "full",
            "detected_issue": "underexposed",
            "operation": "gamma_correct",
            "parameters": {"gamma": 1.4},
            "order": 20,
        },
    ],
}

RAW_EXTREME_PARAMS_RESPONSE = {
    "iteration": 1,
    "reasoning": "Cần xử lý mạnh tay.",
    "actions": [
        {
            "region_id": "full_image",
            "target_prompt": "full",
            "region_type": "full",
            "detected_issue": "very_dark",
            "operation": "gamma_correct",
            "parameters": {"gamma": 50.0},
            "order": 1,
        },
        {
            "region_id": "full_image",
            "target_prompt": "full",
            "region_type": "full",
            "detected_issue": "noise",
            "operation": "denoise",
            "parameters": {"method": "bilateral", "strength": 100.0},
            "order": 2,
        },
    ],
}

RAW_UNKNOWN_OP_RESPONSE = {
    "iteration": 1,
    "reasoning": "Dùng công cụ AI nâng cấp.",
    "actions": [
        {
            "region_id": "face",
            "target_prompt": "face",
            "region_type": "face",
            "detected_issue": "blemishes",
            "operation": "face_beautify",
            "parameters": {"intensity": 0.8},
            "order": 1,
        },
        {
            "region_id": "full_image",
            "target_prompt": "full",
            "region_type": "full",
            "detected_issue": "blur",
            "operation": "sharpen",
            "parameters": {"method": "unsharp_mask", "amount": 1.5},
            "order": 2,
        },
    ],
}

MOCK_VALID_RESPONSE = json.dumps(RAW_VALID_RESPONSE)
MOCK_EXTREME_PARAMS_RESPONSE = json.dumps(RAW_EXTREME_PARAMS_RESPONSE)
MOCK_UNKNOWN_OP_RESPONSE = json.dumps(RAW_UNKNOWN_OP_RESPONSE)
MOCK_INVALID_JSON = "Đây không phải JSON hợp lệ. Tôi nghĩ ảnh đẹp rồi."
MOCK_JSON_WITH_MARKDOWN = f"```json\n{MOCK_VALID_RESPONSE}\n```"
MOCK_JSON_WITH_MARKDOWN_NO_LANG = f"```\n{MOCK_VALID_RESPONSE}\n```"


@pytest.fixture
def mock_gemini_response_valid():
    return MOCK_VALID_RESPONSE


@pytest.fixture
def mock_gemini_response_invalid_json():
    return MOCK_INVALID_JSON


@pytest.fixture
def mock_gemini_response_extreme_params():
    return MOCK_EXTREME_PARAMS_RESPONSE


@pytest.fixture
def mock_gemini_response_unknown_op():
    return MOCK_UNKNOWN_OP_RESPONSE


# ============================================================
# Helper: Tạo mock Gemini model
# ============================================================


def _create_mock_model(response_text: str):
    """Tạo mock `client.models` (google-genai) trả về response text cố định."""
    mock_response = MagicMock()
    mock_response.text = response_text
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response
    return mock_model


DARK_DEFECT = {"type": "underexposed", "region": "full", "severity": 2, "evidence": "tối"}


def _diagnosis_json(*defects: dict, preserve: tuple = (), subjects: tuple = ()) -> str:
    """Phản hồi JSON giai đoạn Perceive."""
    return json.dumps(
        {
            "scene_type": "other",
            "lighting": "",
            "subjects": list(subjects),
            "defects": list(defects),
            "preserve": list(preserve),
            "summary": "Tóm tắt chẩn đoán.",
        }
    )


def _route_gemini(mock_genai: MagicMock, perceive: list, plan: list) -> MagicMock:
    """
    Mock `client.models` trả phản hồi theo giai đoạn (Perceive hay Plan),
    nhận diện qua schema trong config; mỗi giai đoạn tiêu thụ hàng đợi riêng.
    """
    from src.agent.perception import PERCEIVE_RESPONSE_SCHEMA

    queues = {"perceive": list(perceive), "plan": list(plan)}

    def generate_content(*, model: str, contents: list, config: object) -> MagicMock:
        is_perceive = config.response_json_schema == PERCEIVE_RESPONSE_SCHEMA
        return MagicMock(text=queues["perceive" if is_perceive else "plan"].pop(0))

    models = MagicMock()
    models.generate_content.side_effect = generate_content
    mock_genai.Client.return_value.models = models
    return models


# ============================================================
# Test Cases
# ============================================================


class TestVLMValidResponse:
    """Test khi VLM trả về JSON chuẩn."""

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key-for-test"})
    @patch("src.agent.vlm_diagnostician.genai")
    def test_valid_json_parsed_correctly(self, mock_genai, mock_gemini_response_valid):
        mock_genai.Client.return_value.models = _create_mock_model(mock_gemini_response_valid)

        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        plan = diagnose_and_plan(img, {"noise_level": "medium"}, iteration=1)

        assert isinstance(plan, TreatmentPlan)
        assert len(plan.actions) == 2
        assert plan.actions[0].operation == "denoise"
        assert plan.actions[1].operation == "gamma_correct"
        assert plan.reasoning != ""

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key-for-test"})
    @patch("src.agent.vlm_diagnostician.genai")
    def test_json_with_markdown_fences(self, mock_genai):
        """VLM bọc JSON trong ```json ... ``` → parser phải strip được."""
        mock_genai.Client.return_value.models = _create_mock_model(MOCK_JSON_WITH_MARKDOWN)

        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        plan = diagnose_and_plan(img, {}, iteration=1)

        assert isinstance(plan, TreatmentPlan)
        assert len(plan.actions) == 2

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key-for-test"})
    @patch("src.agent.vlm_diagnostician.genai")
    def test_json_with_plain_fences(self, mock_genai):
        """VLM bọc JSON trong ``` ... ``` (không ghi 'json') → parser vẫn strip được."""
        mock_genai.Client.return_value.models = _create_mock_model(MOCK_JSON_WITH_MARKDOWN_NO_LANG)

        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        plan = diagnose_and_plan(img, {}, iteration=1)

        assert isinstance(plan, TreatmentPlan)
        assert len(plan.actions) == 2


class TestVLMExtremeParams:
    """Test khi VLM trả về tham số vượt biên → clamping phải hoạt động."""

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key-for-test"})
    @patch("src.agent.vlm_diagnostician.genai")
    def test_extreme_params_clamped(self, mock_genai, mock_gemini_response_extreme_params):
        mock_genai.Client.return_value.models = _create_mock_model(
            mock_gemini_response_extreme_params
        )

        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        plan = diagnose_and_plan(img, {}, iteration=1)
        validated = validate_and_sort_plan(plan)

        # Sau khi validate+clamp, gamma phải nằm trong [0.5, 2.5]
        gamma_action = [a for a in validated.actions if a.operation == "gamma_correct"][0]
        assert gamma_action.parameters["gamma"] <= 2.5

        # strength phải nằm trong [0.1, 2.0]
        denoise_action = [a for a in validated.actions if a.operation == "denoise"][0]
        assert denoise_action.parameters["strength"] <= 2.0


class TestVLMUnknownOperation:
    """Test khi VLM trả về operation không tồn tại → validator phải loại bỏ."""

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key-for-test"})
    @patch("src.agent.vlm_diagnostician.genai")
    def test_unknown_op_removed(self, mock_genai, mock_gemini_response_unknown_op):
        mock_genai.Client.return_value.models = _create_mock_model(mock_gemini_response_unknown_op)

        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        plan = diagnose_and_plan(img, {}, iteration=1)
        validated = validate_and_sort_plan(plan)

        # "face_beautify" phải bị loại, chỉ còn "sharpen"
        assert len(validated.actions) == 1
        assert validated.actions[0].operation == "sharpen"


def _region_action(**region_fields) -> dict:
    """Action gamma_correct trên vùng 'sky' kèm các trường vùng tùy ý từ VLM."""
    return {
        "region_id": "sky",
        "target_prompt": "sky",
        "detected_issue": "overexposed",
        "operation": "gamma_correct",
        "parameters": {"gamma": 0.8},
        **region_fields,
    }


def _vlm_plan(*actions: dict) -> str:
    return json.dumps({"iteration": 1, "reasoning": "Vùng trời bị chói.", "actions": list(actions)})


class TestVLMRegionFields:
    """M2-INT-04: một trường vùng sai từ VLM không được làm rỗng cả kế hoạch (SHIP âm thầm)."""

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key-for-test"})
    @patch("src.agent.vlm_diagnostician.genai")
    def test_out_of_range_region_values_are_clamped_not_dropped(self, mock_genai):
        response = _vlm_plan(
            _region_action(
                region_type="semantic",
                feather_radius=-3,
                expand_ratio=7.5,
                num_faces=0,
                box_threshold=2.0,
                instance_index=-1,
                bbox=[10.5, 0, "x", 4],
                binary_mask=[[1, 0]],
            )
        )
        mock_genai.Client.return_value.models = _create_mock_model(response)

        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        plan = validate_and_sort_plan(diagnose_and_plan(img, {}, iteration=1))

        assert len(plan.actions) == 1
        action = plan.actions[0]
        assert action.operation == "gamma_correct"
        assert action.region_type == "semantic"
        assert action.feather_radius == 5
        assert action.expand_ratio == 1.0
        assert action.num_faces == 1
        assert action.box_threshold == 1.0
        assert action.instance_index is None
        assert action.bbox is None
        assert action.binary_mask is None

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key-for-test"})
    @patch("src.agent.vlm_diagnostician.genai")
    def test_invalid_enum_values_are_reset(self, mock_genai):
        response = _vlm_plan(
            _region_action(
                region_type="polygon",
                face_mode="sam_refined",
                instance_selection="smallest",
                merge_policy="mean",
            )
        )
        mock_genai.Client.return_value.models = _create_mock_model(response)

        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        plan = diagnose_and_plan(img, {}, iteration=1)

        assert len(plan.actions) == 1
        action = plan.actions[0]
        assert action.region_type is None
        assert action.face_mode == "bbox"
        assert action.instance_selection == "all"
        assert action.merge_policy == "max"

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key-for-test"})
    @patch("src.agent.vlm_diagnostician.genai")
    def test_one_malformed_action_does_not_empty_the_plan(self, mock_genai):
        """Action thiếu trường bắt buộc bị loại riêng, các action khác giữ nguyên."""
        broken = _region_action()
        del broken["operation"]
        response = _vlm_plan(broken, "not an object", _region_action(target_prompt="face"))
        mock_genai.Client.return_value.models = _create_mock_model(response)

        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        plan = diagnose_and_plan(img, {}, iteration=1)

        assert [a.target_prompt for a in plan.actions] == ["face"]
        assert plan.reasoning == "Vùng trời bị chói."

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key-for-test"})
    @patch("src.agent.vlm_diagnostician.genai")
    def test_omitted_region_type_is_inferred(self, mock_genai):
        """Không có region_type → None, executor suy ra từ target_prompt (D2)."""
        from src.agent.executor import _infer_region_kind

        response = _vlm_plan(_region_action(target_prompt="face"), _region_action())
        mock_genai.Client.return_value.models = _create_mock_model(response)

        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        plan = validate_and_sort_plan(diagnose_and_plan(img, {}, iteration=1))

        assert [a.region_type for a in plan.actions] == [None, None]
        assert [_infer_region_kind(a) for a in plan.actions] == ["face", "semantic"]

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key-for-test"})
    @patch("src.agent.vlm_diagnostician.genai")
    def test_pipeline_does_not_ship_on_bad_region_value(self, mock_genai):
        """Trước M2-INT-04, feather_radius=-1 làm plan rỗng → vòng 1 SHIP ngay."""
        bad = _vlm_plan(
            {
                "region_id": "full_image",
                "target_prompt": "full",
                "region_type": "full",
                "feather_radius": -1,
                "detected_issue": "underexposed",
                "operation": "gamma_correct",
                "parameters": {"gamma": 1.5},
            }
        )
        _route_gemini(
            mock_genai,
            perceive=[_diagnosis_json(DARK_DEFECT), _diagnosis_json()],
            plan=[bad],
        )

        img = np.ones((64, 64, 3), dtype=np.uint8) * 60
        result_state = run_pipeline(image=img, max_iterations=2)

        first = result_state["history"][0]
        assert len(first.plan.actions) == 1
        assert first.plan.actions[0].feather_radius == 5


class TestVLMInvalidJSON:
    """Test khi VLM trả về chuỗi không phải JSON → exception handler phải bắt."""

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key-for-test"})
    @patch("src.agent.vlm_diagnostician.genai")
    def test_invalid_json_falls_back_safely(self, mock_genai, mock_gemini_response_invalid_json):
        mock_genai.Client.return_value.models = _create_mock_model(
            mock_gemini_response_invalid_json
        )

        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        plan = diagnose_and_plan(img, {}, iteration=1)

        # Phải fallback an toàn, không crash; metric rỗng → luật không sinh action
        assert isinstance(plan, TreatmentPlan)
        assert plan.source == "vlm_fallback"
        assert len(plan.actions) == 0


NOISY_DARK_METRICS = {"noise_level": "severe", "brightness_level": "underexposed"}


class TestVLMFailureFallsBackToRules:
    """Phase 0: lỗi Gemini không được biến thành plan rỗng (= SHIP âm thầm)."""

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key-for-test"})
    @patch("src.agent.vlm_diagnostician.genai")
    def test_invalid_json_uses_rule_plan(self, mock_genai):
        mock_genai.Client.return_value.models = _create_mock_model(MOCK_INVALID_JSON)

        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        plan = diagnose_and_plan(img, NOISY_DARK_METRICS, iteration=1)

        assert plan.source == "vlm_fallback"
        assert [a.operation for a in plan.actions] == ["denoise", "gamma_correct"]

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key-for-test"})
    @patch("src.agent.vlm_diagnostician.genai")
    def test_api_exception_uses_rule_plan(self, mock_genai):
        mock_genai.Client.return_value.models.generate_content.side_effect = RuntimeError(
            "503 UNAVAILABLE"
        )

        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        plan = diagnose_and_plan(img, NOISY_DARK_METRICS, iteration=1)

        assert plan.source == "vlm_fallback"
        assert len(plan.actions) == 2

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key-for-test"})
    @patch("src.agent.vlm_diagnostician.genai")
    def test_empty_response_text_uses_rule_plan(self, mock_genai):
        """Phản hồi bị chặn (safety) có text=None → không được coi là 'ảnh đã tốt'."""
        mock_genai.Client.return_value.models = _create_mock_model(None)

        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        plan = diagnose_and_plan(img, NOISY_DARK_METRICS, iteration=1)

        assert plan.source == "vlm_fallback"
        assert len(plan.actions) == 2

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key-for-test"})
    @patch("src.agent.vlm_diagnostician.genai")
    def test_all_invalid_actions_use_rule_plan(self, mock_genai):
        response = _vlm_plan({"operation": "denoise"}, "not an object")
        mock_genai.Client.return_value.models = _create_mock_model(response)

        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        plan = diagnose_and_plan(img, NOISY_DARK_METRICS, iteration=1)

        assert plan.source == "vlm_fallback"
        assert len(plan.actions) == 2

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key-for-test"})
    @patch("src.agent.vlm_diagnostician.genai", None)
    def test_missing_sdk_uses_rule_plan(self):
        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        plan = diagnose_and_plan(img, NOISY_DARK_METRICS, iteration=1)

        assert plan.source == "vlm_fallback"
        assert len(plan.actions) == 2

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key-for-test"})
    @patch("src.agent.vlm_diagnostician.genai")
    def test_valid_empty_plan_is_kept_empty(self, mock_genai):
        """VLM trả actions=[] hợp lệ → plan rỗng thật sự (ảnh đã tốt), không dùng luật."""
        empty = json.dumps({"iteration": 1, "reasoning": "Ảnh đã tốt", "actions": []})
        mock_genai.Client.return_value.models = _create_mock_model(empty)

        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        plan = diagnose_and_plan(img, NOISY_DARK_METRICS, iteration=1)

        assert plan.source == "vlm"
        assert plan.actions == []


class TestVLMRequest:
    """Phase 0: structured output, thu nhỏ ảnh và gửi kèm ảnh gốc từ vòng 2."""

    @staticmethod
    def _call_kwargs(mock_genai: MagicMock) -> dict:
        return mock_genai.Client.return_value.models.generate_content.call_args.kwargs

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key-for-test"})
    @patch("src.agent.vlm_diagnostician.genai")
    def test_structured_output_config(self, mock_genai):
        from src.agent.vlm_diagnostician import PLAN_RESPONSE_SCHEMA, SYSTEM_PROMPT

        mock_genai.Client.return_value.models = _create_mock_model(MOCK_VALID_RESPONSE)
        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        diagnose_and_plan(img, {}, iteration=1)

        config = self._call_kwargs(mock_genai)["config"]
        assert config.response_mime_type == "application/json"
        assert config.response_json_schema == PLAN_RESPONSE_SCHEMA
        assert config.system_instruction == SYSTEM_PROMPT

    def test_response_schema_matches_toolbox(self):
        from src.agent.planner import ALLOWED_OPERATIONS, PARAMETER_BOUNDS
        from src.agent.vlm_diagnostician import PLAN_RESPONSE_SCHEMA

        action = PLAN_RESPONSE_SCHEMA["properties"]["actions"]["items"]["properties"]
        assert set(action["operation"]["enum"]) == ALLOWED_OPERATIONS
        params = action["parameters"]["properties"]
        assert params["gamma"]["minimum"] == PARAMETER_BOUNDS["gamma_correct"]["gamma"]["min"]
        assert params["gamma"]["maximum"] == PARAMETER_BOUNDS["gamma_correct"]["gamma"]["max"]
        # method là hợp của enum denoise và sharpen
        assert {"bilateral", "unsharp_mask"} <= set(params["method"]["enum"])

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key-for-test"})
    @patch("src.agent.vlm_diagnostician.genai")
    def test_null_parameters_are_dropped(self, mock_genai):
        response = _vlm_plan(_region_action(parameters={"gamma": 0.8, "strength": None}))
        mock_genai.Client.return_value.models = _create_mock_model(response)

        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        plan = diagnose_and_plan(img, {}, iteration=1)

        assert plan.actions[0].parameters == {"gamma": 0.8}

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key-for-test"})
    @patch("src.agent.vlm_diagnostician.genai")
    def test_large_image_is_downscaled(self, mock_genai):
        from src.agent.vlm_diagnostician import VLM_MAX_SIDE

        mock_genai.Client.return_value.models = _create_mock_model(MOCK_VALID_RESPONSE)
        img = np.zeros((1500, 3000, 3), dtype=np.uint8)
        diagnose_and_plan(img, {}, iteration=1)

        contents = self._call_kwargs(mock_genai)["contents"]
        sent = contents[-1]
        assert sent.size == (VLM_MAX_SIDE, VLM_MAX_SIDE // 2)

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key-for-test"})
    @patch("src.agent.vlm_diagnostician.genai")
    def test_first_iteration_sends_only_current_image(self, mock_genai):
        mock_genai.Client.return_value.models = _create_mock_model(MOCK_VALID_RESPONSE)
        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        diagnose_and_plan(img, {}, iteration=1, original_image=img)

        contents = self._call_kwargs(mock_genai)["contents"]
        assert len(contents) == 2  # prompt + ảnh hiện tại

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key-for-test"})
    @patch("src.agent.vlm_diagnostician.genai")
    def test_later_iteration_sends_original_and_current(self, mock_genai):
        mock_genai.Client.return_value.models = _create_mock_model(MOCK_VALID_RESPONSE)
        original = np.zeros((64, 64, 3), dtype=np.uint8)
        current = np.full((64, 64, 3), 200, dtype=np.uint8)
        diagnose_and_plan(current, {}, iteration=2, original_image=original)

        contents = self._call_kwargs(mock_genai)["contents"]
        images = [c for c in contents if not isinstance(c, str)]
        assert len(images) == 2
        assert np.asarray(images[0]).max() == 0  # ảnh gốc trước
        assert np.asarray(images[1]).min() == 200  # ảnh hiện tại cuối cùng

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key-for-test"})
    @patch("src.agent.vlm_diagnostician.genai")
    def test_grayscale_image_is_accepted(self, mock_genai):
        mock_genai.Client.return_value.models = _create_mock_model(MOCK_VALID_RESPONSE)
        img = np.ones((64, 64, 1), dtype=np.uint8) * 128
        plan = diagnose_and_plan(img, {}, iteration=1)

        assert plan.source == "vlm"
        assert self._call_kwargs(mock_genai)["contents"][-1].mode == "L"


class TestRuleBasedPlan:
    """Phase 0: luật dự phòng thêm overexposed; blur/ám màu cố ý để VLM quyết định."""

    @staticmethod
    def _ops(metrics: dict) -> list:
        img = np.ones((32, 32, 3), dtype=np.uint8) * 128
        return [(a.operation, a.parameters) for a in diagnose_and_plan(img, metrics).actions]

    def test_overexposed_darkens(self):
        assert self._ops({"brightness_level": "overexposed"}) == [("gamma_correct", {"gamma": 0.8})]

    def test_blur_and_color_cast_are_left_to_the_vlm(self):
        assert self._ops({"blur_level": "severe_blur", "color_cast": "warm"}) == []

    def test_offline_plan_source(self):
        img = np.ones((32, 32, 3), dtype=np.uint8) * 128
        assert diagnose_and_plan(img, {}).source == "rule_based"


class TestVLMFallbackRuleBased:
    """Test nhánh fallback khi không có API key."""

    def test_fallback_with_noise(self):
        """Noise cao → fallback sinh action denoise."""
        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        plan = diagnose_and_plan(
            img, {"noise_level": "severe", "brightness_level": "normal"}, iteration=1
        )
        assert any(a.operation == "denoise" for a in plan.actions)

    def test_fallback_with_underexposed(self):
        """Thiếu sáng → fallback sinh action gamma_correct."""
        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        plan = diagnose_and_plan(
            img, {"noise_level": "clean", "brightness_level": "underexposed"}, iteration=1
        )
        assert any(a.operation == "gamma_correct" for a in plan.actions)

    def test_fallback_with_low_contrast(self):
        """Tương phản thấp → fallback sinh action clahe."""
        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        plan = diagnose_and_plan(
            img,
            {"noise_level": "clean", "brightness_level": "normal", "contrast_level": "low"},
            iteration=1,
        )
        assert any(a.operation == "clahe" for a in plan.actions)

    def test_fallback_clean_image(self):
        """Ảnh sạch → fallback không sinh action nào."""
        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        plan = diagnose_and_plan(
            img, {"noise_level": "clean", "brightness_level": "normal"}, iteration=1
        )
        assert len(plan.actions) == 0


class TestVLMEndToEnd:
    """Test toàn bộ pipeline khép kín chạy với mock VLM."""

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key-for-test"})
    @patch("src.agent.vlm_diagnostician.genai")
    def test_pipeline_with_mock_vlm(self, mock_genai):
        """Chạy run_pipeline với mock VLM sinh plan hợp lệ."""
        # Vòng 1: chẩn đoán có lỗi → plan hợp lệ; vòng 2: chẩn đoán sạch → plan rỗng (đã tốt)
        _route_gemini(
            mock_genai,
            perceive=[_diagnosis_json(DARK_DEFECT), _diagnosis_json()],
            plan=[MOCK_VALID_RESPONSE],
        )

        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        result_state = run_pipeline(image=img, max_iterations=2)

        assert result_state["current_image"] is not None
        assert result_state["current_image"].shape == (64, 64, 3)
        assert len(result_state["history"]) >= 1
        assert "intermediate_images" in result_state
        assert len(result_state["intermediate_images"]) >= 1
        assert result_state["decision"] in ["SHIP", "STOP_BEST_EFFORT"]
