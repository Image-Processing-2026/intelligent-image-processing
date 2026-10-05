"""
Unit tests for Module 4: Agent Tool Dispatcher & Robust Executor.
"""

from unittest.mock import patch

import numpy as np
import pytest

from src.agent.executor import _infer_region_kind, _validate_mask, execute_plan
from src.agent.state import RegionOperation, TreatmentPlan
from src.processing_engine.exposure_contrast import apply_gamma
from src.region_engine.detector import PromptSegmentationResult
from src.region_engine.segmentation_backend import GroundingDetection

# Executor không truyền resolver ngữ nghĩa → patch đường production trong controller.
_SEMANTIC_TARGET = "src.region_engine.controller.resolve_prompt_instances"


def _semantic_result(*masks: np.ndarray) -> PromptSegmentationResult:
    """Tạo kết quả GroundingDINO + MobileSAM giả với một detection cho mỗi instance mask."""
    detections = tuple(
        GroundingDetection((0.0, 0.0, 1.0, 1.0), 0.9, "object") for _ in range(len(masks))
    )
    return PromptSegmentationResult("object", "object.", detections, tuple(masks))


def test_validate_mask_direct():
    """Kiểm tra trực tiếp hàm _validate_mask với các trường hợp hợp lệ và không hợp lệ."""
    shape = (64, 64, 3)

    # Valid soft mask
    valid_mask = np.ones((64, 64), dtype=np.float32) * 0.5
    assert _validate_mask(valid_mask, shape) is True

    # Valid 3D single channel mask
    valid_mask_3d = np.ones((64, 64, 1), dtype=np.float32) * 0.8
    assert _validate_mask(valid_mask_3d, shape) is True

    # None mask
    assert _validate_mask(None, shape) is False

    # Not ndarray
    assert _validate_mask([1, 2, 3], shape) is False

    # Mismatched shape
    mismatched_mask = np.ones((32, 32), dtype=np.float32)
    assert _validate_mask(mismatched_mask, shape) is False

    # Wrong dtype (uint8 instead of float32)
    uint8_mask = np.ones((64, 64), dtype=np.uint8) * 255
    assert _validate_mask(uint8_mask, shape) is False

    # All zeros
    zero_mask = np.zeros((64, 64), dtype=np.float32)
    assert _validate_mask(zero_mask, shape) is False

    # Out of bounds (> 1.0)
    over_mask = np.ones((64, 64), dtype=np.float32) * 1.5
    assert _validate_mask(over_mask, shape) is False

    # Out of bounds (< 0.0)
    neg_mask = np.ones((64, 64), dtype=np.float32) * -0.1
    assert _validate_mask(neg_mask, shape) is False

    # Contains NaN
    nan_mask = np.ones((64, 64), dtype=np.float32) * 0.5
    nan_mask[0, 0] = np.nan
    assert _validate_mask(nan_mask, shape) is False

    # 1D array
    oned_mask = np.ones((64,), dtype=np.float32)
    assert _validate_mask(oned_mask, shape) is False


def test_execute_plan_valid_action():
    """Action hợp lệ → ảnh được xử lý thành công."""
    img = np.ones((64, 64, 3), dtype=np.uint8) * 100
    plan = TreatmentPlan(
        iteration=1,
        reasoning="Test",
        actions=[
            RegionOperation(
                region_id="full",
                target_prompt="full",
                detected_issue="low_contrast",
                operation="clahe",
                parameters={"clip_limit": 2.0},
            )
        ],
    )
    result = execute_plan(img, plan)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_execute_plan_all_operations():
    """Kiểm tra từng operation cơ bản chạy thành công qua execute_plan."""
    img = np.ones((64, 64, 3), dtype=np.uint8) * 128
    plan = TreatmentPlan(
        iteration=1,
        reasoning="Test all ops",
        actions=[
            RegionOperation(
                region_id="1",
                target_prompt="full",
                detected_issue="noise",
                operation="denoise",
                parameters={"method": "bilateral", "strength": 1.0},
            ),
            RegionOperation(
                region_id="2",
                target_prompt="all",
                detected_issue="dark",
                operation="gamma_correct",
                parameters={"gamma": 1.2},
            ),
            RegionOperation(
                region_id="3",
                target_prompt="toàn bộ",
                detected_issue="contrast",
                operation="clahe",
                parameters={"clip_limit": 2.0},
            ),
            RegionOperation(
                region_id="4",
                target_prompt="full_image",
                detected_issue="blur",
                operation="sharpen",
                parameters={"method": "unsharp_mask", "amount": 1.0},
            ),
            RegionOperation(
                region_id="5",
                target_prompt="full",
                detected_issue="color",
                operation="color_correct",
                parameters={"saturation_scale": 1.1, "temperature_shift": 0.1},
            ),
        ],
    )
    result = execute_plan(img, plan)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_execute_plan_survives_module3_error():
    """Module 3 ném lỗi → pipeline không crash, trả về ảnh ban đầu của bước đó."""
    img = np.ones((64, 64, 3), dtype=np.uint8) * 128
    plan = TreatmentPlan(
        iteration=1,
        reasoning="Test crash recovery",
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
    with patch("src.agent.executor.apply_denoise", side_effect=RuntimeError("Simulated crash")):
        result = execute_plan(img, plan)
    assert result.shape == img.shape
    assert np.array_equal(result, img)


def test_execute_plan_survives_multiple_actions_with_partial_failure():
    """Một action bị lỗi, các action còn lại vẫn được thực thi bình thường."""
    img = np.ones((64, 64, 3), dtype=np.uint8) * 100
    plan = TreatmentPlan(
        iteration=1,
        reasoning="Test partial failure",
        actions=[
            RegionOperation(
                region_id="fail_op",
                target_prompt="full",
                detected_issue="noise",
                operation="denoise",
                parameters={"strength": 1.0},
            ),
            RegionOperation(
                region_id="success_op",
                target_prompt="full",
                detected_issue="dark",
                operation="gamma_correct",
                parameters={"gamma": 2.0},
            ),
        ],
    )
    with patch("src.agent.executor.apply_denoise", side_effect=RuntimeError("Fail denoise")):
        result = execute_plan(img, plan)

    # Gamma=2.0 làm sáng ảnh (100 -> sáng hơn)
    assert result.shape == img.shape
    assert result.dtype == np.uint8
    assert not np.array_equal(result, img)


def test_execute_plan_no_face_detected():
    """detect_faces trả về rỗng → action bị bỏ qua an toàn."""
    img = np.ones((64, 64, 3), dtype=np.uint8) * 128
    plan = TreatmentPlan(
        iteration=1,
        reasoning="Test no face",
        actions=[
            RegionOperation(
                region_id="face",
                target_prompt="face",
                detected_issue="underexposed",
                operation="gamma_correct",
                parameters={"gamma": 1.3},
            )
        ],
    )
    with patch("src.agent.executor.detect_faces", return_value=[]):
        result = execute_plan(img, plan)
    assert result.shape == img.shape
    assert np.array_equal(result, img)


def test_execute_plan_face_detected_vietnamese_target():
    """detect_faces trả về mask với target='khuôn mặt' → xử lý trên mask đó."""
    img = np.ones((64, 64, 3), dtype=np.uint8) * 128
    fake_face_mask = np.ones((64, 64), dtype=np.float32)
    plan = TreatmentPlan(
        iteration=1,
        reasoning="Test face in Vietnamese",
        actions=[
            RegionOperation(
                region_id="face",
                target_prompt="khuôn mặt",
                detected_issue="underexposed",
                operation="gamma_correct",
                parameters={"gamma": 1.5},
            )
        ],
    )
    with patch("src.agent.executor.detect_faces", return_value=[fake_face_mask]):
        result = execute_plan(img, plan)
    assert result.shape == img.shape
    assert result.dtype == np.uint8
    assert not np.array_equal(result, img)


def test_execute_plan_invalid_mask_uses_heuristic_region():
    """Mask ngữ nghĩa sai kích thước cho 'sky' → dùng heuristic góc trên (D1), không xử lý toàn ảnh."""
    img = np.ones((64, 64, 3), dtype=np.uint8) * 100
    plan = TreatmentPlan(
        iteration=1,
        reasoning="Test invalid mask fallback",
        actions=[
            RegionOperation(
                region_id="sky_region",
                target_prompt="sky",
                detected_issue="dark",
                operation="gamma_correct",
                parameters={"gamma": 2.0},
            )
        ],
    )

    # Giả lập backend ngữ nghĩa trả về instance mask sai kích thước (32, 32)
    invalid = _semantic_result(np.ones((32, 32), dtype=bool))
    with patch(_SEMANTIC_TARGET, return_value=invalid):
        with patch("src.agent.executor.apply_gamma", wraps=apply_gamma) as mock_gamma:
            result = execute_plan(img, plan)
            mock_gamma.assert_called_once()
            mask = mock_gamma.call_args[1]["mask"]

    # Mask heuristic: phủ nửa trên, nửa dưới gần như không bị tác động
    assert mask is not None
    assert mask.shape == (64, 64)
    assert mask[:16].mean() > 0.9
    assert mask[48:].mean() < 0.1
    assert result.shape == img.shape
    assert result.dtype == np.uint8
    assert result[:8].mean() > result[56:].mean()


def test_execute_plan_segmentation_exception_handled():
    """Backend ngữ nghĩa ném exception → action bị bắt lỗi và bỏ qua an toàn."""
    img = np.ones((64, 64, 3), dtype=np.uint8) * 128
    plan = TreatmentPlan(
        iteration=1,
        reasoning="Test segment exception",
        actions=[
            RegionOperation(
                region_id="tree_region",
                target_prompt="tree",
                detected_issue="noise",
                operation="denoise",
                parameters={"strength": 1.0},
            )
        ],
    )
    with patch(_SEMANTIC_TARGET, side_effect=ValueError("SAM model error")):
        result = execute_plan(img, plan)
    assert result.shape == img.shape
    assert np.array_equal(result, img)


def test_execute_plan_clips_and_casts_uint8():
    """Kết quả từ filter là float ngoài [0, 255] → executor tự động clip và cast về np.uint8."""
    img = np.ones((64, 64, 3), dtype=np.uint8) * 128
    plan = TreatmentPlan(
        iteration=1,
        reasoning="Test clipping",
        actions=[
            RegionOperation(
                region_id="full",
                target_prompt="full",
                detected_issue="contrast",
                operation="clahe",
                parameters={},
            )
        ],
    )

    # Giả lập apply_clahe trả về float array với giá trị > 255 và < 0
    fake_output = np.ones((64, 64, 3), dtype=np.float64) * 300.0
    fake_output[0, 0, 0] = -50.0

    with patch("src.agent.executor.apply_clahe", return_value=fake_output):
        result = execute_plan(img, plan)

    assert result.dtype == np.uint8
    assert result[0, 0, 0] == 0
    assert result[1, 1, 1] == 255


def test_execute_plan_unsupported_operation():
    """Operation không nằm trong danh mục hỗ trợ → bỏ qua action an toàn."""
    img = np.ones((64, 64, 3), dtype=np.uint8) * 128
    plan = TreatmentPlan(
        iteration=1,
        reasoning="Test unsupported op",
        actions=[
            RegionOperation(
                region_id="1",
                target_prompt="full",
                detected_issue="magic",
                operation="deep_fake_magic",
                parameters={},
            )
        ],
    )
    result = execute_plan(img, plan)
    assert result.shape == img.shape
    assert np.array_equal(result, img)


def test_execute_plan_empty_or_none():
    """Kế hoạch rỗng hoặc None → trả về bản sao uint8 của ảnh ban đầu."""
    img = np.ones((64, 64, 3), dtype=np.uint8) * 128
    assert np.array_equal(execute_plan(img, None), img)

    empty_plan = TreatmentPlan(iteration=1, reasoning="", actions=[])
    assert np.array_equal(execute_plan(img, empty_plan), img)


def test_execute_plan_semantic_backend_unavailable_uses_heuristic():
    """Không có trọng số GroundingDINO/MobileSAM (mặc định trong CI) → 'sky' dùng heuristic."""
    img = np.ones((64, 64, 3), dtype=np.uint8) * 100
    plan = TreatmentPlan(
        iteration=1,
        reasoning="Real segment_by_prompt without model assets",
        actions=[
            RegionOperation(
                region_id="sky",
                target_prompt="bầu trời",
                region_type="semantic",
                detected_issue="dark",
                operation="gamma_correct",
                parameters={"gamma": 2.0},
            )
        ],
    )
    with patch.dict("os.environ", {"REGION_DINO_MODEL_PATH": "does/not/exist"}):
        result = execute_plan(img, plan)
    assert result[:8].mean() > 100
    assert result[56:].mean() < 105


def test_execute_plan_unknown_prompt_without_backend_is_skipped():
    """Prompt không có heuristic và backend không khả dụng → bỏ qua, không xử lý toàn ảnh (D1)."""
    img = np.ones((64, 64, 3), dtype=np.uint8) * 100
    plan = TreatmentPlan(
        iteration=1,
        reasoning="Unknown prompt",
        actions=[
            RegionOperation(
                region_id="dog",
                target_prompt="dog",
                detected_issue="dark",
                operation="gamma_correct",
                parameters={"gamma": 2.0},
            )
        ],
    )
    with patch.dict("os.environ", {"REGION_DINO_MODEL_PATH": "does/not/exist"}):
        result = execute_plan(img, plan)
    assert np.array_equal(result, img)


def _two_instances() -> tuple[np.ndarray, np.ndarray]:
    """Hai instance: nhỏ ở góc trên-trái (8x8) và lớn ở nửa dưới (32x64)."""
    small = np.zeros((64, 64), dtype=bool)
    small[:8, :8] = True
    large = np.zeros((64, 64), dtype=bool)
    large[32:, :] = True
    return small, large


def _semantic_action(target_prompt: str = "dog", **region_fields) -> RegionOperation:
    return RegionOperation(
        region_id=target_prompt,
        target_prompt=target_prompt,
        region_type="semantic",
        detected_issue="dark",
        operation="gamma_correct",
        parameters={"gamma": 2.0},
        **region_fields,
    )


@pytest.mark.parametrize(
    ("region_fields", "small_edited", "large_edited"),
    [
        ({}, True, True),
        ({"instance_selection": "largest"}, False, True),
        ({"instance_selection": "index", "instance_index": 0}, True, False),
    ],
)
def test_execute_plan_semantic_instance_selection(region_fields, small_edited, large_edited):
    """Đường ngữ nghĩa production tôn trọng instance_selection/instance_index (M2-INT-03)."""
    img = np.ones((64, 64, 3), dtype=np.uint8) * 100
    plan = TreatmentPlan(iteration=1, reasoning="t", actions=[_semantic_action(**region_fields)])
    with patch(_SEMANTIC_TARGET, return_value=_semantic_result(*_two_instances())) as fake:
        result = execute_plan(img, plan)

    fake.assert_called_once()
    assert bool(result[2, 2].mean() > 100) is small_edited
    assert bool(result[60, 32].mean() > 100) is large_edited
    # Vùng không thuộc instance nào luôn giữ nguyên (không xử lý toàn ảnh)
    assert result[16, 48].mean() == 100


def test_execute_plan_semantic_instance_index_out_of_range_is_skipped():
    """instance_index vượt số instance → action bị bỏ qua, ảnh giữ nguyên."""
    img = np.ones((64, 64, 3), dtype=np.uint8) * 100
    action = _semantic_action(instance_selection="index", instance_index=5)
    plan = TreatmentPlan(iteration=1, reasoning="t", actions=[action])
    with patch(_SEMANTIC_TARGET, return_value=_semantic_result(*_two_instances())):
        result = execute_plan(img, plan)
    assert np.array_equal(result, img)


def test_execute_plan_semantic_backend_is_logged(caplog):
    """Backend tạo mask được ghi log để demo thấy vùng đến từ model hay heuristic."""
    img = np.ones((64, 64, 3), dtype=np.uint8) * 100
    plan = TreatmentPlan(iteration=1, reasoning="t", actions=[_semantic_action()])
    with caplog.at_level("INFO", logger="src.agent.executor"):
        with patch(_SEMANTIC_TARGET, return_value=_semantic_result(*_two_instances())):
            execute_plan(img, plan)
    assert "backend 'groundingdino+mobilesam'" in caplog.text


def test_execute_plan_heuristic_fallback_is_logged(caplog):
    """Fallback D1 được ghi nhận là backend 'heuristic', không phải 'geometry'."""
    img = np.ones((64, 64, 3), dtype=np.uint8) * 100
    action = _semantic_action(target_prompt="sky")
    plan = TreatmentPlan(iteration=1, reasoning="t", actions=[action])
    with caplog.at_level("INFO", logger="src.agent.executor"):
        with patch.dict("os.environ", {"REGION_DINO_MODEL_PATH": "does/not/exist"}):
            execute_plan(img, plan)
    assert "backend 'heuristic'" in caplog.text


@pytest.mark.parametrize(
    ("target", "region_type", "expected"),
    [
        ("face", None, "face"),
        ("khuôn mặt", "semantic", "face"),
        ("full", None, "full"),
        ("toàn bộ", "semantic", "full"),
        ("top", None, "spatial"),
        ("sky", None, "semantic"),
        ("sky", "spatial", "spatial"),
        ("anything", "bbox", "bbox"),
    ],
)
def test_infer_region_kind(target, region_type, expected):
    """region_type=None hoặc 'semantic' được suy ra từ target_prompt; loại tường minh khác giữ nguyên (D2)."""
    action = RegionOperation(
        region_id="r",
        target_prompt=target,
        region_type=region_type,
        detected_issue="x",
        operation="denoise",
    )
    assert _infer_region_kind(action) == expected
