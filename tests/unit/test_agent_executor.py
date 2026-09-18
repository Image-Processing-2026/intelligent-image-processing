"""
Unit tests for Module 4: Agent Tool Dispatcher & Robust Executor.
"""

from unittest.mock import patch

import numpy as np

from src.agent.executor import _validate_mask, execute_plan
from src.agent.state import RegionOperation, TreatmentPlan
from src.processing_engine.exposure_contrast import apply_gamma


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


def test_execute_plan_invalid_mask_fallback_to_full():
    """Mặt nạ không hợp lệ (ví dụ: toàn 0 hoặc sai kích thước) → fallback sang full image (mask=None)."""
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

    # Giả lập segment_by_prompt trả về mask sai kích thước (32, 32)
    invalid_mask = np.ones((32, 32), dtype=np.float32)
    with patch("src.agent.executor.segment_by_prompt", return_value=invalid_mask):
        with patch("src.agent.executor.apply_gamma", wraps=apply_gamma) as mock_gamma:
            result = execute_plan(img, plan)
            mock_gamma.assert_called_once()
            # Mask truyền vào apply_gamma phải là None do fallback
            assert mock_gamma.call_args[1]["mask"] is None

    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_execute_plan_segmentation_exception_handled():
    """segment_by_prompt ném exception → action bị bắt lỗi và bỏ qua an toàn."""
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
    with patch("src.agent.executor.segment_by_prompt", side_effect=ValueError("SAM model error")):
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
