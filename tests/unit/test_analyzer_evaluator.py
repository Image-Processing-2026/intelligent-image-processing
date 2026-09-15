"""
Unit tests cho Module 1: Analyzer & Evaluator — Stage 1.
Kiểm tra toàn diện các thuật toán đã triển khai: chuẩn hóa đầu vào, phân tích
độ sáng/tương phản/nhiễu/sắc nét, histogram RGB/HSV và phát hiện ám màu.
"""

import numpy as np
import pytest

from src.analyzer_evaluator import (
    EvaluationResult,
    TechnicalMetrics,
    analyze_image,
    evaluate_no_reference,
    evaluate_reference,
)
from src.analyzer_evaluator.analyzer import (
    _compute_histogram_stats,
    _detect_color_cast,
    _ensure_rgb_and_gray,
    _estimate_noise_immerkaer,
)
from tests.fixtures.synthetic_images import (
    create_color_cast_image,
    create_flat_image,
)

# =============================================================================
# STAGE 0: Kiểm tra Schema
# =============================================================================


def test_schema_instantiation():
    """Kiểm tra việc khởi tạo và hợp lệ hóa dữ liệu của TechnicalMetrics & EvaluationResult."""
    metrics = TechnicalMetrics(
        brightness_mean=128.0,
        brightness_level="normal",
        contrast_std=50.0,
        contrast_level="normal",
        noise_variance=2.0,
        noise_level="clean",
        sharpness_laplacian_var=350.0,
        blur_level="sharp",
        color_cast="none",
        histogram_stats={"gray_min": 10, "gray_max": 240},
    )
    assert metrics.brightness_mean == 128.0
    assert metrics.blur_level == "sharp"

    eval_result = EvaluationResult(
        iteration=1,
        is_reference_eval=True,
        psnr=32.5,
        ssim=0.89,
        mse=12.4,
        technical_metrics=metrics,
        decision="SHIP",
    )
    assert eval_result.psnr == 32.5
    assert eval_result.decision == "SHIP"


# =============================================================================
# STAGE 1.1: Chuẩn hóa đầu vào _ensure_rgb_and_gray
# =============================================================================


def test_ensure_rgb_and_gray_from_rgb(flat_gray_image):
    """Ảnh RGB uint8 truyền qua không bị biến đổi."""
    rgb, gray = _ensure_rgb_and_gray(flat_gray_image)
    assert rgb.shape == (100, 100, 3)
    assert gray.shape == (100, 100)
    assert rgb.dtype == np.uint8
    assert gray.dtype == np.uint8


def test_ensure_rgb_and_gray_from_grayscale():
    """Ảnh Grayscale (H, W) được chuyển sang RGB (H, W, 3) đúng cách."""
    gray_input = np.full((50, 50), 100, dtype=np.uint8)
    rgb, gray = _ensure_rgb_and_gray(gray_input)
    assert rgb.shape == (50, 50, 3)
    assert gray.shape == (50, 50)
    # Tất cả 3 kênh phải bằng nhau sau chuyển đổi từ grayscale
    assert np.all(rgb[:, :, 0] == rgb[:, :, 1])
    assert np.all(rgb[:, :, 1] == rgb[:, :, 2])


def test_ensure_rgb_and_gray_from_rgba():
    """Ảnh RGBA (H, W, 4): kênh alpha bị tách bỏ đúng cách."""
    rgba = np.full((50, 50, 4), 150, dtype=np.uint8)
    rgba[:, :, 3] = 0  # Đặt alpha = 0 để kiểm tra bị loại bỏ
    rgb, gray = _ensure_rgb_and_gray(rgba)
    assert rgb.shape == (50, 50, 3)
    # Các kênh RGB phải là 150, không bị ảnh hưởng bởi alpha
    assert np.all(rgb[:, :, 0] == 150)


def test_ensure_rgb_and_gray_from_float():
    """Ảnh float32 [0.0, 1.0] được chuyển đổi đúng về uint8 [0, 255]."""
    float_img = np.full((50, 50, 3), 0.5, dtype=np.float32)
    rgb, gray = _ensure_rgb_and_gray(float_img)
    assert rgb.dtype == np.uint8
    assert 120 < int(np.mean(rgb)) < 135  # 0.5 * 255 ≈ 127


def test_ensure_rgb_and_gray_invalid_shape():
    """Ảnh có shape không hợp lệ phải raise ValueError."""
    bad_img = np.zeros((50, 50, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported image shape"):
        _ensure_rgb_and_gray(bad_img)


def test_analyze_image_raises_on_empty():
    """Ảnh rỗng phải raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        analyze_image(np.array([]))


# =============================================================================
# STAGE 1.2: Độ sáng & Phơi sáng
# =============================================================================


def test_brightness_flat_normal(flat_gray_image):
    """Ảnh xám 128 phải có brightness_level='normal'."""
    m = analyze_image(flat_gray_image)
    assert m.brightness_mean == 128.0
    assert m.brightness_level == "normal"


def test_brightness_underexposed(black_image):
    """Ảnh toàn đen phải là 'underexposed'."""
    m = analyze_image(black_image)
    assert m.brightness_mean == 0.0
    assert m.brightness_level == "underexposed"


def test_brightness_overexposed(white_image):
    """Ảnh toàn trắng phải là 'overexposed'."""
    m = analyze_image(white_image)
    assert m.brightness_mean == 255.0
    assert m.brightness_level == "overexposed"


def test_brightness_underexposed_by_fixture(underexposed_image):
    """Ảnh fixture thiếu sáng phải phân loại 'underexposed'."""
    m = analyze_image(underexposed_image)
    assert m.brightness_level == "underexposed"
    assert m.brightness_mean < 70.0


def test_brightness_overexposed_by_fixture(overexposed_image):
    """Ảnh fixture cháy sáng phải phân loại 'overexposed'."""
    m = analyze_image(overexposed_image)
    assert m.brightness_level == "overexposed"


def test_histogram_stats_has_required_keys(flat_gray_image):
    """histogram_stats phải chứa đủ 12 trường theo kế hoạch Stage 1.6."""
    m = analyze_image(flat_gray_image)
    required_keys = {
        "gray_min",
        "gray_max",
        "gray_skewness",
        "highlight_clip_ratio",
        "shadow_clip_ratio",
        "dynamic_range",
        "r_mean",
        "g_mean",
        "b_mean",
        "r_std",
        "g_std",
        "b_std",
        "dominant_hue",
    }
    assert required_keys.issubset(set(m.histogram_stats.keys())), (
        f"Missing keys: {required_keys - set(m.histogram_stats.keys())}"
    )


# =============================================================================
# STAGE 1.3: Độ tương phản
# =============================================================================


def test_contrast_flat_is_zero(flat_gray_image):
    """Ảnh đồng nhất phải có contrast_std=0.0 và level='low'."""
    m = analyze_image(flat_gray_image)
    assert m.contrast_std == 0.0
    assert m.contrast_level == "low"


def test_contrast_checkerboard_is_high(checkerboard_image):
    """Ảnh bàn cờ phải có contrast_std > 80 và level='high'."""
    m = analyze_image(checkerboard_image)
    assert m.contrast_std > 80.0
    assert m.contrast_level == "high"


# =============================================================================
# STAGE 1.4: Ước lượng nhiễu (Immerkær)
# =============================================================================


def test_noise_clean_on_flat(flat_gray_image):
    """Ảnh phẳng đồng nhất phải có noise_level='clean'."""
    m = analyze_image(flat_gray_image)
    assert m.noise_level == "clean"
    assert m.noise_variance < 3.0


def test_noise_detected_on_gaussian_noisy(noisy_image):
    """Ảnh có nhiễu Gauss sigma=20 phải được phát hiện."""
    m = analyze_image(noisy_image)
    assert m.noise_variance > 3.0
    assert m.noise_level in ["low", "medium", "severe"]


def test_immerkaer_fallback_tiny_image():
    """Ảnh quá nhỏ (<5x5) phải dùng fallback Median Difference, không crash."""
    tiny = np.full((3, 3), 128, dtype=np.uint8)
    result = _estimate_noise_immerkaer(tiny)
    assert isinstance(result, float)
    assert result >= 0.0


def test_immerkaer_on_flat_image():
    """Immerkær trên ảnh phẳng nên trả về gần 0."""
    gray_flat = np.full((100, 100), 128, dtype=np.uint8)
    sigma = _estimate_noise_immerkaer(gray_flat)
    assert sigma < 1.0


# =============================================================================
# STAGE 1.5: Độ sắc nét & Mờ (Laplacian + Tenengrad)
# =============================================================================


def test_blur_level_flat_is_severe(flat_gray_image):
    """Ảnh phẳng hoàn toàn phải là 'severe_blur' (Laplacian var = 0)."""
    m = analyze_image(flat_gray_image)
    assert m.sharpness_laplacian_var == 0.0
    assert m.blur_level == "severe_blur"


def test_blur_level_checkerboard_is_sharp(checkerboard_image):
    """Ảnh bàn cờ sắc nét phải là 'sharp'."""
    m = analyze_image(checkerboard_image)
    assert m.sharpness_laplacian_var >= 300.0
    assert m.blur_level == "sharp"


def test_blur_level_blurred_is_lower(blurred_checkerboard_image, checkerboard_image):
    """Ảnh bị làm mờ phải có Laplacian var thấp hơn ảnh gốc sắc nét."""
    m_sharp = analyze_image(checkerboard_image)
    m_blur = analyze_image(blurred_checkerboard_image)
    assert m_blur.sharpness_laplacian_var < m_sharp.sharpness_laplacian_var


# =============================================================================
# STAGE 1.6: Phát hiện ám màu CIE LAB
# =============================================================================


def test_color_cast_none_on_gray(flat_gray_image):
    """Ảnh xám trung tính không có ám màu."""
    m = analyze_image(flat_gray_image)
    assert m.color_cast == "none"


def test_color_cast_warm(warm_cast_image):
    """Ảnh thiên đỏ/vàng phải phát hiện 'warm'."""
    m = analyze_image(warm_cast_image)
    assert m.color_cast == "warm"


def test_color_cast_cool(cool_cast_image):
    """Ảnh thiên xanh lam phải phát hiện 'cool'."""
    m = analyze_image(cool_cast_image)
    assert m.color_cast == "cool"


def test_color_cast_greenish():
    """Ảnh thiên xanh lá phải phát hiện 'greenish'."""
    base = create_flat_image(val=128)
    green_img = create_color_cast_image(base, cast_type="greenish", intensity=50)
    m = analyze_image(green_img)
    assert m.color_cast == "greenish"


def test_detect_color_cast_direct():
    """Kiểm tra trực tiếp hàm _detect_color_cast."""
    base = create_flat_image(val=128)
    warm = create_color_cast_image(base, cast_type="warm", intensity=50)
    cool = create_color_cast_image(base, cast_type="cool", intensity=50)
    assert _detect_color_cast(warm) == "warm"
    assert _detect_color_cast(cool) == "cool"


# =============================================================================
# STAGE 1.6: Histogram stats
# =============================================================================


def test_histogram_stats_flat_gray():
    """Kiểm tra histogram_stats trực tiếp trên ảnh xám phẳng."""
    gray = np.full((100, 100), 128, dtype=np.uint8)
    rgb = np.stack([gray, gray, gray], axis=2)
    stats = _compute_histogram_stats(rgb, gray)

    assert stats["gray_min"] == 128
    assert stats["gray_max"] == 128
    assert abs(stats["gray_skewness"]) < 1e-3  # Phải gần 0 vì đồng nhất
    assert stats["highlight_clip_ratio"] == 0.0
    assert stats["shadow_clip_ratio"] == 0.0
    assert stats["dynamic_range"] == 0
    assert stats["r_mean"] == pytest.approx(128.0, abs=1.0)


def test_histogram_stats_full_range():
    """Ảnh có pixel từ 0 đến 255 phải có dynamic_range = ~254."""
    gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
    rgb = np.stack([gray, gray, gray], axis=2)
    stats = _compute_histogram_stats(rgb, gray)
    assert stats["gray_min"] == 0
    assert stats["gray_max"] == 255
    assert stats["dynamic_range"] > 200  # P99 - P1


def test_histogram_stats_nan_guard_flat():
    """Ảnh phẳng không được trả về NaN cho skewness."""
    import math

    gray = np.full((50, 50), 200, dtype=np.uint8)
    rgb = np.stack([gray, gray, gray], axis=2)
    stats = _compute_histogram_stats(rgb, gray)
    assert not math.isnan(stats["gray_skewness"])
    assert stats["gray_skewness"] == 0.0


# =============================================================================
# Edge Cases: Grayscale và RGBA qua analyze_image
# =============================================================================


def test_analyze_image_grayscale_input():
    """analyze_image không được crash khi nhận ảnh grayscale (H, W)."""
    gray_img = np.full((80, 80), 150, dtype=np.uint8)
    m = analyze_image(gray_img)
    assert m.brightness_mean > 0
    assert isinstance(m, TechnicalMetrics)


def test_analyze_image_rgba_input():
    """analyze_image không được crash khi nhận ảnh RGBA (H, W, 4)."""
    rgba = np.full((80, 80, 4), 100, dtype=np.uint8)
    m = analyze_image(rgba)
    assert isinstance(m, TechnicalMetrics)


def test_analyze_image_float32_input():
    """analyze_image không được crash khi nhận ảnh float32 [0, 1]."""
    float_img = np.full((80, 80, 3), 0.5, dtype=np.float32)
    m = analyze_image(float_img)
    assert isinstance(m, TechnicalMetrics)
    assert 100 < m.brightness_mean < 160


# =============================================================================
# Kiểm tra tích hợp nhanh với reference_eval và no_reference_eval
# =============================================================================


def test_reference_evaluation():
    """Kiểm tra tính toán PSNR/SSIM trên 2 ảnh giống hệt nhau."""
    img1 = np.ones((100, 100, 3), dtype=np.uint8) * 100
    img2 = np.ones((100, 100, 3), dtype=np.uint8) * 100

    eval_res = evaluate_reference(img1, img2)
    assert eval_res["psnr"] >= 99.0
    assert eval_res["ssim"] == 1.0
    assert eval_res["mse"] == 0.0


def test_no_reference_evaluation(flat_gray_image):
    """Kiểm tra tính toán đánh giá không cần ảnh gốc."""
    res = evaluate_no_reference(flat_gray_image)
    assert "current_brightness" in res
    assert "current_contrast" in res
    assert res["current_brightness"] == 128.0
