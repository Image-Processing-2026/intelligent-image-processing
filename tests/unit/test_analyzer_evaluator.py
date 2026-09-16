"""
Unit tests cho Module 1: Analyzer & Evaluator — Stage 0, 1, 2, 3, 3.5.
Kiểm tra toàn diện các thuật toán đã triển khai: chuẩn hóa đầu vào, phân tích
độ sáng/tương phản/nhiễu/sắc nét, histogram RGB/HSV, phát hiện ám màu,
Full-Reference (PSNR/SSIM/MSE), No-Reference NR-IQA, Delta Metrics Engine,
evaluate_quality() unified API và Synthetic Dataset Loader.
"""

import numpy as np
import pytest

from src.analyzer_evaluator import (
    EvaluationResult,
    TechnicalMetrics,
    analyze_image,
    evaluate_no_reference,
    evaluate_quality,
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


# =============================================================================
# STAGE 2: Full-Reference Evaluation (PSNR / SSIM / MSE)
# =============================================================================


def test_reference_eval_identical_images():
    """Hai ảnh hoàn toàn giống nhau phải có MSE=0, PSNR>=99, SSIM=1."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 100
    result = evaluate_reference(img, img.copy())

    assert isinstance(result, EvaluationResult)
    assert result.is_reference_eval is True
    assert result.mse == 0.0
    assert result.psnr >= 99.0
    assert result.ssim == pytest.approx(1.0, abs=1e-4)
    assert result.technical_metrics is not None


def test_reference_eval_inverted_images():
    """Hai ảnh cực đối nhau (0 vs 200) phải có PSNR thấp."""
    # pixel=0 vs pixel=200: MSE = 200^2 = 40000, PSNR ≈ 5.1 dB
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.full((100, 100, 3), 200, dtype=np.uint8)
    result = evaluate_reference(img1, img2)

    assert result.psnr < 10.0
    assert result.mse > 10000.0


def test_reference_eval_different_images_have_nonzero_mse():
    """Hai ảnh khác nhau phải có MSE > 0."""
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.full((100, 100, 3), 128, dtype=np.uint8)
    result = evaluate_reference(img1, img2)

    assert result.mse > 0.0
    assert result.psnr < 99.0


def test_reference_eval_uint8_underflow_protection():
    """
    Đảm bảo không bị underflow uint8 khi trừ hai ảnh.
    50 - 100 trong uint8 sẽ wrap thành 206 nếu không ép float64.
    """
    # img1 pixel=50, img2 pixel=100: diff thực tế là 50, MSE = 50^2 = 2500
    img1 = np.full((10, 10, 3), 50, dtype=np.uint8)
    img2 = np.full((10, 10, 3), 100, dtype=np.uint8)
    result = evaluate_reference(img1, img2)

    # Nếu có underflow: (50-100) uint8 = 206, MSE sẽ là 206^2 = 42436 → sai
    # Kết quả đúng: MSE = (50-100)^2 = 2500
    assert result.mse == pytest.approx(2500.0, abs=1.0)


def test_reference_eval_shape_mismatch_auto_resize():
    """evaluate_reference phải tự động resize ground_truth khi kích thước lệch."""
    current = np.full((100, 100, 3), 128, dtype=np.uint8)
    ground_truth = np.full((110, 110, 3), 128, dtype=np.uint8)
    # Không được raise, phải resize và trả về kết quả hợp lệ
    result = evaluate_reference(current, ground_truth)
    assert isinstance(result, EvaluationResult)
    assert result.mse is not None


def test_reference_eval_returns_evaluation_result_schema():
    """evaluate_reference phải trả về EvaluationResult Pydantic model."""
    img = np.ones((50, 50, 3), dtype=np.uint8) * 200
    result = evaluate_reference(img, img.copy(), iteration=2)

    assert isinstance(result, EvaluationResult)
    assert result.iteration == 2
    assert result.is_reference_eval is True
    assert isinstance(result.technical_metrics, TechnicalMetrics)


def test_reference_eval_with_previous_image_delta_metrics():
    """Khi có previous_image, delta_metrics phải được tính đúng."""
    rng = np.random.default_rng(0)
    ground_truth = np.full((80, 80, 3), 128, dtype=np.uint8)
    previous = np.clip(
        ground_truth.astype(np.int16) + rng.integers(-30, 30, ground_truth.shape),
        0,
        255,
    ).astype(np.uint8)
    current = np.clip(
        ground_truth.astype(np.int16) + rng.integers(-5, 5, ground_truth.shape),
        0,
        255,
    ).astype(np.uint8)

    result = evaluate_reference(current, ground_truth, previous_image=previous)

    assert "delta_psnr" in result.delta_metrics
    assert "delta_ssim" in result.delta_metrics
    assert "delta_mse" in result.delta_metrics
    # Current gần ground_truth hơn previous → PSNR tăng → delta_psnr > 0
    assert result.delta_metrics["delta_psnr"] > 0.0
    assert result.quality_improved is True


def test_reference_eval_quality_not_improved_when_worse():
    """Khi ảnh hiện tại tệ hơn vòng trước, quality_improved phải là False."""
    ground_truth = np.full((80, 80, 3), 128, dtype=np.uint8)
    previous = np.clip(
        ground_truth.astype(np.int16)
        + np.random.default_rng(1).integers(-3, 3, ground_truth.shape),
        0,
        255,
    ).astype(np.uint8)
    current = np.clip(
        ground_truth.astype(np.int16)
        + np.random.default_rng(2).integers(-60, 60, ground_truth.shape),
        0,
        255,
    ).astype(np.uint8)

    result = evaluate_reference(current, ground_truth, previous_image=previous)
    assert result.quality_improved is False


def test_reference_eval_raises_on_empty_input():
    """evaluate_reference phải raise ValueError khi ảnh đầu vào rỗng."""
    valid = np.ones((50, 50, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        evaluate_reference(np.array([]), valid)
    with pytest.raises(ValueError):
        evaluate_reference(valid, np.array([]))


def test_reference_eval_multichannel_ssim_rgb():
    """Multichannel SSIM phải hoạt động đúng trên ảnh RGB 3 kênh."""
    img_clean = np.zeros((100, 100, 3), dtype=np.uint8)
    img_clean[:, :, 0] = 200  # Thiên đỏ
    img_noisy = img_clean.copy()
    img_noisy[:, :, 1] = 100  # Thêm kênh xanh → khác biệt cấu trúc
    result = evaluate_reference(img_clean, img_noisy)
    # SSIM phải < 1 vì hai ảnh khác nhau trên kênh G
    assert result.ssim < 1.0


# =============================================================================
# STAGE 3: No-Reference Evaluation (NR-IQA + Delta Metrics)
# =============================================================================


def test_no_reference_eval_returns_evaluation_result(flat_gray_image):
    """evaluate_no_reference phải trả về EvaluationResult Pydantic model."""
    result = evaluate_no_reference(flat_gray_image)
    assert isinstance(result, EvaluationResult)
    assert result.is_reference_eval is False
    assert isinstance(result.technical_metrics, TechnicalMetrics)
    assert result.iteration == 1


def test_no_reference_eval_iteration_param(flat_gray_image):
    """Tham số iteration phải được truyền đúng vào EvaluationResult."""
    result = evaluate_no_reference(flat_gray_image, iteration=3)
    assert result.iteration == 3


def test_no_reference_eval_no_psnr_ssim(flat_gray_image):
    """evaluate_no_reference không được trả về PSNR/SSIM (tuân ADR-002)."""
    result = evaluate_no_reference(flat_gray_image)
    assert result.psnr is None
    assert result.ssim is None
    assert result.mse is None


def test_no_reference_eval_heuristic_score_in_valid_range(flat_gray_image):
    """Heuristic score phải nằm trong khoảng [0, 100]."""
    result = evaluate_no_reference(flat_gray_image)
    # Điểm heuristic được lưu trong vlm_feedback
    assert "Heuristic perceptual score" in result.vlm_feedback
    score_str = result.vlm_feedback.split(":")[1].strip().split("/")[0]
    score = float(score_str)
    assert 0.0 <= score <= 100.0


def test_no_reference_eval_perfect_image_has_high_score():
    """Ảnh cân bằng (brightness=128, contrast tốt) phải có score cao hơn ảnh cực đoan."""
    # Ảnh cân bằng: brightness~128, contrast tốt
    balanced = (
        np.random.default_rng(0).integers(80, 180, (100, 100, 3), dtype=np.uint8).astype(np.uint8)
    )
    # Ảnh toàn đen: brightness=0, contrast=0
    black = np.zeros((100, 100, 3), dtype=np.uint8)

    result_balanced = evaluate_no_reference(balanced)
    result_black = evaluate_no_reference(black)

    def extract_score(r: EvaluationResult) -> float:
        return float(r.vlm_feedback.split(":")[1].strip().split("/")[0])

    assert extract_score(result_balanced) > extract_score(result_black)


def test_no_reference_eval_noise_penalty_is_monotonic():
    """Score phải giảm khi noise_variance tăng (với cùng một ảnh nền)."""
    # Kiểm tra đơn điệu: score(low_noise) > score(high_noise) trên cùng base
    # Dùng ảnh có nội dung thực để tránh zero-sharpness confound

    # Base image: checkerboard 50/200 để có sharpness và contrast thực sự
    base = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            base[i, j] = 200 if (i // 10 + j // 10) % 2 == 0 else 50

    rng = np.random.default_rng(55)
    # Ảnh nhiễu nhẹ: sigma=1
    low_noise = np.clip(base.astype(np.int16) + rng.integers(-1, 1, base.shape), 0, 255).astype(
        np.uint8
    )
    # Ảnh nhiễu cực nặng: sigma=100 (nhiều gấp 100 lần)
    high_noise = np.clip(
        base.astype(np.int16) + rng.integers(-100, 100, base.shape), 0, 255
    ).astype(np.uint8)

    result_low = evaluate_no_reference(low_noise)
    result_high = evaluate_no_reference(high_noise)

    def extract_score(r: EvaluationResult) -> float:
        return float(r.vlm_feedback.split(":")[1].strip().split("/")[0])

    # noise_variance của high_noise phải lớn hơn hẳn
    assert (
        result_high.technical_metrics.noise_variance > result_low.technical_metrics.noise_variance
    )
    # Score phải giảm khi nhiễu tăng
    assert extract_score(result_high) < extract_score(result_low)


def test_no_reference_eval_delta_metrics_computed_with_previous(flat_gray_image, noisy_image):
    """Khi có previous_image, delta_metrics phải chứa 4 key chuẩn."""
    result = evaluate_no_reference(flat_gray_image, previous_image=noisy_image)
    assert "delta_contrast" in result.delta_metrics
    assert "delta_sharpness" in result.delta_metrics
    assert "delta_noise" in result.delta_metrics
    assert "delta_brightness" in result.delta_metrics


def test_no_reference_eval_delta_metrics_empty_without_previous(flat_gray_image):
    """Không có previous_image, delta_metrics phải là dict rỗng."""
    result = evaluate_no_reference(flat_gray_image)
    assert result.delta_metrics == {}


def test_no_reference_eval_denoise_quality_improved():
    """Ảnh sau khử nhiễu (ít nhiễu hơn) phải được đánh giá quality_improved=True."""
    rng = np.random.default_rng(42)
    base = np.full((100, 100, 3), 128, dtype=np.uint8)
    # Ảnh cũ: nhiều nhiễu (sigma=30)
    prev_noisy = np.clip(base.astype(np.int16) + rng.integers(-30, 30, base.shape), 0, 255).astype(
        np.uint8
    )
    # Ảnh hiện tại: ít nhiễu hơn nhiều (sigma=3)
    curr_clean = np.clip(base.astype(np.int16) + rng.integers(-3, 3, base.shape), 0, 255).astype(
        np.uint8
    )

    result = evaluate_no_reference(curr_clean, previous_image=prev_noisy)
    # Delta noise âm → noise giảm → cải thiện
    assert result.delta_metrics["delta_noise"] < -1.5
    assert result.quality_improved is True


def test_no_reference_eval_severe_noise_increase_degrades():
    """Nhiễu tăng đột biến > 10.0 phải dẫn tới quality_improved=False."""
    base = np.full((100, 100, 3), 128, dtype=np.uint8)
    # Ảnh cũ: sạch
    prev_clean = base.copy()
    # Ảnh hiện tại: nhiễu rất nặng (sigma >> 15)
    rng = np.random.default_rng(99)
    curr_very_noisy = np.clip(
        base.astype(np.int16) + rng.integers(-80, 80, base.shape), 0, 255
    ).astype(np.uint8)

    result = evaluate_no_reference(curr_very_noisy, previous_image=prev_clean)
    assert result.quality_improved is False


def test_no_reference_eval_raises_on_empty():
    """evaluate_no_reference phải raise ValueError khi ảnh rỗng."""
    with pytest.raises(ValueError, match="empty"):
        evaluate_no_reference(np.array([]))


def test_no_reference_eval_grayscale_input():
    """evaluate_no_reference không crash với ảnh grayscale."""
    gray = np.full((80, 80), 128, dtype=np.uint8)
    result = evaluate_no_reference(gray)
    assert isinstance(result, EvaluationResult)


def test_no_reference_eval_float32_input():
    """evaluate_no_reference không crash với ảnh float32 [0, 1]."""
    float_img = np.full((60, 60, 3), 0.5, dtype=np.float32)
    result = evaluate_no_reference(float_img)
    assert isinstance(result, EvaluationResult)


# =============================================================================
# STAGE 3.5.2: evaluate_quality() — Unified Entry Point (ADR-002)
# =============================================================================


def test_evaluate_quality_routes_to_reference_when_synthetic():
    """is_synthetic=True + original_image → phải gọi evaluate_reference."""
    img = np.ones((60, 60, 3), dtype=np.uint8) * 150
    result = evaluate_quality(img, original_image=img.copy(), is_synthetic=True)
    assert result.is_reference_eval is True
    assert result.psnr is not None
    assert result.ssim is not None


def test_evaluate_quality_routes_to_no_reference_for_real_image():
    """is_synthetic=False → phải gọi evaluate_no_reference dù có original_image."""
    img = np.ones((60, 60, 3), dtype=np.uint8) * 128
    result = evaluate_quality(img, original_image=img.copy(), is_synthetic=False)
    assert result.is_reference_eval is False
    assert result.psnr is None
    assert result.ssim is None


def test_evaluate_quality_no_reference_without_original():
    """is_synthetic=False, không có original_image → No-Reference cơ bản."""
    img = np.ones((60, 60, 3), dtype=np.uint8) * 100
    result = evaluate_quality(img)
    assert result.is_reference_eval is False
    assert result.delta_metrics == {}


def test_evaluate_quality_synthetic_no_original_falls_back_to_nr():
    """is_synthetic=True nhưng không có original_image → fallback No-Reference."""
    img = np.ones((60, 60, 3), dtype=np.uint8) * 100
    result = evaluate_quality(img, original_image=None, is_synthetic=True)
    assert result.is_reference_eval is False


def test_evaluate_quality_iteration_param_passed_through():
    """Tham số iteration phải được truyền qua đúng."""
    img = np.ones((60, 60, 3), dtype=np.uint8) * 100
    result = evaluate_quality(img, iteration=2)
    assert result.iteration == 2


# =============================================================================
# STAGE 3.5.1: synthetic_loader — load_synthetic_pairs & run_benchmark_suite
# =============================================================================


def test_load_synthetic_pairs_nonexistent_dirs(tmp_path):
    """Thư mục không tồn tại → trả về list rỗng, không crash."""
    from src.analyzer_evaluator.synthetic_loader import load_synthetic_pairs

    pairs = load_synthetic_pairs(
        clean_dir=str(tmp_path / "clean"),
        degraded_dir=str(tmp_path / "degraded"),
    )
    assert pairs == []


def test_load_synthetic_pairs_finds_pairs(tmp_path):
    """Đặt đúng tên file → load_synthetic_pairs tìm đúng cặp."""
    import cv2

    from src.analyzer_evaluator.synthetic_loader import load_synthetic_pairs

    clean_dir = tmp_path / "clean"
    degraded_dir = tmp_path / "degraded"
    clean_dir.mkdir()
    degraded_dir.mkdir()

    # Tạo ảnh PNG giả để kiểm tra
    dummy = np.full((20, 20, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(clean_dir / "photo.png"), dummy)
    cv2.imwrite(str(degraded_dir / "photo_gaussian_noise.png"), dummy)
    cv2.imwrite(str(degraded_dir / "photo_motion_blur.png"), dummy)
    # File không khớp prefix → không được ghép vào
    cv2.imwrite(str(degraded_dir / "other_noise.png"), dummy)

    pairs = load_synthetic_pairs(clean_dir=str(clean_dir), degraded_dir=str(degraded_dir))

    assert len(pairs) == 2  # photo_gaussian_noise và photo_motion_blur
    degraded_names = {p[1].name for p in pairs}
    assert "photo_gaussian_noise.png" in degraded_names
    assert "photo_motion_blur.png" in degraded_names
    assert "other_noise.png" not in degraded_names


def test_run_benchmark_suite_returns_dict(tmp_path):
    """run_benchmark_suite phải trả về dict với keys là tên file."""
    import cv2

    from src.analyzer_evaluator.synthetic_loader import run_benchmark_suite

    clean_dir = tmp_path / "clean"
    degraded_dir = tmp_path / "degraded"
    clean_dir.mkdir()
    degraded_dir.mkdir()

    dummy_clean = np.full((30, 30, 3), 128, dtype=np.uint8)
    dummy_degraded = np.full((30, 30, 3), 100, dtype=np.uint8)
    cv2.imwrite(str(clean_dir / "img.png"), dummy_clean)
    cv2.imwrite(str(degraded_dir / "img_noisy.png"), dummy_degraded)

    results = run_benchmark_suite(clean_dir=str(clean_dir), degraded_dir=str(degraded_dir))

    assert "img_noisy.png" in results
    assert "psnr" in results["img_noisy.png"]
    assert "ssim" in results["img_noisy.png"]
    assert "mse" in results["img_noisy.png"]
    assert results["img_noisy.png"]["psnr"] > 0.0
