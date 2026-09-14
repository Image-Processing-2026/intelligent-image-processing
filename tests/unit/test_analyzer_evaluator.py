"""
Unit tests cho Module 1: Analyzer & Evaluator.
Kiểm tra các chỉ số kỹ thuật cơ bản, hợp đồng schemas và fixtures ảnh nhân tạo.
"""

import numpy as np

from src.analyzer_evaluator import (
    EvaluationResult,
    TechnicalMetrics,
    analyze_image,
    evaluate_no_reference,
    evaluate_reference,
)


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


def test_analyze_image_with_flat_fixture(flat_gray_image: np.ndarray):
    """Kiểm tra trích xuất chỉ số kỹ thuật trên ảnh phẳng xám đồng nhất từ fixture."""
    metrics = analyze_image(flat_gray_image)

    assert metrics.brightness_mean == 128.0
    assert metrics.brightness_level == "normal"
    assert metrics.contrast_std == 0.0
    assert metrics.contrast_level == "low"
    assert metrics.noise_level == "clean"
    assert metrics.blur_level == "severe_blur"


def test_analyze_image_exposure_fixtures(black_image: np.ndarray, white_image: np.ndarray):
    """Kiểm tra phân loại phơi sáng với ảnh đen và ảnh trắng cực đoan."""
    metrics_black = analyze_image(black_image)
    assert metrics_black.brightness_mean == 0.0
    assert metrics_black.brightness_level == "underexposed"

    metrics_white = analyze_image(white_image)
    assert metrics_white.brightness_mean == 255.0
    assert metrics_white.brightness_level == "overexposed"


def test_analyze_image_noisy_fixture(noisy_image: np.ndarray):
    """Kiểm tra việc phát hiện nhiễu trên ảnh có nhiễu Gaussian."""
    metrics = analyze_image(noisy_image)
    assert metrics.noise_variance > 5.0
    assert metrics.noise_level in ["low", "medium", "severe"]


def test_reference_evaluation():
    """Kiểm tra tính toán PSNR/SSIM trên 2 ảnh giống hệt nhau."""
    img1 = np.ones((100, 100, 3), dtype=np.uint8) * 100
    img2 = np.ones((100, 100, 3), dtype=np.uint8) * 100

    eval_res = evaluate_reference(img1, img2)
    assert eval_res["psnr"] >= 99.0
    assert eval_res["ssim"] == 1.0
    assert eval_res["mse"] == 0.0


def test_no_reference_evaluation(flat_gray_image: np.ndarray):
    """Kiểm tra tính toán đánh giá không cần ảnh gốc."""
    res = evaluate_no_reference(flat_gray_image)
    assert "current_brightness" in res
    assert "current_contrast" in res
    assert res["current_brightness"] == 128.0
