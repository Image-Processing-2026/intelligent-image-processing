"""
Unit tests for Module 1: Analyzer & Evaluator.
"""

import numpy as np
from src.analyzer_evaluator.analyzer import analyze_image
from src.analyzer_evaluator.no_reference_eval import evaluate_no_reference
from src.analyzer_evaluator.reference_eval import evaluate_reference


def test_analyze_image_metrics():
    """Kiểm tra việc trích xuất chỉ số kỹ thuật ảnh."""
    # Tạo ảnh mẫu RGB xám đồng nhất
    dummy_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    metrics = analyze_image(dummy_img)

    assert metrics.brightness_mean == 128.0
    assert metrics.brightness_level == "normal"
    assert metrics.contrast_std == 0.0
    assert metrics.contrast_level == "low"
    assert metrics.noise_level == "clean"


def test_reference_evaluation():
    """Kiểm tra tính toán PSNR/SSIM trên 2 ảnh giống hệt nhau."""
    img1 = np.ones((100, 100, 3), dtype=np.uint8) * 100
    img2 = np.ones((100, 100, 3), dtype=np.uint8) * 100

    eval_res = evaluate_reference(img1, img2)
    assert eval_res["psnr"] > 90.0  # Gần như vô hạn khi 2 ảnh giống nhau
    assert eval_res["ssim"] == 1.0
    assert eval_res["mse"] == 0.0


def test_no_reference_evaluation():
    """Kiểm tra tính toán đánh giá không cần ảnh gốc."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 150
    res = evaluate_no_reference(img)
    assert "current_brightness" in res
    assert "current_contrast" in res
