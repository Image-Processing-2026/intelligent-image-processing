"""
Đánh giá chất lượng ảnh không cần ảnh gốc (No-Reference Evaluation).
Áp dụng cho ảnh thực tế tải lên từ người dùng.
"""

from typing import Dict, Optional
import numpy as np
from .analyzer import analyze_image


def evaluate_no_reference(
    current_image: np.ndarray,
    previous_image: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Đánh giá chất lượng cảm nhận và đo lường độ cải thiện kỹ thuật qua từng vòng lặp.

    Args:
        current_image: Ảnh ở vòng lặp hiện tại (RGB, uint8).
        previous_image: Ảnh ở vòng lặp trước đó (hoặc ảnh đầu vào ban đầu).

    Returns:
        Dict chứa các chỉ số ước lượng chất lượng và độ chênh lệch delta.
    """
    curr_metrics = analyze_image(current_image)
    result = {
        "current_brightness": curr_metrics.brightness_mean,
        "current_contrast": curr_metrics.contrast_std,
        "current_sharpness": curr_metrics.sharpness_laplacian_var,
        "current_noise": curr_metrics.noise_variance,
    }

    # Nếu có ảnh của vòng lặp trước, tính toán biến thiên (delta metrics)
    if previous_image is not None:
        prev_metrics = analyze_image(previous_image)
        result["delta_contrast"] = float(curr_metrics.contrast_std - prev_metrics.contrast_std)
        result["delta_sharpness"] = float(curr_metrics.sharpness_laplacian_var - prev_metrics.sharpness_laplacian_var)
        result["delta_noise"] = float(curr_metrics.noise_variance - prev_metrics.noise_variance)

    # Thử tính BRISQUE/NIQE nếu thư viện hỗ trợ, nếu không trả về fallback
    try:
        # Placeholder cho pyiqa / OpenCV BRISQUE score
        # Thang điểm BRISQUE: càng thấp chất lượng càng tốt [0, 100]
        result["estimated_quality_score"] = float(max(0.0, 100.0 - curr_metrics.noise_variance * 2.0))
    except Exception:
        result["estimated_quality_score"] = 50.0

    return result
