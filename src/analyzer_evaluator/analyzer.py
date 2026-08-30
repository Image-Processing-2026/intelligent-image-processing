"""
Trích xuất chỉ số kỹ thuật ảnh (Technical Metrics Analyzer).
Tính toán độ sáng, độ tương phản, mức độ nhiễu, độ mờ/sắc nét và phân bố histogram.
"""

from typing import Any, Dict
import cv2
import numpy as np
from pydantic import BaseModel, Field


class TechnicalMetrics(BaseModel):
    """Mô hình dữ liệu chứa các chỉ số kỹ thuật của ảnh."""
    brightness_mean: float = Field(..., description="Độ sáng trung bình [0, 255]")
    brightness_level: str = Field(..., description="Mức độ: underexposed, normal, overexposed")
    contrast_std: float = Field(..., description="Độ lệch chuẩn thể hiện độ tương phản")
    contrast_level: str = Field(..., description="Mức độ: low, normal, high")
    noise_variance: float = Field(..., description="Ước lượng phương sai nhiễu")
    noise_level: str = Field(..., description="Mức độ nhiễu: clean, low, medium, severe")
    sharpness_laplacian_var: float = Field(..., description="Phương sai toán tử Laplacian")
    blur_level: str = Field(..., description="Mức độ mờ: sharp, mild_blur, severe_blur")
    color_cast: str = Field(default="none", description="Ám màu: warm, cool, greenish, none")
    histogram_stats: Dict[str, Any] = Field(default_factory=dict)


def analyze_image(image: np.ndarray) -> TechnicalMetrics:
    """
    Phân tích định lượng các thuộc tính kỹ thuật của bức ảnh.

    Args:
        image: Ảnh đầu vào định dạng RGB, np.uint8, shape (H, W, 3) hoặc grayscale (H, W).

    Returns:
        TechnicalMetrics: Cấu trúc dữ liệu chứa toàn bộ chỉ số đã trích xuất.
    """
    # Chuyển đổi sang ảnh xám để tính toán các chỉ số cường độ sáng
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # 1. Tính toán độ sáng (Brightness)
    brightness_mean = float(np.mean(gray))
    if brightness_mean < 70.0:
        brightness_level = "underexposed"
    elif brightness_mean > 185.0:
        brightness_level = "overexposed"
    else:
        brightness_level = "normal"

    # 2. Tính toán độ tương phản (Contrast) dựa trên độ lệch chuẩn
    contrast_std = float(np.std(gray))
    if contrast_std < 40.0:
        contrast_level = "low"
    elif contrast_std > 80.0:
        contrast_level = "high"
    else:
        contrast_level = "normal"

    # 3. Tính toán độ sắc nét / mờ dựa trên phương sai toán tử Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness_laplacian_var = float(laplacian.var())
    if sharpness_laplacian_var < 100.0:
        blur_level = "severe_blur"
    elif sharpness_laplacian_var < 300.0:
        blur_level = "mild_blur"
    else:
        blur_level = "sharp"

    # 4. Ước lượng mức độ nhiễu (Noise Variance) sử dụng bộ lọc làm mịn
    # Sử dụng chênh lệch giữa ảnh gốc và ảnh sau khi lọc trung vị
    median_filtered = cv2.medianBlur(gray, 3)
    noise_diff = cv2.absdiff(gray, median_filtered)
    noise_variance = float(np.mean(noise_diff))
    if noise_variance < 3.0:
        noise_level = "clean"
    elif noise_variance < 8.0:
        noise_level = "low"
    elif noise_variance < 15.0:
        noise_level = "medium"
    else:
        noise_level = "severe"

    # 5. Phân tích màu sắc cơ bản
    color_cast = "none"
    if len(image.shape) == 3:
        r_mean = float(np.mean(image[:, :, 0]))
        g_mean = float(np.mean(image[:, :, 1]))
        b_mean = float(np.mean(image[:, :, 2]))
        if r_mean > b_mean + 20 and r_mean > g_mean + 15:
            color_cast = "warm"
        elif b_mean > r_mean + 20 and b_mean > g_mean + 15:
            color_cast = "cool"

    return TechnicalMetrics(
        brightness_mean=brightness_mean,
        brightness_level=brightness_level,
        contrast_std=contrast_std,
        contrast_level=contrast_level,
        noise_variance=noise_variance,
        noise_level=noise_level,
        sharpness_laplacian_var=sharpness_laplacian_var,
        blur_level=blur_level,
        color_cast=color_cast,
        histogram_stats={"gray_min": int(np.min(gray)), "gray_max": int(np.max(gray))}
    )
