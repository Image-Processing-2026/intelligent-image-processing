"""
Các thuật toán khử nhiễu kinh điển (Classical Image Denoising Algorithms).
Bao gồm: Lọc Gauss, Lọc Trung vị (Median), Lọc Song phương (Bilateral), Non-Local Means (NLM).

Quy ước `strength` (thống nhất cho mọi method):
    σ_target = NOISE_SIGMA_PER_STRENGTH · strength = 12.5 · strength
là độ lệch chuẩn nhiễu Gauss mà bộ lọc nhắm tới, theo thang cường độ 0–255.
Planner kẹp strength ∈ [0.1, 2.0] ↔ σ_target ∈ [1.25, 25].

Mỗi method suy ra tham số từ σ_target theo hệ số đã hiệu chỉnh bằng benchmark
(scripts/benchmark_module3.py): 4 ảnh scikit-image, nhiễu Gauss σ ∈ {5, 10, 15, 20, 25}, seed 2026.
Quy tắc chọn: PSNR trung bình cao nhất; nếu ứng viên khác kém ≤ 0.15 dB nhưng SSIM cao hơn thì chọn nó.
"""

import math
from typing import Literal, Optional

import cv2
import numpy as np

from .base import apply_region_op, require_choice, require_number

DENOISE_METHODS = ("gaussian", "median", "bilateral", "nlm")

# σ nhiễu mục tiêu (thang 0–255) ứng với strength = 1.0
NOISE_SIGMA_PER_STRENGTH = 12.5

# Gaussian: σ_blur (pixel) = 0.06 · σ_target. Kernel tự tính từ σ_blur (≈ ±3σ_blur) để không bị cắt cụt.
GAUSSIAN_BLUR_SIGMA_PER_NOISE_SIGMA = 0.06
# Median: kích thước cửa sổ tăng 2 sau mỗi 17.5 đơn vị σ_target: 3×3 (σ < 17.5), 5×5 (< 35), 7×7 ...
MEDIAN_SIGMA_PER_KERNEL_STEP = 17.5
# Bilateral: σ_color = σ_space = 4 · σ_target; đường kính lân cận d = max(5, ⌊0.4 · σ_target⌋) pixel.
# Vì σ_space ≫ d/2 nên trọng số không gian gần như đều trong cửa sổ d×d; tính chất giữ cạnh
# đến từ trọng số cường độ (chênh lệch màu lớn hơn ~σ_color bị giảm trọng số).
BILATERAL_SIGMA_PER_NOISE_SIGMA = 4.0
BILATERAL_DIAMETER_PER_NOISE_SIGMA = 0.4
# NLM: tham số lọc h = 0.8 · σ_target (OpenCV khuyến nghị h ≈ σ; patch 7×7, vùng tìm 21×21).
NLM_H_PER_NOISE_SIGMA = 0.8


# NLM: kích thước patch và vùng tìm kiếm (pixel, số lẻ)
NLM_TEMPLATE_WINDOW = 7
NLM_SEARCH_WINDOW = 21


def noise_sigma_for_strength(strength: float) -> float:
    """σ nhiễu mục tiêu (thang 0–255) ứng với strength."""
    return NOISE_SIGMA_PER_STRENGTH * strength


def _gaussian_blur_sigma(sigma: float) -> float:
    return GAUSSIAN_BLUR_SIGMA_PER_NOISE_SIGMA * sigma


def _median_ksize(sigma: float) -> int:
    return 2 * math.floor(sigma / MEDIAN_SIGMA_PER_KERNEL_STEP) + 3


def _crop_halo(method: str, strength: float) -> Optional[int]:
    """
    Bán kính ảnh hưởng (pixel) để cắt theo mask, None nếu method không được cắt.

    - gaussian: OpenCV dùng kernel bán kính ≈ 3σ_blur với ảnh uint8; lấy 4σ_blur + 1 cho dư.
    - median: nửa cửa sổ k×k.
    - nlm: điểm ảnh p lấy trung bình các patch tâm q trong vùng tìm kiếm, mỗi patch phủ thêm
      nửa patch → bán kính = nửa vùng tìm + nửa patch = 10 + 3.
    - bilateral: None — đường SIMD số thực của OpenCV làm tròn khác nhau theo vị trí điểm ảnh
      (lệch 1 mức xám với ảnh xám), nên không đảm bảo giống hệt kết quả toàn ảnh.
    """
    sigma = noise_sigma_for_strength(strength)
    if method == "gaussian":
        return math.ceil(4 * _gaussian_blur_sigma(sigma)) + 1
    if method == "median":
        return _median_ksize(sigma) // 2
    if method == "nlm":
        return NLM_SEARCH_WINDOW // 2 + NLM_TEMPLATE_WINDOW // 2
    return None


def _raw_denoise(
    image: np.ndarray,
    method: Literal["gaussian", "median", "bilateral", "nlm"] = "bilateral",
    strength: float = 1.0,
) -> np.ndarray:
    """Thuật toán khử nhiễu thuần túy trên toàn ảnh, tham số suy ra từ σ_target."""
    sigma = noise_sigma_for_strength(strength)

    if method == "gaussian":
        # Tích chập với nhân Gauss G_σ: làm mờ đều, giảm nhiễu nhưng cũng mất chi tiết cạnh
        return cv2.GaussianBlur(image, (0, 0), sigmaX=_gaussian_blur_sigma(sigma))

    if method == "median":
        # Trung vị cửa sổ k×k: tốt nhất với nhiễu xung (muối tiêu); với nhiễu Gauss chỉ ở mức vừa
        return cv2.medianBlur(image, _median_ksize(sigma))

    if method == "bilateral":
        # Bộ lọc song phương: trọng số = khoảng cách không gian × chênh lệch cường độ → giữ cạnh
        d = max(5, int(BILATERAL_DIAMETER_PER_NOISE_SIGMA * sigma))
        sigma_range = BILATERAL_SIGMA_PER_NOISE_SIGMA * sigma
        return cv2.bilateralFilter(image, d, sigma_range, sigma_range)

    # method == "nlm" (đã được kiểm tra ở apply_denoise)
    # Non-Local Means: trung bình các patch tương tự trên vùng tìm kiếm rộng
    h = NLM_H_PER_NOISE_SIGMA * sigma
    if image.ndim == 2:
        return cv2.fastNlMeansDenoising(image, None, h, NLM_TEMPLATE_WINDOW, NLM_SEARCH_WINDOW)
    return cv2.fastNlMeansDenoisingColored(
        image, None, h, h, NLM_TEMPLATE_WINDOW, NLM_SEARCH_WINDOW
    )


def apply_denoise(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    method: Literal["gaussian", "median", "bilateral", "nlm"] = "bilateral",
    strength: float = 1.0,
) -> np.ndarray:
    """
    Khử nhiễu cục bộ theo mặt nạ mềm hoặc toàn cục.

    Args:
        method: "gaussian" | "median" | "bilateral" | "nlm".
        strength: Không thứ nguyên; σ_target = 12.5 · strength (σ nhiễu Gauss, thang 0–255).
            VD: strength 0.8 ↔ σ 10, strength 2.0 ↔ σ 25.

    Raises:
        ProcessingInputError: method không thuộc DENOISE_METHODS, strength không phải số > 0,
            hoặc ảnh/mặt nạ vi phạm hợp đồng.
    """
    require_choice(method, DENOISE_METHODS, "method", "apply_denoise")
    strength = require_number(strength, "strength", "apply_denoise", gt=0.0)
    return apply_region_op(
        image,
        mask,
        _raw_denoise,
        crop_halo=_crop_halo(method, strength),
        method=method,
        strength=strength,
    )
