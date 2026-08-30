"""
Đánh giá chất lượng ảnh dựa trên ảnh chuẩn (Full-Reference Evaluation).
Chỉ áp dụng cho tập kiểm thử nhân tạo (Synthetic Benchmark) có ground-truth.
"""

from typing import Dict

import cv2
import numpy as np

try:
    from skimage.metrics import peak_signal_noise_ratio as compute_psnr
    from skimage.metrics import structural_similarity as compute_ssim

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


def evaluate_reference(
    current_image: np.ndarray, ground_truth_image: np.ndarray
) -> Dict[str, float]:
    """
    Tính toán các chỉ số so khớp điểm ảnh giữa ảnh hiện tại và ground-truth.

    Args:
        current_image: Ảnh đã qua xử lý (RGB, uint8).
        ground_truth_image: Ảnh gốc sạch chưa qua làm suy giảm (RGB, uint8).

    Returns:
        Dict chứa PSNR, SSIM, MSE.
    """
    # Đảm bảo kích thước hai ảnh khớp nhau
    if current_image.shape != ground_truth_image.shape:
        ground_truth_image = cv2.resize(
            ground_truth_image,
            (current_image.shape[1], current_image.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    # 1. Tính toán MSE (Mean Squared Error)
    mse_val = float(
        np.mean((ground_truth_image.astype(np.float64) - current_image.astype(np.float64)) ** 2)
    )

    # 2. Tính toán PSNR (Peak Signal-to-Noise Ratio)
    if mse_val == 0:
        psnr_val = 100.0
    elif SKIMAGE_AVAILABLE:
        psnr_val = float(compute_psnr(ground_truth_image, current_image, data_range=255))
    else:
        psnr_val = float(20 * np.log10(255.0 / np.sqrt(mse_val)))

    # 3. Tính toán SSIM (Structural Similarity Index)
    if SKIMAGE_AVAILABLE:
        channel_axis = 2 if len(current_image.shape) == 3 else None
        ssim_val = float(
            compute_ssim(
                ground_truth_image, current_image, data_range=255, channel_axis=channel_axis
            )
        )
    else:
        # Fallback SSIM đơn giản dựa trên covariance và variance nếu chưa cài scikit-image
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        img1 = ground_truth_image.astype(np.float64)
        img2 = current_image.astype(np.float64)
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1_sq = np.var(img1)
        sigma2_sq = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        ssim_val = float(
            ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2))
            / ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
        )

    return {"psnr": psnr_val, "ssim": ssim_val, "mse": mse_val}
