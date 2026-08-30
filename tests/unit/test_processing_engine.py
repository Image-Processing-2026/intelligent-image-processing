"""
Unit tests for Module 3: Image Processing Engine.
"""

import numpy as np

from src.processing_engine.denoise import apply_denoise
from src.processing_engine.exposure_contrast import apply_clahe, apply_gamma
from src.processing_engine.sharpen import apply_sharpen


def test_gamma_correction():
    """Kiểm tra biến đổi gamma."""
    img = np.ones((50, 50, 3), dtype=np.uint8) * 100
    brightened = apply_gamma(img, mask=None, gamma=1.5)
    assert brightened.mean() > img.mean()

    darkened = apply_gamma(img, mask=None, gamma=0.7)
    assert darkened.mean() < img.mean()


def test_denoise_preserves_shape():
    """Kiểm tra các thuật toán khử nhiễu không làm thay đổi kích thước ảnh."""
    noisy_img = (np.random.rand(60, 60, 3) * 255).astype(np.uint8)
    denoised_gauss = apply_denoise(noisy_img, method="gaussian")
    denoised_bilateral = apply_denoise(noisy_img, method="bilateral")

    assert denoised_gauss.shape == noisy_img.shape
    assert denoised_bilateral.shape == noisy_img.shape


def test_clahe_contrast_enhancement():
    """Kiểm tra CLAHE thực thi đúng trên kênh màu."""
    img = np.ones((64, 64, 3), dtype=np.uint8) * 128
    res = apply_clahe(img, clip_limit=2.0)
    assert res.shape == img.shape


def test_sharpen():
    """Kiểm tra hàm làm nét."""
    img = np.ones((40, 40, 3), dtype=np.uint8) * 100
    sharp = apply_sharpen(img, method="unsharp_mask", amount=1.2)
    assert sharp.shape == img.shape
