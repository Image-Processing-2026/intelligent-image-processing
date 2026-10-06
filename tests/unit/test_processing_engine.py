"""
Unit tests for Module 3: Image Processing Engine.
"""

import numpy as np
import pytest

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


@pytest.mark.parametrize("method", ["unsharp_mask", "laplacian"])
@pytest.mark.parametrize("amount", [0.2, 0.5, 1.0, 2.0])
def test_sharpen_preserves_flat_brightness(method, amount):
    """Làm nét không được đổi độ sáng vùng phẳng (kernel laplacian cũ nhân độ sáng với amount)."""
    img = np.full((32, 32, 3), 120, dtype=np.uint8)
    out = apply_sharpen(img, method=method, amount=amount)
    assert np.array_equal(out, img)


def test_laplacian_sharpen_boosts_edges_more_with_amount():
    """amount lớn hơn → chênh lệch tại cạnh lớn hơn, độ sáng trung bình gần như giữ nguyên."""
    img = np.full((32, 32), 100, dtype=np.uint8)
    img[:, 16:] = 150
    weak = apply_sharpen(img, method="laplacian", amount=0.3)
    strong = apply_sharpen(img, method="laplacian", amount=1.0)
    edge_weak = int(weak[:, 16].mean()) - int(weak[:, 15].mean())
    edge_strong = int(strong[:, 16].mean()) - int(strong[:, 15].mean())
    assert edge_strong > edge_weak > 50
    assert abs(float(strong.mean()) - float(img.mean())) < 2.0
