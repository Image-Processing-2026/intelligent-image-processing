"""
Kiểm tra Module 3 hoạt động đúng với ảnh grayscale (1 kênh, shape (H, W)),
không chỉ ảnh RGB (H, W, 3) như test ban đầu.
"""
import numpy as np

from src.processing_engine.color import apply_color_balance
from src.processing_engine.denoise import apply_denoise
from src.processing_engine.exposure_contrast import apply_clahe, apply_gamma
from src.processing_engine.sharpen import apply_sharpen


def _make_gray_image():
    rng = np.random.default_rng(0)
    return (rng.random((80, 80)) * 255).astype(np.uint8)


def test_denoise_grayscale_all_methods():
    gray = _make_gray_image()
    for method in ["gaussian", "median", "bilateral", "nlm"]:
        result = apply_denoise(gray, method=method, strength=1.0)
        assert result.shape == gray.shape
        assert result.dtype == np.uint8


def test_gamma_grayscale():
    gray = _make_gray_image()
    result = apply_gamma(gray, gamma=1.5)
    assert result.shape == gray.shape
    assert result.dtype == np.uint8


def test_clahe_grayscale():
    gray = _make_gray_image()
    result = apply_clahe(gray)
    assert result.shape == gray.shape
    assert result.dtype == np.uint8


def test_sharpen_grayscale_both_methods():
    gray = _make_gray_image()
    for method in ["unsharp_mask", "laplacian"]:
        result = apply_sharpen(gray, method=method)
        assert result.shape == gray.shape
        assert result.dtype == np.uint8


def test_color_balance_grayscale_is_noop():
    """Ảnh grayscale không có kênh màu -> color balance phải trả về nguyên trạng, không lỗi."""
    gray = _make_gray_image()
    result = apply_color_balance(gray, saturation_scale=1.8, temperature_shift=0.5)
    assert result.shape == gray.shape
    np.testing.assert_array_equal(result, gray)


def test_grayscale_with_soft_mask():
    """Đảm bảo blend_regions cũng hoạt động đúng khi ảnh gốc là grayscale."""
    gray = _make_gray_image()
    mask = np.ones_like(gray, dtype=np.float32) * 0.6
    result = apply_gamma(gray, mask=mask, gamma=2.0)
    assert result.shape == gray.shape
    assert result.dtype == np.uint8
