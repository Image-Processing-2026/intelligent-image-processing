import os
import cv2
import numpy as np
import pytest

from src.analyzer_evaluator.analyzer import analyze_image
from src.processing_engine.denoise import apply_denoise
from src.processing_engine.exposure_contrast import apply_gamma, apply_clahe
from src.processing_engine.color import apply_color_balance
from src.processing_engine.sharpen import apply_sharpen

DATA_DIR = os.path.join("data", "real")

@pytest.fixture
def noisy_img():
    img = cv2.imread(os.path.join(DATA_DIR, "astro_noisy.jpg"))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros((100, 100, 3), dtype=np.uint8)

@pytest.fixture
def dark_img():
    img = cv2.imread(os.path.join(DATA_DIR, "coffee_underexposed.jpg"))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros((100, 100, 3), dtype=np.uint8)

@pytest.fixture
def warm_img():
    img = cv2.imread(os.path.join(DATA_DIR, "cat_warm_cast.jpg"))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros((100, 100, 3), dtype=np.uint8)

@pytest.mark.parametrize("method", ["gaussian", "median", "bilateral", "nlm"])
@pytest.mark.parametrize("strength", [0.5, 1.0, 1.5])
def test_denoise_real(noisy_img, method, strength):
    input_metrics = analyze_image(noisy_img)
    out = apply_denoise(noisy_img, method=method, strength=strength)
    out_metrics = analyze_image(out)
    assert out.shape == noisy_img.shape
    assert out.dtype == np.uint8
    assert out_metrics.noise_variance is not None

@pytest.mark.parametrize("gamma", [1.2, 1.5, 2.0, 2.5])
def test_gamma_real(dark_img, gamma):
    input_metrics = analyze_image(dark_img)
    out = apply_gamma(dark_img, gamma=gamma)
    out_metrics = analyze_image(out)
    assert out_metrics.brightness_mean > input_metrics.brightness_mean
    assert out_metrics.histogram_stats["highlight_clip_ratio"] < 0.25

@pytest.mark.parametrize("clip_limit", [1.5, 2.0, 3.0, 4.0])
def test_clahe_real(dark_img, clip_limit):
    input_metrics = analyze_image(dark_img)
    out = apply_clahe(dark_img, clip_limit=clip_limit)
    out_metrics = analyze_image(out)
    assert out_metrics.contrast_std > input_metrics.contrast_std

@pytest.mark.parametrize("temperature_shift", [-0.3, -0.5, -0.8])
def test_color_balance_real(warm_img, temperature_shift):
    input_metrics = analyze_image(warm_img)
    out = apply_color_balance(warm_img, temperature_shift=temperature_shift)
    out_metrics = analyze_image(out)
    diff_in = abs(input_metrics.histogram_stats["r_mean"] - input_metrics.histogram_stats["b_mean"])
    diff_out = abs(out_metrics.histogram_stats["r_mean"] - out_metrics.histogram_stats["b_mean"])
    assert diff_out < diff_in

@pytest.mark.parametrize("method", ["unsharp_mask", "laplacian"])
@pytest.mark.parametrize("amount", [0.5, 1.0, 1.5])
def test_sharpen_real(noisy_img, method, amount):
    blurred = cv2.GaussianBlur(noisy_img, (15, 15), 0)
    input_metrics = analyze_image(blurred)
    out = apply_sharpen(blurred, method=method, amount=amount)
    out_metrics = analyze_image(out)
    assert out_metrics.sharpness_laplacian_var > input_metrics.sharpness_laplacian_var
