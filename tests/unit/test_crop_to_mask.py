"""
Unit tests cho cắt theo mask (crop-to-mask) trong apply_region_op — Module 3.

Tiêu chí: kết quả khi cắt phải GIỐNG HỆT TỪNG BYTE với chạy thuật toán trên toàn ảnh rồi hòa trộn.
Kết quả tham chiếu được tính độc lập với logic cắt: blend_regions(ảnh, raw_op(toàn ảnh), mask).
"""

import cv2
import numpy as np
import pytest

from src.processing_engine import (
    apply_clahe,
    apply_color_balance,
    apply_denoise,
    apply_gamma,
    apply_region_op,
    apply_sharpen,
    apply_white_balance,
)
from src.processing_engine.color import _raw_color_balance
from src.processing_engine.denoise import _crop_halo, _raw_denoise
from src.processing_engine.exposure_contrast import _raw_gamma
from src.processing_engine.sharpen import _raw_sharpen
from src.region_engine.mask_utils import blend_regions

H, W = 96, 128

# (hàm public, hàm thuần túy, tham số) cho mọi thuật toán được phép cắt theo mask
CROPPED_OPS = [
    (apply_denoise, _raw_denoise, {"method": "gaussian", "strength": 0.4}),
    (apply_denoise, _raw_denoise, {"method": "gaussian", "strength": 2.0}),
    (apply_denoise, _raw_denoise, {"method": "median", "strength": 0.8}),
    (apply_denoise, _raw_denoise, {"method": "median", "strength": 2.0}),
    (apply_denoise, _raw_denoise, {"method": "nlm", "strength": 0.8}),
    (apply_denoise, _raw_denoise, {"method": "nlm", "strength": 2.0}),
    (apply_sharpen, _raw_sharpen, {"method": "unsharp_mask", "amount": 0.3}),
    (apply_sharpen, _raw_sharpen, {"method": "unsharp_mask", "amount": 1.5}),
    (apply_gamma, _raw_gamma, {"gamma": 2.2}),
    (apply_color_balance, _raw_color_balance, {"temperature_shift": 0.4}),
]
OP_IDS = [f"{fn.__name__}-{'-'.join(str(v) for v in kw.values())}" for fn, _, kw in CROPPED_OPS]


def _feathered(binary: np.ndarray) -> np.ndarray:
    """Mask mềm giống Module 2 (kernel Gauss cố định → bằng đúng 0 ngoài dải làm mềm)."""
    return cv2.GaussianBlur(binary.astype(np.float32), (15, 15), 5.0)


def _ellipse(center, axes) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, center, axes, 0, 0, 360, 1, -1)
    return _feathered(m)


MASKS = {
    "inner_blob": _ellipse((70, 50), (18, 12)),
    "touching_corner": _ellipse((4, 3), (20, 15)),
    "two_blobs": np.maximum(_ellipse((20, 20), (8, 8)), _ellipse((110, 80), (10, 7))),
    "thin_row": _feathered(np.pad(np.ones((1, 40), np.uint8), ((50, H - 51), (30, W - 70)))),
}


@pytest.fixture(params=["rgb", "gray2d", "gray3d"])
def image(request):
    rng = np.random.default_rng(2026)
    base = cv2.resize(
        rng.integers(0, 256, (12, 16, 3), dtype=np.uint8), (W, H), interpolation=cv2.INTER_CUBIC
    )
    img = np.clip(base + rng.normal(0, 15, base.shape), 0, 255).astype(np.uint8)
    if request.param == "gray2d":
        return img[..., 0].copy()
    if request.param == "gray3d":
        return img[..., :1].copy()
    return img


def _reference(image, mask, raw_op, kwargs):
    """Chạy thuật toán trên toàn ảnh rồi hòa trộn — hành vi trước khi có crop-to-mask."""
    work = image[..., 0] if image.ndim == 3 and image.shape[2] == 1 else image
    out = blend_regions(work, raw_op(work, **kwargs), mask)
    return out[..., None] if work is not image else out


@pytest.mark.parametrize("mask_name", list(MASKS))
@pytest.mark.parametrize("public_op, raw_op, kwargs", CROPPED_OPS, ids=OP_IDS)
def test_cropped_result_is_byte_identical_to_full_image(
    image, mask_name, public_op, raw_op, kwargs
):
    mask = MASKS[mask_name]
    out = public_op(image, mask=mask, **kwargs)
    expected = _reference(image, mask, raw_op, kwargs)
    assert out.shape == image.shape and out.dtype == np.uint8
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("public_op, raw_op, kwargs", CROPPED_OPS, ids=OP_IDS)
def test_mask_with_channel_axis_matches_2d_mask(image, public_op, raw_op, kwargs):
    mask = MASKS["inner_blob"]
    np.testing.assert_array_equal(
        public_op(image, mask=mask[..., None], **kwargs), public_op(image, mask=mask, **kwargs)
    )


def _spy(calls):
    def op(img, **_):
        calls.append(img.shape)
        return 255 - img

    return op


def test_crop_runs_op_only_on_mask_box_plus_halo():
    image = np.zeros((H, W, 3), np.uint8)
    mask = np.zeros((H, W), np.float32)
    mask[40:50, 60:80] = 1.0
    calls = []
    out = apply_region_op(image, mask, _spy(calls), crop_halo=5)
    assert calls == [(10 + 2 * 5, 20 + 2 * 5, 3)]
    assert (out[40:50, 60:80] == 255).all()
    assert out.sum() == 255 * 10 * 20 * 3


def test_crop_box_is_clipped_at_image_border():
    image = np.zeros((H, W), np.uint8)
    mask = np.zeros((H, W), np.float32)
    mask[0:4, W - 3 :] = 0.5
    calls = []
    apply_region_op(image, mask, _spy(calls), crop_halo=6)
    assert calls == [(4 + 6, 3 + 6)]


def test_empty_mask_skips_op_and_returns_copy():
    image = np.full((H, W, 3), 7, np.uint8)
    calls = []
    out = apply_region_op(image, np.zeros((H, W), np.float32), _spy(calls), crop_halo=3)
    assert calls == []
    np.testing.assert_array_equal(out, image)
    assert out is not image


def test_full_mask_and_no_mask_run_on_whole_image():
    image = np.zeros((H, W, 3), np.uint8)
    calls = []
    apply_region_op(image, np.ones((H, W), np.float32), _spy(calls), crop_halo=3)
    apply_region_op(image, None, _spy(calls), crop_halo=3)
    assert calls == [(H, W, 3), (H, W, 3)]


def test_input_is_not_mutated_by_cropped_path(image):
    before = image.copy()
    apply_denoise(image, mask=MASKS["inner_blob"], method="nlm", strength=1.0)
    np.testing.assert_array_equal(image, before)


# ---------------------------------------------------------------------------
# Thuật toán KHÔNG được cắt: vẫn chạy toàn ảnh
# ---------------------------------------------------------------------------


def test_bilateral_is_not_cropped():
    # Bilateral của OpenCV làm tròn phụ thuộc vị trí điểm ảnh (lệch 1 mức xám) → không cắt
    assert _crop_halo("bilateral", 1.0) is None


@pytest.mark.parametrize(
    "module_name, public_op, kwargs",
    [
        ("denoise", apply_denoise, {"method": "bilateral", "strength": 1.0}),
        ("sharpen", apply_sharpen, {"method": "laplacian", "amount": 0.5}),
        ("color", apply_color_balance, {"saturation_scale": 1.3}),
        ("color", apply_white_balance, {"method": "gray_world"}),
        ("exposure_contrast", apply_clahe, {"clip_limit": 2.0}),
    ],
)
def test_non_croppable_ops_run_on_whole_image(monkeypatch, module_name, public_op, kwargs):
    import importlib

    module = importlib.import_module(f"src.processing_engine.{module_name}")
    seen = []
    original = module.apply_region_op

    def recording(image, mask, op_func, *, crop_halo=None, **kw):
        seen.append(crop_halo)
        return original(image, mask, op_func, crop_halo=crop_halo, **kw)

    monkeypatch.setattr(module, "apply_region_op", recording)
    rng = np.random.default_rng(0)
    public_op(rng.integers(0, 256, (H, W, 3), dtype=np.uint8), mask=MASKS["inner_blob"], **kwargs)
    assert seen == [None]
