"""
Unit tests for Module 3: Image Processing Engine.
"""

import cv2
import numpy as np
import pytest

from src.processing_engine import ProcessingInputError
from src.processing_engine import base as engine_base
from src.processing_engine.color import apply_color_balance
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
    noisy_img = (np.random.default_rng(0).random((60, 60, 3)) * 255).astype(np.uint8)
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


# ---------------------------------------------------------------------------
# Hợp đồng dữ liệu & bất biến hòa trộn (contract & blending invariants)
# ---------------------------------------------------------------------------

# (tên, hàm gọi với tham số "có tác dụng" để đầu ra khác đầu vào)
OPERATIONS = [
    ("denoise_gaussian", lambda img, m: apply_denoise(img, mask=m, method="gaussian")),
    ("denoise_median", lambda img, m: apply_denoise(img, mask=m, method="median")),
    ("denoise_bilateral", lambda img, m: apply_denoise(img, mask=m, method="bilateral")),
    ("denoise_nlm", lambda img, m: apply_denoise(img, mask=m, method="nlm")),
    ("gamma", lambda img, m: apply_gamma(img, mask=m, gamma=1.6)),
    ("clahe", lambda img, m: apply_clahe(img, mask=m, clip_limit=3.0)),
    ("sharpen_unsharp", lambda img, m: apply_sharpen(img, mask=m, method="unsharp_mask")),
    ("sharpen_laplacian", lambda img, m: apply_sharpen(img, mask=m, method="laplacian")),
    (
        "color",
        lambda img, m: apply_color_balance(
            img, mask=m, saturation_scale=1.3, temperature_shift=0.5
        ),
    ),
]
OP_IDS = [name for name, _ in OPERATIONS]
OP_FUNCS = [func for _, func in OPERATIONS]


@pytest.fixture
def textured_rgb():
    """Ảnh RGB có cấu trúc + nhiễu cố định (seed) để mọi thao tác đều tạo khác biệt."""
    rng = np.random.default_rng(42)
    y, x = np.mgrid[0:48, 0:64]
    base = np.stack([x * 3, y * 4, (x + y) * 2], axis=-1).astype(np.float32) + 30
    noisy = base + rng.normal(0, 12, base.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("op", OP_FUNCS, ids=OP_IDS)
def test_zero_mask_leaves_image_untouched(op, textured_rgb):
    """Mặt nạ toàn 0 nghĩa là không vùng nào được xử lý → ảnh giữ nguyên từng byte."""
    mask = np.zeros(textured_rgb.shape[:2], dtype=np.float32)
    np.testing.assert_array_equal(op(textured_rgb, mask), textured_rgb)


@pytest.mark.parametrize("op", OP_FUNCS, ids=OP_IDS)
def test_full_mask_equals_global_processing(op, textured_rgb):
    """Mặt nạ toàn 1 phải cho kết quả giống hệt xử lý toàn cục (mask=None)."""
    mask = np.ones(textured_rgb.shape[:2], dtype=np.float32)
    np.testing.assert_array_equal(op(textured_rgb, mask), op(textured_rgb, None))


@pytest.mark.parametrize("op", OP_FUNCS, ids=OP_IDS)
def test_half_mask_only_changes_masked_side(op, textured_rgb):
    """Mặt nạ cứng nửa trái: nửa phải giữ nguyên, nửa trái bằng kết quả xử lý toàn ảnh."""
    mask = np.zeros(textured_rgb.shape[:2], dtype=np.float32)
    mask[:, :32] = 1.0
    out = op(textured_rgb, mask)
    full = op(textured_rgb, None)
    np.testing.assert_array_equal(out[:, 32:], textured_rgb[:, 32:])
    np.testing.assert_array_equal(out[:, :32], full[:, :32])


@pytest.mark.parametrize("op", OP_FUNCS, ids=OP_IDS)
def test_output_contract_and_input_not_mutated(op, textured_rgb):
    """Đầu ra uint8 cùng shape; ảnh đầu vào không bị sửa tại chỗ."""
    before = textured_rgb.copy()
    out = op(textured_rgb, np.full(textured_rgb.shape[:2], 0.5, dtype=np.float32))
    assert out.shape == textured_rgb.shape
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(textured_rgb, before)


@pytest.mark.parametrize("op", OP_FUNCS, ids=OP_IDS)
def test_single_channel_axis_is_preserved(op, textured_rgb):
    """Ảnh xám (H, W, 1) trả về đúng (H, W, 1) và khớp kết quả xử lý trên (H, W)."""
    gray = np.ascontiguousarray(textured_rgb[..., 1])
    out_3d = op(gray[..., None], None)
    assert out_3d.shape == (*gray.shape, 1)
    np.testing.assert_array_equal(out_3d[..., 0], op(gray, None))


@pytest.mark.parametrize(
    "bad_image",
    [
        np.zeros((8, 8, 4), dtype=np.uint8),  # RGBA
        np.zeros((8, 8, 3), dtype=np.float32),  # float [0, 1]
        np.zeros((8, 8, 3), dtype=np.uint16),
        np.zeros((0, 8, 3), dtype=np.uint8),
        np.zeros((2, 8, 8, 3), dtype=np.uint8),
        [[0, 1], [2, 3]],
    ],
    ids=["rgba", "float32", "uint16", "empty", "4d", "list"],
)
@pytest.mark.parametrize("op", OP_FUNCS, ids=OP_IDS)
def test_invalid_images_raise_clear_error(op, bad_image):
    with pytest.raises(ProcessingInputError):
        op(bad_image, None)


@pytest.mark.parametrize(
    "bad_mask",
    [
        np.ones((10, 10), dtype=np.float32),  # sai kích thước
        np.ones((48, 64), dtype=np.float64),  # sai dtype
        np.ones((48, 64), dtype=np.uint8) * 255,  # mặt nạ nhị phân 0/255
        np.full((48, 64), 1.5, dtype=np.float32),  # vượt [0, 1]
        np.full((48, 64), np.nan, dtype=np.float32),
        np.ones((48, 64, 3), dtype=np.float32),
    ],
    ids=["shape", "float64", "uint8", "range", "nan", "3ch"],
)
def test_invalid_masks_raise_before_running_filter(bad_mask, textured_rgb):
    """Mặt nạ sai bị chặn trước khi chạy bộ lọc (fail-fast, không tốn thời gian chạy NLM)."""
    calls = []

    def spy_op(img):
        calls.append("op")
        return img

    with pytest.raises(ProcessingInputError):
        engine_base.apply_region_op(textured_rgb, bad_mask, spy_op)
    assert calls == []


@pytest.mark.parametrize(
    "call",
    [
        lambda img: apply_denoise(img, method="wavelet"),
        lambda img: apply_denoise(img, strength=0.0),
        lambda img: apply_denoise(img, strength=float("nan")),
        lambda img: apply_denoise(img, strength="1.0"),
        lambda img: apply_sharpen(img, method="highpass"),
        lambda img: apply_sharpen(img, amount=-0.5),
        lambda img: apply_gamma(img, gamma=0.0),
        lambda img: apply_gamma(img, gamma=-1.0),
        lambda img: apply_clahe(img, clip_limit=0.0),
        lambda img: apply_clahe(img, tile_grid_size=(8, 0)),
        lambda img: apply_clahe(img, tile_grid_size=(8,)),
        lambda img: apply_clahe(img, tile_grid_size=(8.0, 8.0)),
        lambda img: apply_color_balance(img, saturation_scale=-0.1),
        lambda img: apply_color_balance(img, temperature_shift=float("inf")),
        lambda img: apply_color_balance(img, temperature_shift=True),
    ],
    ids=[
        "denoise-method",
        "denoise-zero-strength",
        "denoise-nan",
        "denoise-str",
        "sharpen-method",
        "sharpen-negative",
        "gamma-zero",
        "gamma-negative",
        "clahe-clip-zero",
        "clahe-tile-zero",
        "clahe-tile-len",
        "clahe-tile-float",
        "color-saturation",
        "color-inf",
        "color-bool",
    ],
)
def test_invalid_parameters_raise_instead_of_silent_noop(call, textured_rgb):
    """Tham số sai phải báo lỗi; trước đây method lạ âm thầm trả về ảnh gốc."""
    with pytest.raises(ProcessingInputError):
        call(textured_rgb)


def test_processing_input_error_is_value_error():
    """Giữ tương thích: code gọi bắt ValueError vẫn bắt được lỗi của Module 3."""
    assert issubclass(ProcessingInputError, ValueError)


def test_clahe_accepts_tile_grid_from_json_list(textured_rgb):
    """Tham số từ JSON là list → được chấp nhận như tuple."""
    np.testing.assert_array_equal(
        apply_clahe(textured_rgb, tile_grid_size=[4, 4]),
        apply_clahe(textured_rgb, tile_grid_size=(4, 4)),
    )


def test_clahe_only_touches_luminance(textured_rgb):
    """CLAHE xử lý kênh L trong LAB: kênh a/b chỉ lệch do làm tròn khi đổi không gian màu."""
    out = apply_clahe(textured_rgb, clip_limit=4.0)
    lab_in = cv2.cvtColor(textured_rgb, cv2.COLOR_RGB2LAB).astype(np.int16)
    lab_out = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.int16)
    assert np.abs(lab_out[..., 0] - lab_in[..., 0]).mean() > 3.0  # L thực sự thay đổi
    # Sai số chuyển đổi RGB↔LAB uint8 + cắt gam màu: trung bình nhỏ, không méo màu hệ thống
    assert np.abs(lab_out[..., 1:] - lab_in[..., 1:]).mean() < 1.5


def test_gamma_matches_closed_form_formula():
    """Kiểm tra công thức I_out = 255 · (I_in / 255)^(1/γ) (sai số ≤ 1 mức xám do lượng tử)."""
    levels = np.arange(256, dtype=np.uint8).reshape(16, 16)
    out = apply_gamma(levels, gamma=2.2).astype(np.float64)
    expected = 255.0 * (levels / 255.0) ** (1 / 2.2)
    assert np.abs(out - expected).max() <= 1.0


def test_gamma_one_is_identity(textured_rgb):
    np.testing.assert_array_equal(apply_gamma(textured_rgb, gamma=1.0), textured_rgb)


@pytest.mark.parametrize("method", ["gaussian", "median", "bilateral", "nlm"])
def test_denoise_reduces_noise_on_flat_patch(method):
    """Trên mảng phẳng có nhiễu Gauss (seed cố định), độ lệch chuẩn phải giảm rõ rệt."""
    rng = np.random.default_rng(7)
    noisy = np.clip(128 + rng.normal(0, 15, (64, 64, 3)), 0, 255).astype(np.uint8)
    out = apply_denoise(noisy, method=method, strength=1.5)
    assert out.std() < 0.7 * noisy.std()


def test_color_balance_is_noop_on_neutral_params(textured_rgb):
    np.testing.assert_array_equal(apply_color_balance(textured_rgb), textured_rgb)
