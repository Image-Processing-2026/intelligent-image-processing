"""
Unit tests cho cân bằng trắng (Gray World / White Patch) và tông ấm/lạnh bằng gain kênh — Module 3.
"""

import numpy as np
import pytest

from src.processing_engine import ProcessingInputError, apply_color_balance, apply_white_balance
from src.processing_engine.color import TEMPERATURE_GAIN_PER_UNIT

# Ám màu nhân (von Kries): nguồn sáng ấm làm kênh R mạnh lên, B yếu đi
WARM_CAST = np.array([1.25, 1.0, 0.75])


@pytest.fixture
def neutral_scene():
    """Cảnh trung tính: các mảng xám nhiều mức + một mảng trắng 230, kèm nhiễu nhỏ (seed cố định)."""
    rng = np.random.default_rng(3)
    levels = rng.integers(40, 200, size=(6, 8))
    gray = np.kron(levels, np.ones((8, 8))).astype(np.float64)
    gray[:8, :8] = 230.0  # vật trắng cho White Patch
    rgb = np.repeat(gray[..., None], 3, axis=-1) + rng.normal(0, 1.0, (48, 64, 3))
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _cast(image: np.ndarray, gains: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image.astype(np.float64) * gains), 0, 255).astype(np.uint8)


def _channel_means(image: np.ndarray) -> np.ndarray:
    return image.reshape(-1, 3).astype(np.float64).mean(axis=0)


# ---------------------------------------------------------------------------
# Gray World
# ---------------------------------------------------------------------------


def test_gray_world_equalizes_channel_means(neutral_scene):
    out = apply_white_balance(_cast(neutral_scene, WARM_CAST), method="gray_world")
    means = _channel_means(out)
    assert means.max() - means.min() < 1.5  # sai số làm tròn uint8


def test_gray_world_recovers_neutral_scene(neutral_scene):
    """Ám màu nhân thuần túy → Gray World khôi phục gần đúng ảnh trung tính (sai số làm tròn)."""
    casted = _cast(neutral_scene, WARM_CAST)
    out = apply_white_balance(casted, method="gray_world")
    err_before = np.abs(casted.astype(int) - neutral_scene.astype(int)).mean()
    err_after = np.abs(out.astype(int) - neutral_scene.astype(int)).mean()
    assert err_after < 2.0
    assert err_after < err_before / 5


def test_gray_world_preserves_overall_brightness(neutral_scene):
    casted = _cast(neutral_scene, WARM_CAST)
    out = apply_white_balance(casted, method="gray_world")
    assert abs(out.mean() - casted.mean()) < 1.0


def test_neutral_image_is_unchanged(neutral_scene):
    """Ảnh vốn trung tính có gain ≈ 1 → lệch tối đa 1 mức do nhiễu ước lượng."""
    for method in ("gray_world", "white_patch"):
        out = apply_white_balance(neutral_scene, method=method)
        assert np.abs(out.astype(int) - neutral_scene.astype(int)).max() <= 1


def test_gray_world_ignores_clipped_pixels(neutral_scene):
    """Mảng cháy sáng 255 (kênh bị cắt) không được kéo lệch ước lượng nguồn sáng."""
    casted = _cast(neutral_scene, WARM_CAST)
    casted[24:, 32:] = 255
    out = apply_white_balance(casted, method="gray_world")
    unclipped = out[:24].reshape(-1, 3).astype(np.float64).mean(axis=0)
    assert unclipped.max() - unclipped.min() < 2.0


# ---------------------------------------------------------------------------
# White Patch
# ---------------------------------------------------------------------------


def test_white_patch_makes_white_object_neutral(neutral_scene):
    casted = _cast(neutral_scene, WARM_CAST)
    out = apply_white_balance(casted, method="white_patch", percentile=99.0)
    white = out[:8, :8].reshape(-1, 3).astype(np.float64).mean(axis=0)
    assert white.max() - white.min() < 3.0


def test_white_patch_does_not_raise_exposure(neutral_scene):
    """Đích là kênh sáng nhất → kênh đó giữ nguyên, phơi sáng không bị đẩy lên 255."""
    casted = _cast(neutral_scene, WARM_CAST)
    out = apply_white_balance(casted, method="white_patch")
    np.testing.assert_allclose(out[..., 0], casted[..., 0], atol=1)


def test_white_patch_percentile_resists_hot_pixels(neutral_scene):
    """Vài điểm nóng màu lam không được quyết định gain khi dùng phân vị 99."""
    casted = _cast(neutral_scene, WARM_CAST)
    reference = apply_white_balance(casted, method="white_patch", percentile=99.0)
    hot = casted.copy()
    hot[47, 63] = [0, 0, 245]
    hot[46, 63] = [0, 0, 245]
    out = apply_white_balance(hot, method="white_patch", percentile=99.0)
    assert np.abs(out[:40].astype(int) - reference[:40].astype(int)).max() <= 1


# ---------------------------------------------------------------------------
# Mặt nạ: vừa là trọng số ước lượng, vừa là vùng hòa trộn
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["gray_world", "white_patch"])
def test_mask_selects_region_and_its_illuminant(neutral_scene, method):
    """Nửa trái ám ấm, nửa phải trung tính; mask nửa trái → chỉ trái được sửa, theo ánh sáng của trái."""
    image = neutral_scene.copy()
    image[:, :32] = _cast(neutral_scene[:, :32], WARM_CAST)
    image[:8, 32:40] = 230  # vật trắng cho nửa phải
    mask = np.zeros(image.shape[:2], dtype=np.float32)
    mask[:, :32] = 1.0

    out = apply_white_balance(image, mask=mask, method=method)

    np.testing.assert_array_equal(out[:, 32:], image[:, 32:])
    if method == "gray_world":
        # Trung bình kênh của chính vùng được chọn phải cân bằng
        assert np.ptp(_channel_means(out[:, :32])) < 1.5
    else:
        # Vật trắng nằm trong vùng được chọn phải trở về trung tính
        assert np.ptp(_channel_means(out[:8, :8])) < 3.0


def test_zero_mask_leaves_image_untouched(neutral_scene):
    image = _cast(neutral_scene, WARM_CAST)
    mask = np.zeros(image.shape[:2], dtype=np.float32)
    np.testing.assert_array_equal(apply_white_balance(image, mask=mask), image)


def test_full_mask_equals_global(neutral_scene):
    image = _cast(neutral_scene, WARM_CAST)
    mask = np.ones(image.shape[:2], dtype=np.float32)
    np.testing.assert_array_equal(apply_white_balance(image, mask=mask), apply_white_balance(image))


def test_extreme_cast_gain_is_limited():
    """Kênh lam gần như bằng 0 → gain bị giới hạn x4, không khuếch đại nhiễu thành mảng lam."""
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[..., 0] = 200
    image[..., 1] = 120
    image[..., 2] = 2
    out = apply_white_balance(image, method="gray_world")
    assert out[..., 2].max() <= 8


def test_grayscale_is_noop():
    gray = np.arange(64, dtype=np.uint8).reshape(8, 8)
    np.testing.assert_array_equal(apply_white_balance(gray), gray)
    assert apply_white_balance(gray[..., None]).shape == (8, 8, 1)


@pytest.mark.parametrize(
    "kwargs",
    [{"method": "retinex"}, {"percentile": 50.0}, {"percentile": 100.5}, {"percentile": "99"}],
)
def test_invalid_parameters_raise(neutral_scene, kwargs):
    with pytest.raises(ProcessingInputError):
        apply_white_balance(neutral_scene, **kwargs)


def test_rgba_input_raises():
    with pytest.raises(ProcessingInputError):
        apply_white_balance(np.zeros((4, 4, 4), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Tông ấm/lạnh bằng gain kênh (apply_color_balance.temperature_shift)
# ---------------------------------------------------------------------------


def test_temperature_keeps_black_point():
    """Nhân hệ số: điểm đen giữ 0 (cách cộng ±20 cũ làm vùng tối ám đỏ)."""
    black = np.zeros((4, 4, 3), dtype=np.uint8)
    np.testing.assert_array_equal(apply_color_balance(black, temperature_shift=1.0), black)


def test_temperature_mid_gray_shift_matches_previous_scale():
    """Tại xám 128, t = 1 dịch R +20 và B −20 như trước → giá trị playbook giữ nguyên ý nghĩa."""
    gray = np.full((4, 4, 3), 128, dtype=np.uint8)
    out = apply_color_balance(gray, temperature_shift=1.0).astype(int)
    assert out[0, 0].tolist() == [148, 128, 108]
    assert TEMPERATURE_GAIN_PER_UNIT == pytest.approx(20 / 128)


@pytest.mark.parametrize("shift", [-1.0, -0.3, 0.3, 1.0])
def test_temperature_direction_and_green_untouched(neutral_scene, shift):
    out = apply_color_balance(neutral_scene, temperature_shift=shift).astype(int)
    src = neutral_scene.astype(int)
    np.testing.assert_array_equal(out[..., 1], src[..., 1])
    assert np.sign(out[..., 0].mean() - src[..., 0].mean()) == np.sign(shift)
    assert np.sign(out[..., 2].mean() - src[..., 2].mean()) == -np.sign(shift)


def test_temperature_out_of_gain_domain_raises(neutral_scene):
    with pytest.raises(ProcessingInputError):
        apply_color_balance(neutral_scene, temperature_shift=7.0)
