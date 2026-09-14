"""
Cấu hình và fixtures toàn cục cho Pytest.
Cung cấp các mẫu ảnh giả lập cho các bài kiểm thử unit và integration.
"""

import numpy as np
import pytest

from tests.fixtures.synthetic_images import (
    create_black_image,
    create_blurred_image,
    create_checkerboard_image,
    create_color_cast_image,
    create_flat_image,
    create_noisy_image,
    create_overexposed_image,
    create_salt_pepper_image,
    create_underexposed_image,
    create_white_image,
)


@pytest.fixture
def flat_gray_image() -> np.ndarray:
    """Ảnh RGB xám đồng nhất mức 128 — contrast=0, noise=0."""
    return create_flat_image(val=128, shape=(100, 100, 3))


@pytest.fixture
def black_image() -> np.ndarray:
    """Ảnh toàn đen — underexposed cực đoan, mean=0."""
    return create_black_image(shape=(100, 100, 3))


@pytest.fixture
def white_image() -> np.ndarray:
    """Ảnh toàn trắng — overexposed cực đoan, mean=255."""
    return create_white_image(shape=(100, 100, 3))


@pytest.fixture
def checkerboard_image() -> np.ndarray:
    """Ảnh bàn cờ 10x10 — tương phản và độ sắc nét Laplacian cực cao."""
    return create_checkerboard_image(shape=(100, 100, 3), block_size=10)


@pytest.fixture
def noisy_image(flat_gray_image: np.ndarray) -> np.ndarray:
    """Ảnh có nhiễu Gaussian sigma=20 trên nền xám 128."""
    return create_noisy_image(flat_gray_image, noise_std=20.0, seed=42)


@pytest.fixture
def salt_pepper_image(flat_gray_image: np.ndarray) -> np.ndarray:
    """Ảnh có nhiễu muối tiêu xác suất 5%."""
    return create_salt_pepper_image(flat_gray_image, prob=0.05, seed=42)


@pytest.fixture
def blurred_checkerboard_image(checkerboard_image: np.ndarray) -> np.ndarray:
    """Ảnh bàn cờ đã bị làm mờ bởi Gaussian kernel 15x15."""
    return create_blurred_image(checkerboard_image, kernel_size=15)


@pytest.fixture
def warm_cast_image(flat_gray_image: np.ndarray) -> np.ndarray:
    """Ảnh bị lệch màu ấm (warm cast)."""
    return create_color_cast_image(flat_gray_image, cast_type="warm", intensity=40)


@pytest.fixture
def cool_cast_image(flat_gray_image: np.ndarray) -> np.ndarray:
    """Ảnh bị lệch màu lạnh (cool cast)."""
    return create_color_cast_image(flat_gray_image, cast_type="cool", intensity=40)


@pytest.fixture
def underexposed_image(flat_gray_image: np.ndarray) -> np.ndarray:
    """Ảnh bị thiếu sáng (underexposed)."""
    return create_underexposed_image(flat_gray_image, factor=0.3)


@pytest.fixture
def overexposed_image(flat_gray_image: np.ndarray) -> np.ndarray:
    """Ảnh bị cháy sáng (overexposed)."""
    return create_overexposed_image(flat_gray_image, factor=2.5)
