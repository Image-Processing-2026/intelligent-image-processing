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


@pytest.fixture(autouse=True)
def _offline_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Đảm bảo test không bao giờ gọi Gemini thật: xóa GEMINI_API_KEY có thể đã được
    nạp từ .env (ví dụ khi import src.api.main). Test cần key giả tự đặt bằng patch.dict.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def _no_case_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Tắt bộ nhớ ca bệnh trong mọi test (src.api.main bật nó khi import), để test không ghi
    vào data/memory. Test cần bộ nhớ tự cấu hình một file tạm.
    """
    from src.agent import memory

    monkeypatch.setattr(memory, "_MEMORY", None)


@pytest.fixture(autouse=True)
def _single_perceive_sample(monkeypatch: pytest.MonkeyPatch):
    """
    Các test mock một phản hồi Perceive cho mỗi vòng nên luôn chạy với 1 lần, kể cả khi máy đặt
    biến môi trường PERCEIVE_SAMPLES. Test self-consistency tự đặt lại.
    Cache mask chủ thể của region_engine là toàn process: xóa trước mỗi test.
    """
    from src.agent import perception
    from src.region_engine import clear_subject_mask_cache

    monkeypatch.delenv(perception.PERCEIVE_SAMPLES_ENV, raising=False)
    monkeypatch.setattr(perception, "PERCEIVE_SAMPLES", 1)
    clear_subject_mask_cache()
    yield
    clear_subject_mask_cache()


@pytest.fixture(autouse=True)
def _no_verify_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Fusion (P6) gọi thêm Gemini verify khi bảng quyết định yêu cầu. Các test cũ mock đúng số
    phản hồi Perceive/Plan, nên verify tắt mặc định; test fusion tự bật (VERIFY_ENABLED = True).
    Bằng chứng yếu từ giả thuyết bị loại cũng tắt, kể cả khi máy đặt biến môi trường.
    """
    from src.agent import fusion, perception

    monkeypatch.setattr(perception, "VERIFY_ENABLED", False)
    monkeypatch.delenv(fusion.USE_WEAK_REJECTED_ENV, raising=False)


@pytest.fixture(autouse=True)
def _no_review_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Cổng review (R-08) gọi thêm Gemini ở cuối pipeline. Các test cũ mock đúng số phản hồi
    Perceive/Plan nên review tắt mặc định; test review tự bật (REVIEW_ENABLED = True).
    """
    from src.agent import review

    monkeypatch.setattr(review, "REVIEW_ENABLED", False)
    monkeypatch.delenv(review.REVIEW_MODEL_ENV, raising=False)


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
