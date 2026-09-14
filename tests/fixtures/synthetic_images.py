"""
Bộ sinh ảnh giả lập chuẩn hóa (Synthetic Image Generators) phục vụ Unit Tests.
Cung cấp các mẫu ảnh với các loại khuyết tật/thuộc tính kỹ thuật được kiểm soát chính xác.
"""

from typing import Literal, Tuple

import cv2
import numpy as np


def create_flat_image(
    val: int = 128,
    shape: Tuple[int, int, int] = (100, 100, 3),
    dtype: type = np.uint8,
) -> np.ndarray:
    """
    Tạo ảnh đơn sắc đồng nhất với mức xám xác định.
    Độ tương phản (contrast_std) và phương sai Laplacian phải bằng 0.
    """
    return np.full(shape, val, dtype=dtype)


def create_black_image(
    shape: Tuple[int, int, int] = (100, 100, 3),
) -> np.ndarray:
    """Tạo ảnh toàn đen (underexposed cực đoan, mean=0)."""
    return np.zeros(shape, dtype=np.uint8)


def create_white_image(
    shape: Tuple[int, int, int] = (100, 100, 3),
) -> np.ndarray:
    """Tạo ảnh toàn trắng (overexposed cực đoan, mean=255)."""
    return np.full(shape, 255, dtype=np.uint8)


def create_checkerboard_image(
    shape: Tuple[int, int, int] = (100, 100, 3),
    block_size: int = 10,
) -> np.ndarray:
    """
    Tạo ảnh bàn cờ xen kẽ trắng đen (0 và 255).
    Độ sắc nét (sharpness_laplacian_var) và contrast_std sẽ đạt giá trị rất cao.
    """
    h, w = shape[:2]
    img = np.zeros(shape, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if ((i // block_size) + (j // block_size)) % 2 == 0:
                img[i, j] = 255
    return img


def create_noisy_image(
    base_image: np.ndarray,
    noise_std: float = 20.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Thêm nhiễu trắng Gauss (Additive White Gaussian Noise) có kiểm soát độ lệch chuẩn.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_std, base_image.shape)
    noisy = base_image.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def create_salt_pepper_image(
    base_image: np.ndarray,
    prob: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """
    Thêm nhiễu muối tiêu (Impulse / Salt & Pepper Noise) với xác suất điểm ảnh bị nhiễu prob.
    """
    rng = np.random.default_rng(seed)
    output = base_image.copy()
    num_salt = int(prob * base_image.size * 0.5)
    num_pepper = int(prob * base_image.size * 0.5)

    # Muối (White = 255)
    coords = [rng.integers(0, i - 1, num_salt) for i in base_image.shape]
    output[tuple(coords)] = 255

    # Tiêu (Black = 0)
    coords = [rng.integers(0, i - 1, num_pepper) for i in base_image.shape]
    output[tuple(coords)] = 0

    return output


def create_blurred_image(
    base_image: np.ndarray,
    kernel_size: int = 15,
) -> np.ndarray:
    """
    Làm mờ ảnh bằng bộ lọc Gauss (Gaussian Blur).
    Làm giảm mạnh phương sai toán tử Laplacian.
    """
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(base_image, (k, k), 0)


def create_color_cast_image(
    base_image: np.ndarray,
    cast_type: Literal["warm", "cool", "greenish"] = "warm",
    intensity: int = 40,
) -> np.ndarray:
    """
    Tạo ảnh bị lệch màu (Color Cast).
    - 'warm': Tăng kênh R (+intensity), giảm kênh B (-intensity/2).
    - 'cool': Tăng kênh B (+intensity), giảm kênh R (-intensity/2).
    - 'greenish': Tăng kênh G (+intensity), giảm kênh R & B.
    """
    if len(base_image.shape) < 3 or base_image.shape[2] != 3:
        raise ValueError("Ảnh đầu vào phải là ảnh màu RGB 3 kênh")

    img_float = base_image.astype(np.float64)
    if cast_type == "warm":
        img_float[:, :, 0] += intensity
        img_float[:, :, 2] -= intensity // 2
    elif cast_type == "cool":
        img_float[:, :, 2] += intensity
        img_float[:, :, 0] -= intensity // 2
    elif cast_type == "greenish":
        img_float[:, :, 1] += intensity
        img_float[:, :, 0] -= intensity // 2
        img_float[:, :, 2] -= intensity // 2
    else:
        raise ValueError(f"Không hỗ trợ cast_type '{cast_type}'")

    return np.clip(img_float, 0, 255).astype(np.uint8)


def create_underexposed_image(
    base_image: np.ndarray,
    factor: float = 0.3,
) -> np.ndarray:
    """Tạo ảnh thiếu sáng (underexposed) bằng phép nhân hệ số tuyến tính hoặc gamma."""
    img_float = base_image.astype(np.float64) * factor
    return np.clip(img_float, 0, 255).astype(np.uint8)


def create_overexposed_image(
    base_image: np.ndarray,
    factor: float = 2.5,
) -> np.ndarray:
    """Tạo ảnh chói sáng (overexposed) bị clipping ở các vùng highlight."""
    img_float = base_image.astype(np.float64) * factor
    return np.clip(img_float, 0, 255).astype(np.uint8)
