"""Tiện ích ảnh dùng chung trong Module 4 (không xử lý điểm ảnh theo nghĩa toolbox)."""

import cv2
import numpy as np

# Cạnh dài tối đa của ảnh preview: đo vùng, render phiên bản và so sánh không cần ảnh gốc
PREVIEW_MAX_SIDE = 1024


def downscale(image: np.ndarray, max_side: int = PREVIEW_MAX_SIDE) -> np.ndarray:
    """Thu nhỏ ảnh (giữ tỉ lệ) để cạnh dài không vượt max_side; ảnh nhỏ hơn giữ nguyên."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / longest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    # cv2.resize bỏ trục kênh của ảnh (H, W, 1) → khôi phục để giữ đúng định dạng đầu vào
    return resized[:, :, None] if image.ndim == 3 and resized.ndim == 2 else resized
