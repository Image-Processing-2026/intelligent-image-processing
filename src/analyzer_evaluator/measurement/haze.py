"""Đo sương mù: median kênh tối (dark channel) ngoài vùng trời ước lượng.

Dark channel = min kênh RGB, lọc min trên cửa sổ 15×15. Ảnh không sương có dark channel gần 0;
sương nâng nó lên. Vùng trời (nửa trên, sáng, ít màu, phẳng) bị loại vì trời luôn sáng và làm
dark channel cao ngay cả khi không có sương.
"""

import cv2
import numpy as np

from ._util import Section, put_estimate

DARK_WINDOW = 15
SKY_TOP_FRACTION = 0.5
SKY_LIGHTNESS = 60.0
SKY_CHROMA = 15.0
SKY_FLATNESS = 0.03
FLATNESS_WINDOW = 5


def _sky_mask(rgb01: np.ndarray, l_star: np.ndarray, chroma: np.ndarray) -> np.ndarray:
    """Pixel nghi là trời: nửa trên ảnh, sáng, ít màu và phẳng cục bộ."""
    height = rgb01.shape[0]
    top = np.arange(height)[:, None] < SKY_TOP_FRACTION * height
    bright = l_star > SKY_LIGHTNESS
    neutral = chroma < SKY_CHROMA
    gray = cv2.cvtColor(rgb01, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((FLATNESS_WINDOW, FLATNESS_WINDOW), dtype=np.uint8)
    local_range = cv2.dilate(gray, kernel) - cv2.erode(gray, kernel)
    flat = local_range < SKY_FLATNESS
    return top & bright & neutral & flat


def measure_haze(rgb: np.ndarray) -> Section:
    """Đo dark channel median trên ảnh RGB uint8 (H, W, 3)."""
    section = Section()
    rgb01 = rgb.astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb01, cv2.COLOR_RGB2LAB)
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    dark = cv2.erode(rgb01.min(axis=2), np.ones((DARK_WINDOW, DARK_WINDOW), dtype=np.uint8))
    # Trời chỉ nằm ở nửa trên, nên luôn còn ít nhất nửa ảnh để đo.
    usable = ~_sky_mask(rgb01, lab[..., 0], chroma)
    value = float(np.median(dark[usable]))
    put_estimate(section, "dark_channel_median", value, float(usable.mean()), int(usable.sum()))
    return section
