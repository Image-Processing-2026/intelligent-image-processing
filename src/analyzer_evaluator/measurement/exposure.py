"""Đo phơi sáng: L* (CIELAB), clip theo kênh, vùng cháy (connected components), key ảnh.

Vùng cháy là blob clip sáng có diện tích ≥ BURNT_BLOB_FRACTION của ảnh. Blob nhỏ hơn (phản xạ
gương, đèn) không được coi là cháy. Key là trung bình log luminance tuyến tính (Ansel Adams).
"""

import math

import cv2
import numpy as np

from ._util import Section, put_estimate, srgb_to_linear
from .tiles import tile_mean

CLIP_HIGH = 250
CLIP_LOW = 5
BURNT_BLOB_FRACTION = 0.005
KEY_EPSILON = 1e-3
# Hệ số Rec.709 cho luminance tuyến tính.
REC709_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _largest_burnt_blob(clip_mask: np.ndarray) -> float:
    """Tỉ lệ diện tích của blob cháy lớn nhất; 0 nếu không có blob nào ≥ BURNT_BLOB_FRACTION."""
    if not clip_mask.any():
        return 0.0
    _, _, stats, _ = cv2.connectedComponentsWithStats(clip_mask.astype(np.uint8), connectivity=8)
    fractions = stats[1:, cv2.CC_STAT_AREA] / clip_mask.size
    burnt = fractions[fractions >= BURNT_BLOB_FRACTION]
    return float(burnt.max()) if burnt.size else 0.0


def measure_exposure(rgb: np.ndarray, tile_size: int) -> Section:
    """Đo phơi sáng trên ảnh RGB uint8 (H, W, 3)."""
    height, width = rgb.shape[:2]
    total = height * width
    rgb01 = rgb.astype(np.float32) / 255.0
    l_star = cv2.cvtColor(rgb01, cv2.COLOR_RGB2LAB)[..., 0]

    high = rgb >= CLIP_HIGH
    low = rgb <= CLIP_LOW
    high_any = high.any(axis=2)
    low_any = low.any(axis=2)
    linear_luma = srgb_to_linear(rgb01) @ REC709_WEIGHTS
    key = math.exp(float(np.mean(np.log(linear_luma + KEY_EPSILON))))
    p5, p95 = np.percentile(l_star, [5, 95])

    section = Section()
    put_estimate(section, "mean_l", float(l_star.mean()), 1.0, total)
    put_estimate(section, "p5_l", float(p5), 1.0, total)
    put_estimate(section, "p95_l", float(p95), 1.0, total)
    put_estimate(section, "clip_high_any", float(high_any.mean()), 1.0, total)
    put_estimate(section, "clip_high_all", float(high.all(axis=2).mean()), 1.0, total)
    put_estimate(section, "clip_low_any", float(low_any.mean()), 1.0, total)
    put_estimate(section, "clip_low_all", float(low.all(axis=2).mean()), 1.0, total)
    put_estimate(section, "largest_clip_blob", _largest_burnt_blob(high_any), 1.0, total)
    put_estimate(section, "key", key, 1.0, total)

    section.maps["exposure.mean_l"] = tile_mean(l_star, tile_size)
    section.maps["exposure.clip_high"] = tile_mean(high_any.astype(np.float32), tile_size)
    section.maps["exposure.clip_low"] = tile_mean(low_any.astype(np.float32), tile_size)
    return section
