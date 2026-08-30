"""
Unit tests for Module 2: Region Engine & Mask Utilities.
"""

import numpy as np
from src.region_engine.mask_utils import blend_regions, create_soft_mask
from src.region_engine.spatial import create_bbox_mask, create_quadrant_mask


def test_soft_mask_creation():
    """Kiểm tra tạo mặt nạ mềm float32 trong khoảng [0.0, 1.0]."""
    bin_mask = np.zeros((100, 100), dtype=np.uint8)
    bin_mask[20:80, 20:80] = 255

    soft = create_soft_mask(bin_mask, feather_radius=5)
    assert soft.dtype == np.float32
    assert soft.min() >= 0.0
    assert soft.max() <= 1.0
    assert soft.shape == (100, 100)


def test_blend_regions():
    """Kiểm tra tính đúng đắn của phép hòa trộn alpha."""
    orig = np.zeros((50, 50, 3), dtype=np.uint8)
    proc = np.ones((50, 50, 3), dtype=np.uint8) * 200
    mask = np.ones((50, 50), dtype=np.float32) * 0.5

    blended = blend_regions(orig, proc, mask)
    # 0.5 * 200 + 0.5 * 0 = 100
    assert np.allclose(blended, 100, atol=2)


def test_spatial_quadrant_mask():
    """Kiểm tra tạo mặt nạ phân vùng không gian."""
    mask_top = create_quadrant_mask((100, 100), "top", feather_radius=0)
    assert mask_top[10, 50] == 1.0
    assert mask_top[90, 50] == 0.0
