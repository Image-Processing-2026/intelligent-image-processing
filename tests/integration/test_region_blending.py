"""Integration tests for soft-mask creation, blending, and Module 3 wrappers."""

import cv2
import numpy as np

from src.processing_engine.base import apply_region_op
from src.processing_engine.exposure_contrast import apply_gamma
from src.region_engine.mask_utils import blend_regions, create_soft_mask


def test_create_soft_mask_then_apply_region_op_matches_direct_blend() -> None:
    original = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    binary = np.zeros((8, 10), dtype=np.uint8)
    binary[2:6, 3:8] = 1
    mask = create_soft_mask(binary, feather_radius=3)

    def invert(image: np.ndarray) -> np.ndarray:
        return 255 - image

    expected_processed = invert(original)
    expected = blend_regions(original, expected_processed, mask)
    actual = apply_region_op(original, mask, invert)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.uint8
    assert actual.shape == original.shape


def test_processing_engine_gamma_wrapper_blends_full_frame_result() -> None:
    original = np.array(
        [
            [[0, 30, 60], [90, 120, 150]],
            [[180, 210, 240], [10, 100, 200]],
        ],
        dtype=np.uint8,
    )
    mask = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)
    inverse_gamma = 1.0 / 1.5
    lut = np.array([((i / 255.0) ** inverse_gamma) * 255 for i in range(256)]).astype(np.uint8)
    processed = cv2.LUT(original, lut)

    actual = apply_gamma(original, mask=mask, gamma=1.5)
    expected = blend_regions(original, processed, mask)

    np.testing.assert_array_equal(actual, expected)
