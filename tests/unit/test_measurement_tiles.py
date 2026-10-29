"""Kiểm thử lưới tile, gộp mask (aggregate) và khung measure_image."""

import numpy as np
import pytest

from src.analyzer_evaluator.measurement import (
    Estimate,
    MeasurementReport,
    MeasurementResult,
    aggregate,
    measure_image,
)
from src.analyzer_evaluator.measurement.tiles import (
    MAP_KEYS,
    tile_grid_shape,
    tile_mean,
    tile_mean_at,
    tile_sum,
)


def _result(height: int, width: int, tile_size: int, value: float = 0.5) -> MeasurementResult:
    """Kết quả giả có bản đồ hằng số để kiểm thử gộp mask."""
    report = MeasurementReport(width=width, height=height, tile_size=tile_size)
    grid = tile_grid_shape(height, width, tile_size)
    maps = {key: np.full(grid, value, dtype=np.float32) for key in MAP_KEYS}
    return MeasurementResult(report=report, maps=maps, tile_size=tile_size)


def test_tile_grid_shape_rounds_up_partial_tiles() -> None:
    assert tile_grid_shape(100, 130, 64) == (2, 3)
    assert tile_grid_shape(64, 64, 64) == (1, 1)


def test_tile_mean_ignores_nan_and_uses_real_pixel_count() -> None:
    values = np.ones((10, 10), dtype=np.float32)
    values[0, 0] = np.nan
    values[8:, 8:] = 3.0  # tile mép 2×2 pixel, trung bình theo số pixel thực tế
    means = tile_mean(values, 8)
    assert means.shape == (2, 2)
    assert means[0, 0] == pytest.approx(1.0)  # 63 pixel = 1, một pixel NaN bị loại
    assert means[1, 1] == pytest.approx(3.0)


def test_tile_mean_is_nan_for_all_nan_tile() -> None:
    values = np.ones((8, 16), dtype=np.float32)
    values[:, :8] = np.nan
    means = tile_mean(values, 8)
    assert np.isnan(means[0, 0])
    assert means[0, 1] == pytest.approx(1.0)


def test_tile_sum_treats_nan_as_zero() -> None:
    values = np.full((4, 4), 0.5)
    values[0, 0] = np.nan
    sums = tile_sum(values, 4)
    assert sums.shape == (1, 1)
    assert sums[0, 0] == pytest.approx(7.5)


def test_tile_mean_at_counts_points_per_tile() -> None:
    rows = np.array([0, 1, 9, 9])
    cols = np.array([0, 2, 9, 8])
    values = np.array([2.0, 4.0, np.nan, 6.0])
    means, counts = tile_mean_at(rows, cols, values, 10, 10, 8)
    assert means.shape == (2, 2)
    assert means[0, 0] == pytest.approx(3.0)
    assert counts[0, 0] == 2
    # Hai mẫu ở tile (1,1): một NaN bị loại, còn lại 6.0
    assert means[1, 1] == pytest.approx(6.0)
    assert counts[1, 1] == 1
    assert np.isnan(means[1, 0])


def test_aggregate_soft_mask_weights_tiles() -> None:
    result = _result(16, 16, 8)
    result.maps["noise.sigma_luma"] = np.array([[2.0, 4.0], [np.nan, 4.0]], dtype=np.float32)
    mask = np.zeros((16, 16), dtype=np.float32)
    mask[:8, :8] = 1.0  # chỉ tile (0,0)
    mask[:8, 8:] = 0.5  # tile (0,1) có trọng số nửa
    stats = aggregate(result, mask, region="subject")
    assert stats.region == "subject"
    assert stats.area_fraction == pytest.approx(float(mask.mean()))
    sigma = stats.values["noise.sigma_luma"]
    # Tile (0,0) trọng số 64, tile (0,1) trọng số 32 → trung bình (2*64 + 4*32)/96
    assert sigma.value == pytest.approx((2.0 * 64 + 4.0 * 32) / 96.0)
    assert sigma.n_samples == 2
    assert 0.0 < sigma.confidence <= 1.0
    assert stats.mask_confidence == 1.0


def test_aggregate_zero_mask_has_zero_confidence_and_no_values() -> None:
    result = _result(16, 16, 8)
    stats = aggregate(result, np.zeros((16, 16), dtype=np.float32))
    assert stats.mask_confidence == 0.0
    assert stats.area_fraction == 0.0
    assert stats.values == {}


def test_aggregate_confidence_drops_when_mask_sits_on_unmeasurable_tiles() -> None:
    result = _result(16, 16, 8)
    result.maps["color.chroma"] = np.array([[1.0, np.nan], [np.nan, np.nan]], dtype=np.float32)
    mask = np.ones((16, 16), dtype=np.float32)
    stats = aggregate(result, mask)
    chroma = stats.values["color.chroma"]
    assert chroma.value == pytest.approx(1.0)
    assert chroma.n_samples == 1
    # 1 tile trên 4 có trọng số (coverage 0.25) × 1 tile < 4 tile (0.25) = 0.0625
    assert chroma.confidence == pytest.approx(0.0625)


def test_aggregate_rejects_mask_with_wrong_shape() -> None:
    result = _result(16, 16, 8)
    with pytest.raises(ValueError, match="does not match"):
        aggregate(result, np.ones((16, 15), dtype=np.float32))


def test_aggregate_accepts_trailing_singleton_channel() -> None:
    result = _result(16, 16, 8)
    stats = aggregate(result, np.ones((16, 16, 1), dtype=np.float32))
    assert stats.area_fraction == pytest.approx(1.0)
    assert stats.values["exposure.mean_l"].value == pytest.approx(0.5)


def test_estimate_rejects_nan_and_out_of_range_confidence() -> None:
    with pytest.raises(ValueError):
        Estimate(value=float("nan"), confidence=0.5)
    with pytest.raises(ValueError):
        Estimate(value=1.0, confidence=1.5)


@pytest.mark.parametrize("shape", [(32, 32, 3), (33, 47, 3), (1, 1, 3), (32, 32), (32, 32, 1)])
def test_measure_image_accepts_small_and_grayscale_inputs(shape: tuple) -> None:
    image = np.full(shape, 127, dtype=np.uint8)
    result = measure_image(image, tile_size=16)
    assert result.report.height == shape[0]
    assert result.report.width == shape[1]
    assert result.tile_size == 16


def test_measure_image_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="uint8"):
        measure_image(np.zeros((8, 8, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="expected"):
        measure_image(np.zeros((8, 8, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="empty"):
        measure_image(np.zeros((0, 8, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="tile_size"):
        measure_image(np.zeros((8, 8, 3), dtype=np.uint8), tile_size=0)
