"""Contract and geometry tests for :func:`create_bbox_mask`."""

import numpy as np
import pytest

from src.region_engine.spatial import create_bbox_mask


def test_public_rectangle_baselines_use_half_open_xy_coordinates() -> None:
    expected_extent = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    expected_end = np.array(
        [
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(create_bbox_mask((5, 5), (1, 1, 4, 4), 0), expected_extent)
    np.testing.assert_array_equal(create_bbox_mask((5, 5), (1, 0, 4, 4), 0), expected_end)


@pytest.mark.parametrize(
    ("bbox", "expected_sum"),
    [
        ((1, 1, 5, 3), 8),
        ((-2, -1, 3, 2), 6),
        ((4, 2, 9, 8), 4),
        ((-5, -5, 10, 10), 24),
        ((5, 3, 6, 4), 1),
        ((6, 0, 9, 3), 0),
        ((-3, 0, -1, 3), 0),
        ((0, 4, 3, 6), 0),
        ((0, -3, 3, -1), 0),
        ((2, 1, 2, 3), 0),
        ((1, 2, 4, 2), 0),
        ((1, 1, 1, 1), 0),
    ],
)
def test_clipping_and_degenerate_boxes(bbox: tuple[int, ...], expected_sum: int) -> None:
    shape = (4, 6)
    actual = create_bbox_mask(shape, bbox, feather_radius=0)

    assert actual.dtype == np.float32
    assert actual.shape == shape
    assert float(actual.sum()) == expected_sum
    assert np.isfinite(actual).all()
    assert float(actual.min()) >= 0.0
    assert float(actual.max()) <= 1.0


def test_one_pixel_and_row_column_boxes_keep_orientation() -> None:
    np.testing.assert_array_equal(
        create_bbox_mask((1, 5), (1, 0, 4, 1), 0),
        np.array([[0, 1, 1, 1, 0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        create_bbox_mask((5, 1), (0, 1, 1, 4), 0),
        np.array([[0], [1], [1], [1], [0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(create_bbox_mask((1, 1), (0, 0, 1, 1), 15), [[1.0]])


@pytest.mark.parametrize("bbox", [(4, 1, 2, 3), (1, 3, 4, 2), (9, 1, 8, 3)])
def test_reversed_bbox_is_rejected_before_clipping(bbox: tuple[int, int, int, int]) -> None:
    with pytest.raises(ValueError, match="ordered"):
        create_bbox_mask((4, 6), bbox, feather_radius=0)


@pytest.mark.parametrize(
    ("shape", "bbox"),
    [
        ((4, 6, 3), (1, 1, 5, 3)),
        (np.array([[4, 6]]), (1, 1, 5, 3)),
        ((4, 6, 1), (1, 1, 5, 3)),
    ],
)
def test_shape_must_be_a_two_element_one_dimensional_vector(shape: object, bbox: object) -> None:
    with pytest.raises(ValueError):
        create_bbox_mask(shape, bbox, feather_radius=0)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [None, "4,6", {"height": 4, "width": 6}, 4])
def test_shape_container_type_is_not_coerced(value: object) -> None:
    with pytest.raises(TypeError):
        create_bbox_mask(value, (1, 1, 2, 2), feather_radius=0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [
        (4.0, 6),
        (True, 6),
        (4, np.bool_(False)),
        (4, "6"),
        (4, np.float64(6)),
    ],
)
def test_shape_scalars_must_be_integers(value: object) -> None:
    with pytest.raises(TypeError):
        create_bbox_mask(value, (1, 1, 2, 2), feather_radius=0)  # type: ignore[arg-type]


@pytest.mark.parametrize("bbox", [(1, 1, 2), (1, 1, 2, 2, 3), np.array([[1, 1, 2, 2]])])
def test_bbox_length_and_dimension_are_validated(bbox: object) -> None:
    with pytest.raises(ValueError):
        create_bbox_mask((4, 6), bbox, feather_radius=0)  # type: ignore[arg-type]


@pytest.mark.parametrize("bbox", [(1.0, 1, 2, 2), (1, True, 2, 2), (1, 1, "2", 2)])
def test_bbox_scalars_must_be_integers(bbox: object) -> None:
    with pytest.raises(TypeError):
        create_bbox_mask((4, 6), bbox, feather_radius=0)  # type: ignore[arg-type]


@pytest.mark.parametrize("radius", [-1, np.int64(-1)])
def test_negative_feather_radius_is_rejected_for_empty_and_full_boxes(radius: object) -> None:
    for bbox in ((2, 1, 2, 3), (-5, -5, 10, 10)):
        with pytest.raises(ValueError):
            create_bbox_mask((4, 6), bbox, feather_radius=radius)  # type: ignore[arg-type]


@pytest.mark.parametrize("radius", [True, np.bool_(False), 2.5, "15"])
def test_non_integer_feather_radius_is_rejected_for_empty_and_full_boxes(radius: object) -> None:
    for bbox in ((2, 1, 2, 3), (-5, -5, 10, 10)):
        with pytest.raises(TypeError):
            create_bbox_mask((4, 6), bbox, feather_radius=radius)  # type: ignore[arg-type]


def test_numpy_integers_read_only_inputs_and_output_ownership() -> None:
    shape = np.array([4, 6], dtype=np.int64)
    bbox = np.array([1, 1, 5, 3], dtype=np.int64)
    shape.setflags(write=False)
    bbox.setflags(write=False)
    shape_before = shape.copy()
    bbox_before = bbox.copy()

    first = create_bbox_mask(shape, bbox, feather_radius=np.int64(0))
    second = create_bbox_mask(shape, bbox, feather_radius=np.int64(0))

    np.testing.assert_array_equal(shape, shape_before)
    np.testing.assert_array_equal(bbox, bbox_before)
    np.testing.assert_array_equal(first, second)
    assert not np.shares_memory(first, second)


def test_invalid_positive_shape_dimensions_are_rejected() -> None:
    with pytest.raises(ValueError):
        create_bbox_mask((0, 4), (0, 0, 1, 1), 0)
    with pytest.raises(ValueError):
        create_bbox_mask((-1, 4), (0, 0, 1, 1), 0)
