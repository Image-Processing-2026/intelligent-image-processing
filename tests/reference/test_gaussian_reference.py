"""Reproduce the public SciPy ``gaussian_filter`` example used in the plan."""

import numpy as np
from scipy.ndimage import gaussian_filter


def test_scipy_public_gaussian_filter_example() -> None:
    values = np.arange(0, 50, 2, dtype=np.int64).reshape(5, 5)
    expected = np.array(
        [
            [4, 6, 8, 9, 11],
            [10, 12, 14, 15, 17],
            [20, 22, 24, 25, 27],
            [29, 31, 33, 34, 36],
            [35, 37, 39, 40, 42],
        ],
        dtype=np.int64,
    )

    actual = gaussian_filter(values, sigma=1)

    np.testing.assert_array_equal(actual, expected)
