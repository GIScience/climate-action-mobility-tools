from unittest.mock import patch

import numpy as np

from mobility_tools.slope import get_paths_slopes
from mobility_tools.slope.slope import calc_slope


def test_calc_slope():
    path_elev = [0, 1, 10, 5, 5, 0]
    segment_lengths = np.array([10, 10, 10, 0, 20])
    expected_slopes = np.array([10, 90, -50, np.nan, -25])

    calculated_slopes = calc_slope(path_elev, segment_lengths)
    print(calculated_slopes)

    assert np.allclose(calculated_slopes, expected_slopes, equal_nan=True)


def test_get_paths_slopes(default_test_slope_path):
    with patch('mobility_tools.slope.slope.get_point_elevations') as mock_path_smoothed_elevs:
        mock_path_smoothed_elevs.return_value = [0, 1, 10, 5, 0]

        calculated_slopes_grouped = get_paths_slopes(default_test_slope_path, s3settings=None, segment_length=None)[
            'slope'
        ].values

    expected_slopes = np.array([10, 90, -50, -25])

    assert np.allclose(calculated_slopes_grouped, expected_slopes, rtol=1.0e-2)
