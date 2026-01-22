import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from vcr import use_cassette
from approvaltests import verify

from mobility_tools.detour_factors_batched import (
    calculate_detour_factors,
    extract_coordinates,
    get_detour_factors_batched,
)
from mobility_tools.ors_settings import ORSSettings


@use_cassette
def test_get_detour_factors_batched(default_ors_settings):
    aoi = shapely.box(8.671217, 49.408404, 8.6800658, 49.410400)

    result = get_detour_factors_batched(aoi, default_ors_settings, profile='foot-walking')

    assert 'detour_factor' in result.columns
    for detour_factor in result.detour_factor:
        assert isinstance(detour_factor, float)
    assert result.active_geometry_name is not None

@use_cassette
def test_get_detour_factors_approval_test(small_aoi, default_ors_settings):
    
    result = get_detour_factors_batched(small_aoi, default_ors_settings, profile='foot-walking')
    verify(result.to_csv())


def test_extract_coordinates():
    cell_center = shapely.Point(0.0, 0.0)
    ring = [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (1.0, 1.0)]
    geometry = shapely.Polygon(ring)

    expected_corners = ring.copy()
    expected_corners.pop()

    row = pd.Series(data=[cell_center, geometry], index=['cell_center', 'geometry'])
    result = extract_coordinates(row)
    center = result['center']
    corners = result['corners']

    assert center == (0.0, 0.0)
    assert corners == expected_corners


def test_calculate_detour_factors():
    # TODO think about how to make this test more understandable, made up coordinates for easy math
    # atm this basicall tests the current implementation against it's own results
    chunk_distances = [
        {
            'distances': [66.4, 73.7, 101.2, 52.4, 72.9, 833.4],
            'snapped_coordinates': {
                'center': [8.674202, 49.409577],
                'corners': [
                    [8.673291, 49.409591],
                    [8.67319, 49.409592],
                    [8.673858, 49.40931],
                    [8.674926, 49.409588],
                    [8.675209, 49.409593],
                    [8.674519, 49.4113],
                ],
            },
        },
        {
            'distances': [495.2, 397.8, 416.5, 500.7, 544.8, 495.1],
            'snapped_coordinates': {
                'center': [8.677646, 49.409741],
                'corners': [
                    [8.676915, 49.410034],
                    [8.676694, 49.409398],
                    [8.677398, 49.408886],
                    [8.678332, 49.409522],
                    [8.67864, 49.409852],
                    [8.677954, 49.409818],
                ],
            },
        },
    ]

    result = calculate_detour_factors(
        chunk_distances, transform=Transformer.from_crs(crs_from='EPSG:4326', crs_to='EPSG:32632')
    )

    expected_result = [np.float64(1.8221085635843355), np.float64(9.080471601313818)]

    assert result == expected_result
