import pandas as pd
import shapely
from vcr import use_cassette

from mobility_tools.detour_factors_batched import extract_coordinates, get_detour_factors_batched


@use_cassette
def test_get_detour_factors_batched(default_ors_settings):
    aoi = shapely.box(8.671217, 49.408404, 8.6800658, 49.410400)

    result = get_detour_factors_batched(aoi, default_ors_settings, profile='foot-walking')

    # approval test here is hard because batching happens in non-deterministic order,
    # so recorded request returns static data for varied orders of cells
    assert 'detour_factor' in result.columns
    for detour_factor in result.detour_factor:
        assert isinstance(detour_factor, float)
    assert result.active_geometry_name is not None


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
