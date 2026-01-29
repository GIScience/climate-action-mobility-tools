import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from approvaltests import verify
from pyproj import Transformer
from vcr import use_cassette

from mobility_tools.detour_factors_batched import (
    calculate_detour_factors,
    create_waypoint_path,
    exclude_ferries,
    extract_coordinates,
    extract_data_from_ors_result,
    get_detour_factors_batched,
)


@use_cassette
def test_get_detour_factors_batched(default_ors_settings):
    aoi = shapely.box(8.671217, 49.408404, 8.6800658, 49.410400)
    paths = gpd.GeoDataFrame(data={'test': [0]}, geometry=[aoi], crs='EPSG:4326')

    result = get_detour_factors_batched(aoi, paths, default_ors_settings, profile='foot-walking')

    assert 'detour_factor' in result.columns
    for detour_factor in result.detour_factor:
        assert isinstance(detour_factor, float)
    assert result.active_geometry_name is not None


@use_cassette
def test_get_detour_factors_approval_test(small_aoi, default_ors_settings):
    paths = gpd.GeoDataFrame(data={'test': [0]}, geometry=[small_aoi], crs='EPSG:4326')
    result = get_detour_factors_batched(small_aoi, paths, default_ors_settings, profile='foot-walking')
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


@pytest.fixture
def default_chunk_coordinates() -> list[dict]:
    chunk_coordinates = [
        {'center': 'center0', 'corners': [f'corner0-{x}' for x in range(6)]},
        {'center': 'center1', 'corners': [f'corner1-{x}' for x in range(6)]},
    ]

    return chunk_coordinates


@pytest.fixture
def expected_hexcell_corner_paths() -> list[str]:
    first_cell_path = [
        'center0',
        'corner0-0',
        'corner0-1',
        'center0',
        'corner0-2',
        'corner0-3',
        'center0',
        'corner0-4',
        'corner0-5',
        'center0',
    ]
    second_cell_path = [
        'center1',
        'corner1-0',
        'corner1-1',
        'center1',
        'corner1-2',
        'corner1-3',
        'center1',
        'corner1-4',
        'corner1-5',
        'center1',
    ]
    expected = first_cell_path + second_cell_path
    return expected


def test_create_waypoint_path(default_chunk_coordinates, expected_hexcell_corner_paths):
    received = create_waypoint_path(chunk_coordinates=default_chunk_coordinates)

    assert received == expected_hexcell_corner_paths


def test_extract_data_from_ors_result(expected_hexcell_corner_paths, default_chunk_coordinates):
    json = {
        'features': [
            {
                'geometry': {'coordinates': expected_hexcell_corner_paths},
                'properties': {
                    'way_points': list(range(len(expected_hexcell_corner_paths))),
                    'segments': [{'distance': distance} for distance in range(len(expected_hexcell_corner_paths) - 1)],
                },
            }
        ]
    }

    expected = [
        {'distances': [0, 2, 3, 5, 6, 8], 'snapped_coordinates': default_chunk_coordinates[0]},
        {'distances': [10, 12, 13, 15, 16, 18], 'snapped_coordinates': default_chunk_coordinates[1]},
    ]
    recieved = extract_data_from_ors_result(json)

    assert expected == recieved


def test_extract_data_from_ors_result_zero_distanc_snapping(expected_hexcell_corner_paths, default_chunk_coordinates):
    # simulating that the first corner snapped to the same point as the center
    paths_with_same_snapping = expected_hexcell_corner_paths.copy()
    paths_with_same_snapping[1] = 'center0'

    json = {
        'features': [
            {
                'geometry': {'coordinates': paths_with_same_snapping},
                'properties': {
                    'way_points': list(range(len(expected_hexcell_corner_paths))),
                    'segments': [{'distance': distance} for distance in range(len(expected_hexcell_corner_paths) - 1)],
                },
            }
        ]
    }

    # first corner should be missing as it snapped to the center
    snapped_chunk_coordinates = default_chunk_coordinates[0]
    snapped_chunk_coordinates['corners'].pop(0)

    expected = [
        {'distances': [2, 3, 5, 6, 8], 'snapped_coordinates': snapped_chunk_coordinates},
        {'distances': [10, 12, 13, 15, 16, 18], 'snapped_coordinates': default_chunk_coordinates[1]},
    ]
    recieved = extract_data_from_ors_result(json)

    assert expected == recieved


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


def test_exclude_ferries():
    # TODO rewrite test so it's a bit more useful and has some cells that pass
    snapped_input = pd.DataFrame(
        data={'snapped_location': [[8.773085, 49.376161], None], 'snapped_distance': [122.49, None]},
        index=['8a1faad69927fff', '8a1faad6992ffff'],
    ).rename_axis('id')

    paths = gpd.GeoDataFrame(geometry=[shapely.LineString([(0, 0), (1, 1)])])

    result = exclude_ferries(snapped_input, paths)

    verify(result.to_json())
