import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from approvaltests import verify
from pyproj import Transformer
from responses import RequestsMock, matchers
from vcr import use_cassette

from mobility_tools.detour_factors.detour_factors import (
    _PathResponse,
    calculate_detour_factors,
    create_waypoint_path,
    exclude_ferries,
    extract_coordinates,
    extract_data_from_ors_result,
    get_detour_factors,
    ors_request,
)
from mobility_tools.settings import ORSSettings

# Hexagons with 100m distances to points, in UTM CRS = EPSG:25832
HEX1 = {
    'center': [8.6839368, 49.4158878],
    'corners': [
        [8.6839310, 49.4167872],
        [8.6839426, 49.4149883],
        [8.6851336, 49.4154413],
        [8.6827457, 49.4154347],
        [8.6851279, 49.4163407],
        [8.6827399, 49.4163342],
    ],
}
HEX2 = {
    'center': [8.6863190, 49.4167937],
    'corners': [
        [8.6851279, 49.4163407],
        [8.6875102, 49.4172467],
        [8.6875159, 49.4163472],
        [8.6851221, 49.4172402],
        [8.6863133, 49.4176932],
        [8.6863247, 49.4158943],
    ],
}


@use_cassette
def test_get_detour_factors(default_ors_settings: ORSSettings):
    aoi = shapely.box(8.671217, 49.408404, 8.6800658, 49.410400)
    paths = gpd.GeoDataFrame(data={'test': [0]}, geometry=[aoi], crs='EPSG:4326')

    result = get_detour_factors(aoi, paths, default_ors_settings, profile='foot-walking')

    assert 'detour_factor' in result.columns
    for detour_factor in result.detour_factor:
        assert isinstance(detour_factor, float)
    assert result.active_geometry_name is not None


@use_cassette
def test_get_detour_factors_approval_test(small_aoi: shapely.Polygon, default_ors_settings: ORSSettings):
    paths = gpd.GeoDataFrame(data={'test': [0]}, geometry=[small_aoi], crs='EPSG:4326')
    result = get_detour_factors(small_aoi, paths, default_ors_settings, profile='foot-walking')
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


def test_create_waypoint_path():
    received = create_waypoint_path(chunk_coordinates=[HEX1, HEX2])

    expected = [
        HEX1['center'],
        HEX1['corners'][0],
        HEX1['corners'][1],
        HEX1['center'],
        HEX1['corners'][2],
        HEX1['corners'][3],
        HEX1['center'],
        HEX1['corners'][4],
        HEX1['corners'][5],
        HEX1['center'],
        HEX2['center'],
        HEX2['corners'][0],
        HEX2['corners'][1],
        HEX2['center'],
        HEX2['corners'][2],
        HEX2['corners'][3],
        HEX2['center'],
        HEX2['corners'][4],
        HEX2['corners'][5],
        HEX2['center'],
    ]
    assert received == expected


def test_ors_request(responses_mock: RequestsMock, default_ors_settings: ORSSettings):
    a = [0.0, 0.0]
    midpoint_ab = [0.0, 0.5]
    b = [0.0, 1.0]
    midpoint_bc1 = [0.3, 1.0]
    midpoint_bc2 = [0.6, 1.0]
    c = [1.0, 1.0]
    responses_mock.post(
        'http://localhost:8080/ors/v2/directions/foot-walking/geojson',
        json={
            'features': [
                {
                    'geometry': {'coordinates': [a, midpoint_ab, b, midpoint_bc1, midpoint_bc2, c, a]},
                    'properties': {
                        'way_points': [0, 2, 5, 6],
                        'segments': [{'distance': 1}, {'distance': 2}, {'distance': 3}],
                    },
                }
            ]
        },
    )

    path_response = ors_request(ors_settings=default_ors_settings, profile='foot-walking', coordinates=[a, b, c, a])

    assert path_response.coordinates == [a, b, c, a]
    assert path_response.distances == [1, 2, 3]


def test_ors_request_route_not_found_first_link(responses_mock: RequestsMock, default_ors_settings: ORSSettings):
    a = [0.0, 0.0]
    b = [0.0, 1.0]
    c = [1.0, 1.0]

    responses_mock.post(
        'http://localhost:8080/ors/v2/directions/foot-walking/geojson',
        match=[matchers.json_params_matcher({'coordinates': [a, b, c, a]})],
        status=404,
        json={
            'error': {
                'code': 2009,
                'message': 'Route could not be found - Unable to find a route between points 0 (0.0, 0.0) and 1 (0.0, 1.0).',
            },
        },
    )
    responses_mock.post(
        'http://localhost:8080/ors/v2/directions/foot-walking/geojson',
        match=[matchers.json_params_matcher({'coordinates': [b, c, a]})],
        json={
            'features': [
                {
                    'geometry': {'coordinates': [b, c, a]},
                    'properties': {
                        'way_points': [0, 1, 2],
                        'segments': [{'distance': 2}, {'distance': 3}],
                    },
                }
            ]
        },
    )

    path_response = ors_request(ors_settings=default_ors_settings, profile='foot-walking', coordinates=[a, b, c, a])

    assert path_response.coordinates == [None, b, c, a]
    assert path_response.distances == [np.inf, 2, 3]


def test_ors_request_route_not_found_middle_link(responses_mock: RequestsMock, default_ors_settings: ORSSettings):
    a = [0.0, 0.0]
    b = [0.0, 1.0]
    c = [1.0, 1.0]

    responses_mock.post(
        'http://localhost:8080/ors/v2/directions/foot-walking/geojson',
        match=[matchers.json_params_matcher({'coordinates': [a, b, c, a]})],
        status=404,
        json={
            'error': {
                'code': 2009,
                'message': 'Route could not be found - Unable to find a route between points 1 (0.0, 1.0) and 2 (1.0, 1.0).',
            },
        },
    )
    responses_mock.post(
        'http://localhost:8080/ors/v2/directions/foot-walking/geojson',
        match=[matchers.json_params_matcher({'coordinates': [a, b]})],
        json={
            'features': [
                {
                    'geometry': {'coordinates': [a, b]},
                    'properties': {
                        'way_points': [0, 1],
                        'segments': [{'distance': 1}],
                    },
                }
            ]
        },
    )
    responses_mock.post(
        'http://localhost:8080/ors/v2/directions/foot-walking/geojson',
        match=[matchers.json_params_matcher({'coordinates': [c, a]})],
        json={
            'features': [
                {
                    'geometry': {'coordinates': [c, a]},
                    'properties': {
                        'way_points': [0, 1],
                        'segments': [{'distance': 3}],
                    },
                }
            ]
        },
    )

    path_response = ors_request(ors_settings=default_ors_settings, profile='foot-walking', coordinates=[a, b, c, a])

    assert path_response.coordinates == [a, b, c, a]
    assert path_response.distances == [1, np.inf, 3]


def test_ors_request_route_not_found_last_link(responses_mock: RequestsMock, default_ors_settings: ORSSettings):
    a = [0.0, 0.0]
    b = [0.0, 1.0]
    c = [1.0, 1.0]

    responses_mock.post(
        'http://localhost:8080/ors/v2/directions/foot-walking/geojson',
        match=[matchers.json_params_matcher({'coordinates': [a, b, c, a]})],
        status=404,
        json={
            'error': {
                'code': 2009,
                'message': 'Route could not be found - Unable to find a route between points 2 (1.0, 1.0) and 3 (0.0, 0.0).',
            },
        },
    )
    responses_mock.post(
        'http://localhost:8080/ors/v2/directions/foot-walking/geojson',
        match=[matchers.json_params_matcher({'coordinates': [a, b, c]})],
        json={
            'features': [
                {
                    'geometry': {'coordinates': [a, b, c]},
                    'properties': {
                        'way_points': [0, 1, 2],
                        'segments': [{'distance': 1}, {'distance': 2}],
                    },
                }
            ]
        },
    )

    path_response = ors_request(ors_settings=default_ors_settings, profile='foot-walking', coordinates=[a, b, c, a])

    assert path_response.coordinates == [a, b, c, None]
    assert path_response.distances == [1, 2, np.inf]


def test_extract_data_from_ors_result():
    expected = [
        {
            'distances': [110, 120, 130, 140, 150, 160],
            'snapped_coordinates': HEX1,
        },
        {
            'distances': [210, 220, 230, 240, 250, 260],
            'snapped_coordinates': HEX2,
        },
    ]

    path_with_distances = [
        [HEX1['center'], 110],
        [HEX1['corners'][0], 999],
        [HEX1['corners'][1], 120],
        [HEX1['center'], 130],
        [HEX1['corners'][2], 999],
        [HEX1['corners'][3], 140],
        [HEX1['center'], 150],
        [HEX1['corners'][4], 999],
        [HEX1['corners'][5], 160],
        [HEX1['center'], 999],
        [HEX2['center'], 210],
        [HEX2['corners'][0], 999],
        [HEX2['corners'][1], 220],
        [HEX2['center'], 230],
        [HEX2['corners'][2], 999],
        [HEX2['corners'][3], 240],
        [HEX2['center'], 250],
        [HEX2['corners'][4], 999],
        [HEX2['corners'][5], 260],
        [HEX2['center'], None],
    ]
    coordinates, distances = list(zip(*path_with_distances))
    recieved = extract_data_from_ors_result(
        _PathResponse(coordinates=list(coordinates), distances=list(distances[:-1]))
    )

    assert expected == recieved


def test_extract_data_from_ors_result_with_unroutable_links():
    # TODO this needs more thought about special cases
    expected = [
        {
            'distances': [110, 120, 130, 140, 150, 160],
            'snapped_coordinates': HEX1,
        },
        {
            'distances': [210, 220, np.inf, 240, 250, np.inf],
            'snapped_coordinates': {
                'center': HEX2['center'],
                'corners': [HEX2['corners'][0], HEX2['corners'][1], None, HEX2['corners'][3], HEX2['corners'][4], None],
            },
        },
    ]

    path_with_distances = [
        [HEX1['center'], 110],
        [HEX1['corners'][0], 999],
        [HEX1['corners'][1], 120],
        [HEX1['center'], 130],
        [HEX1['corners'][2], 999],
        [HEX1['corners'][3], 140],
        [HEX1['center'], 150],
        [HEX1['corners'][4], 999],
        [HEX1['corners'][5], 160],
        [HEX1['center'], 999],
        [HEX2['center'], 210],
        [HEX2['corners'][0], 999],
        [HEX2['corners'][1], 220],
        [HEX2['center'], np.inf],  # next node is unroutable
        [None, np.inf],  # unroutable node
        [HEX2['corners'][3], 240],
        [HEX2['center'], 250],
        [HEX2['corners'][4], np.inf],  # next node is unroutable
        [None, np.inf],  # unroutable node
        [None, np.inf],  # last node is unroutable because previous node was unroutable
    ]
    coordinates, distances = list(zip(*path_with_distances))
    recieved = extract_data_from_ors_result(
        _PathResponse(coordinates=list(coordinates), distances=list(distances[:-1]))
    )

    assert expected == recieved


def test_extract_data_from_ors_result_point_snapped_to_center():
    path_response = _PathResponse(
        coordinates=[
            HEX1['center'],
            HEX1['center'],  # simulating that the first corner snapped to the same point as the center
            HEX1['corners'][1],
            HEX1['center'],
            HEX1['corners'][2],
            HEX1['corners'][3],
            HEX1['center'],
            HEX1['corners'][4],
            HEX1['corners'][5],
            HEX1['center'],
        ],
        distances=[0, 120, 120, 130, 999, 140, 150, 999, 160],
    )

    expected = [
        {
            'distances': [120, 130, 140, 150, 160],
            'snapped_coordinates': {
                'center': HEX1['center'],
                'corners': [
                    HEX1['corners'][1],
                    HEX1['corners'][2],
                    HEX1['corners'][3],
                    HEX1['corners'][4],
                    HEX1['corners'][5],
                ],
            },
        },
    ]

    recieved = extract_data_from_ors_result(path_response)

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


def test_calculate_detour_factors_with_unroutable_corner():
    chunk_distances = [
        {
            'distances': [110, 120, 130, 140, 150, 160],
            'snapped_coordinates': HEX1,
        },
        {
            'distances': [210, 220, np.inf, 240, 250, np.inf],
            'snapped_coordinates': {
                'center': HEX2['center'],
                'corners': [HEX2['corners'][0], HEX2['corners'][1], None, HEX2['corners'][3], HEX2['corners'][4], None],
            },
        },
    ]

    result = calculate_detour_factors(
        chunk_distances, transform=Transformer.from_crs(crs_from='EPSG:4326', crs_to='EPSG:25832')
    )

    expected_result = [np.float64(1.35), np.inf]

    np.testing.assert_array_almost_equal(result, expected_result, decimal=2)


def test_exclude_ferries():
    # TODO rewrite test so it's a bit more useful and has some cells that pass
    snapped_input = pd.DataFrame(
        data={'snapped_location': [[8.773085, 49.376161], None], 'snapped_distance': [122.49, None]},
        index=['8a1faad69927fff', '8a1faad6992ffff'],
    ).rename_axis('id')

    paths = gpd.GeoDataFrame(geometry=[shapely.LineString([(0, 0), (1, 1)])])

    result = exclude_ferries(snapped_input, paths)

    verify(result.to_json())
