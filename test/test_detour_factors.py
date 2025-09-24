import json
from functools import partial
from unittest.mock import patch

import geopandas as gpd
import h3
import h3pandas
import numpy as np
import pandas as pd
import pytest
import responses
import responses.matchers
import shapely
from approvaltests import DiffReporter, set_default_reporter, verify
from geopandas.testing import assert_geodataframe_equal
from openrouteservice.exceptions import ApiError
from pandas.testing import assert_frame_equal, assert_series_equal
from requests.exceptions import HTTPError, RetryError
from vcr import use_cassette

from mobility_tools.detour_factors import (
    batch_and_filter_spurs,
    batching,
    check_aoi_contains_cell,
    create_destinations,
    generate_waypoint_pairs,
    get_cell_distance,
    get_detour_factors,
    get_i_or_j_spurs,
    get_ij_spurs,
    get_ors_walking_distances,
    match_ors_distance_to_cells,
    ors_request,
    snap_batched_records,
    snap_destinations,
)
from mobility_tools.ors_settings import ORSSettings


@pytest.fixture(autouse=True)
def configure_approvaltests():
    set_default_reporter(DiffReporter())


def test_get_detour_factors(
    small_aoi,
    expected_detour_factors,
    snapping_response,
    default_ors_settings,
    ors_directions_responses,
):
    assert h3pandas.version is not None

    def request_handling(request, directions: list[dict]) -> tuple[int, dict, str] | Exception:
        payload: dict[str, list] = json.loads(request.body)
        request_length = len(payload['coordinates'])

        status_code: int = 404
        body: dict | None = None
        for index, response in enumerate(directions):
            if len(response['routes'][0]['segments']) == (request_length - 1):
                body = response
                status_code = 200
                break

        if body is None:
            body = {'error': 'not_found'}

        headers = {'request_id': '0'}

        return (status_code, headers, json.dumps(body))

    request_call_back = partial(request_handling, directions=ors_directions_responses['responses'])

    with responses.RequestsMock() as rsps:
        rsps.add(method='POST', url='http://localhost:8080/ors/v2/snap/foot-walking', json=snapping_response)

        rsps.add_callback(
            method='POST',
            url='http://localhost:8080/ors/v2/directions/foot-walking/json',
            callback=request_call_back,
        )

        result = get_detour_factors(aoi=small_aoi, ors_settings=default_ors_settings, profile='foot-walking')

        assert_geodataframe_equal(
            result.drop(columns='detour_factor').sort_index(),
            expected_detour_factors.drop(columns='detour_factor').sort_index(),
        )
        assert_frame_equal(
            result.drop(columns='geometry').sort_index(),
            expected_detour_factors.drop(columns='geometry').sort_index(),
            check_exact=False,
            atol=0.1,
            rtol=0.1,
        )


def test_create_destinations(default_aoi):
    result: pd.DataFrame = create_destinations(default_aoi)

    hexgrid_size = gpd.GeoDataFrame(geometry=[default_aoi], crs='EPSG:4326').h3.polyfill(10).shape[0]

    assert result.shape[0] > 3 * hexgrid_size

    length_counts = result.groupby(by='spur_id').count()

    length_under_2 = length_counts[length_counts['id'] < 2]
    assert length_under_2.empty

    length_over_50 = length_counts[length_counts['id'] > 50]
    assert length_over_50.empty

    for spur_id in set(result['spur_id']):
        spur: pd.DataFrame = result[result['spur_id'] == spur_id]
        id_sublist = spur['id'].to_list()
        for index, _ in enumerate(id_sublist):
            if index >= len(id_sublist) - 1:
                break
            first_cell = np.array(h3.cell_to_local_ij(origin=id_sublist[0], h=id_sublist[index]))
            second_cell = np.array(h3.cell_to_local_ij(origin=id_sublist[0], h=id_sublist[index + 1]))
            difference = second_cell - first_cell
            assert difference[0] in [0, 1]
            assert difference[1] in [0, 1]
            assert np.any(difference)


@pytest.fixture
def default_hexgrid(default_aoi) -> gpd.GeoDataFrame:
    hexgrid = gpd.GeoDataFrame(geometry=[default_aoi], crs='EPSG:4326').h3.polyfill_resample(10).reset_index()
    origin_id = hexgrid.loc[0, 'h3_polyfill']
    hexgrid['local_ij'] = hexgrid['h3_polyfill'].apply(lambda cell_id: h3.cell_to_local_ij(origin=origin_id, h=cell_id))
    hexgrid['local_i'] = hexgrid['local_ij'].apply(lambda ij: ij[0])
    hexgrid['local_j'] = hexgrid['local_ij'].apply(lambda ij: ij[1])
    return hexgrid


@pytest.fixture
def default_origin_id(default_hexgrid):
    return default_hexgrid.loc[0, 'h3_polyfill']


def test_get_i_or_j_spurs(default_aoi, default_hexgrid, default_origin_id):
    min_ij: tuple[int, int] = (default_hexgrid.local_i.min(), default_hexgrid.local_j.min())
    max_ij = (default_hexgrid.local_i.max(), default_hexgrid.local_j.max())

    result_i: pd.DataFrame = get_i_or_j_spurs(
        aoi=default_aoi,
        origin_id=default_origin_id,
        min_ij=min_ij,
        max_ij=max_ij,
        current_value=min_ij[1],
        current_direction='i',
    )
    result_j: pd.DataFrame = get_i_or_j_spurs(
        aoi=default_aoi,
        origin_id=default_origin_id,
        min_ij=min_ij,
        max_ij=max_ij,
        current_value=min_ij[0],
        current_direction='j',
    )

    for spur_id in result_i['spur_id']:
        spur = result_i[result_i['spur_id'] == spur_id]
        id_sublist = spur['id'].to_list()
        for index, _ in enumerate(id_sublist):
            if index >= len(id_sublist) - 1:
                break

            first_cell = np.array(h3.cell_to_local_ij(origin=id_sublist[0], h=id_sublist[index]))
            second_cell = np.array(h3.cell_to_local_ij(origin=id_sublist[0], h=id_sublist[index + 1]))
            difference = second_cell - first_cell
            assert difference[0] == 1
            assert difference[1] == 0

    for spur_id in result_j['spur_id']:
        spur = result_j[result_j['spur_id'] == spur_id]
        id_sublist = spur['id'].to_list()
        for index, _ in enumerate(id_sublist):
            if index >= len(id_sublist) - 1:
                break

            first_cell = np.array(h3.cell_to_local_ij(origin=id_sublist[0], h=id_sublist[index]))
            second_cell = np.array(h3.cell_to_local_ij(origin=id_sublist[0], h=id_sublist[index + 1]))
            difference = second_cell - first_cell
            assert difference[0] == 0
            assert difference[1] == 1


def test_get_ij_spurs(default_aoi, default_hexgrid, default_origin_id):
    max_ij: tuple[int, int] = (default_hexgrid.local_i.max(), default_hexgrid.local_j.max())
    min_ij: tuple[int, int] = (default_hexgrid.local_i.min(), default_hexgrid.local_j.min())

    result: pd.DataFrame = get_ij_spurs(
        aoi=default_aoi, hexgrid=default_hexgrid, origin_id=default_origin_id, min_ij=min_ij, max_ij=max_ij
    )

    for spur_id in result['spur_id']:
        spur = result[result['spur_id'] == spur_id]
        id_sublist = spur['id'].to_list()
        for index, _ in enumerate(id_sublist):
            if index >= len(id_sublist) - 1:
                break

            first_cell = np.array(h3.cell_to_local_ij(origin=id_sublist[0], h=id_sublist[index]))
            second_cell = np.array(h3.cell_to_local_ij(origin=id_sublist[0], h=id_sublist[index + 1]))
            difference = second_cell - first_cell
            assert difference[0] == 1
            assert difference[1] == 1


def test_ij_spurs_order(small_aoi):
    hexgrid = gpd.GeoDataFrame(geometry=[small_aoi], crs='EPSG:4326').h3.polyfill_resample(10).reset_index()
    origin_id = hexgrid.loc[0, 'h3_polyfill']
    hexgrid['local_ij'] = hexgrid['h3_polyfill'].apply(lambda cell_id: h3.cell_to_local_ij(origin=origin_id, h=cell_id))
    hexgrid['local_i'] = hexgrid['local_ij'].apply(lambda ij: ij[0])
    hexgrid['local_j'] = hexgrid['local_ij'].apply(lambda ij: ij[1])
    result: pd.DataFrame = get_ij_spurs(
        small_aoi,
        hexgrid,
        origin_id,
        max_ij=(hexgrid.local_i.max(), hexgrid.local_j.max()),
        min_ij=(hexgrid.local_i.min(), hexgrid.local_j.min()),
    )

    verify(result.to_json(indent=4))


def test_check_aoi_contains_cell(default_aoi):
    h3_cell_in_default_aoi = '8a1f8d2f492ffff'
    h3_cell_not_in_default_aoi = '8a1faad6992ffff'

    assert check_aoi_contains_cell(aoi=default_aoi, cell_id=h3_cell_in_default_aoi)
    assert not check_aoi_contains_cell(aoi=default_aoi, cell_id=h3_cell_not_in_default_aoi)


def test_batch_spurs():
    spurs = pd.DataFrame(
        data={
            'id': ['a', 'b', 'c', 'a', 'b', 'a', 'b', 'c', 'd', 'e', 'a', 'b', 'c', 'd', 'a'],
            'spur_id': ['a', 'a', 'a', 'b', 'b', 'c', 'c', 'c', 'c', 'c', 'd', 'd', 'd', 'd', 'e'],
            'ordinal': [0, 1, 2, 0, 1, 0, 1, 2, 3, 4, 0, 1, 2, 3, 0],
        }
    )

    result = batch_and_filter_spurs(spurs, max_waypoint_number=3)

    expected_result = pd.DataFrame(
        data={
            'id': ['a', 'b', 'c', 'a', 'b', 'a', 'b', 'c', 'c', 'd', 'e', 'a', 'b', 'c', 'c', 'd'],
            'spur_id': [
                'a',
                'a',
                'a',
                'b',
                'b',
                'c:0',
                'c:0',
                'c:0',
                'c:1',
                'c:1',
                'c:1',
                'd:0',
                'd:0',
                'd:0',
                'd:1',
                'd:1',
            ],
            'ordinal': [0, 1, 2, 0, 1, 0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3],
        }
    )

    assert_frame_equal(result, expected_result)


def test_get_cell_distance():
    destinations = pd.DataFrame(
        data={
            'id': ['8a1faa99684ffff', '8a1faa99685ffff', '8a1faa9968effff', '8a1faa9968c7fff'],
            'spur_id': ['ij:2:0', 'ij:2:0', 'ij:2:1', 'ij:2:1'],
            'ordinal': [0, 1, 0, 1],
        }
    )
    result = get_cell_distance(destinations)
    assert pytest.approx(result, abs=1.0) == 131.75


@pytest.fixture
def small_ors_snapping_response():
    return [
        {'locations': [None]},
        {'locations': [{'location': [8.773085, 49.376161], 'name': 'Schulstraße', 'snapped_distance': 114.44}]},
    ]


@use_cassette
def test_snap_destinations(default_ors_settings):
    destinations = pd.DataFrame(
        data={
            'id': ['8a1faad6992ffff', '8a1faad69927fff', '8a1faad69927fff', '8a1faad6992ffff'],
            'spur_id': ['ij:1', 'ij:1', 'ij:2', 'ij:2'],
            'ordinal': [0, 1, 0, 1],
        }
    )

    expected_results = pd.DataFrame(
        data={'snapped_location': [[8.773085, 49.376161], None], 'snapped_distance': [122.49, None]},
        index=['8a1faad69927fff', '8a1faad6992ffff'],
    ).rename_axis('id')

    settings = default_ors_settings.model_copy(deep=True)
    settings.ors_snapping_request_size_limit = 1

    results = snap_destinations(destinations, ors_settings=settings, profile='foot-walking')

    assert_frame_equal(results, expected_results)


def test_snap_batched_records(small_ors_snapping_response, default_ors_settings):
    locations = [
        gpd.GeoSeries(
            index=['8a1faad6992ffff', '8a1faad69927fff'],
            data=[
                shapely.Point(8.774708093757534, 49.37706987059154),
                shapely.Point(8.772978584588666, 49.377259809601995),
            ],
            crs='EPSG:4326',
        ).rename_axis('id'),
    ]

    expected_result = pd.DataFrame(
        data={'snapped_location': [None, [8.773085, 49.376161]], 'snapped_distance': [None, 114.44]},
        index=['8a1faad6992ffff', '8a1faad69927fff'],
    ).rename_axis('id')

    with responses.RequestsMock() as rsps:
        rsps.add(
            method='POST',
            url='http://localhost:8080/ors/v2/snap/foot-walking',
            json={'locations': [None, small_ors_snapping_response[1]['locations'][0]]},
        )

        result = snap_batched_records(
            ors_settings=default_ors_settings, batched_locations=locations, profile='foot-walking'
        )

        assert_frame_equal(result, expected_result)


@use_cassette
def test_snapping_request_fail_bad_gateway(default_ors_settings):
    coordinates = [gpd.GeoSeries([shapely.Point(1.0, 1.0), shapely.Point(1.1, 1.1)], crs='EPSG:4326')]
    with pytest.raises(RetryError):
        snap_batched_records(default_ors_settings, batched_locations=coordinates, profile='foot-walking')


@use_cassette
def test_snapping_request_fail_forbidden(default_ors_settings):
    coordinates = [gpd.GeoSeries([shapely.Point(1.0, 1.0), shapely.Point(1.1, 1.1)], crs='EPSG:4326')]
    with pytest.raises(HTTPError):
        snap_batched_records(default_ors_settings, batched_locations=coordinates, profile='foot-walking')


def test_batching():
    origins = pd.Series([i for i in range(0, 6)])
    batch_size = 2

    expected_result = [pd.Series([0, 1]), pd.Series([2, 3]), pd.Series([4, 5])]
    result = batching(origins, batch_size)  # type: ignore

    for index, batch in enumerate(result):
        assert len(batch) == batch_size
        assert_series_equal(batch, expected_result[index], check_index=False)


@use_cassette
def test_get_ors_walking_distances(default_ors_settings):
    destinations = pd.DataFrame(
        data={
            'id': ['8a1faad6992ffff', '8a1faad69927fff'],
            'spur_id': ['a', 'a'],
            'ordinal': [0, 1],
            'snapped_location': [[8.773, 49.376], [8.773085, 49.376161]],
            'snapped_distance': [114.44, 114.44],
        },
    )
    result = get_ors_walking_distances(
        ors_settings=default_ors_settings,
        cell_distance=150,
        destinations_with_snapping=destinations,
        profile='foot-walking',
    )

    verify(result.to_json(indent=2))


@pytest.fixture
def ors_directions_request_fail():
    with patch('openrouteservice.directions.directions') as mock:
        mock.side_effect = mock_directions_with_ors_error
        yield mock


def mock_directions_with_ors_error(
    client: ORSSettings, coordinates: list[list[float]], profile: str, geometry: bool
) -> None:
    raise ApiError(status=500)


@pytest.fixture
def mock_sleep():
    with patch('time.sleep') as mock:
        mock.return_value = None


def test_ors_request_fail(ors_directions_request_fail, mock_sleep, default_ors_settings):
    coordinates = [[1.0, 1.0], [1.1, 1.1]]
    with pytest.raises(ApiError):
        ors_request(default_ors_settings, coordinates, profile='foot-walking')

    ors_directions_request_fail.assert_called()
    assert ors_directions_request_fail.call_count == 5


@use_cassette
def test_ors_request(default_ors_settings):
    result, start_time = ors_request(
        ors_settings=default_ors_settings, coordinates=[[8.773, 49.376], [8.773085, 49.376161]], profile='foot-walking'
    )
    assert isinstance(start_time, float)
    verify(result)


def test_match_ors_distance_to_cells():
    spur = pd.DataFrame(
        data={
            'id': ['0', '1'],
            'spur_id': ['a', 'a'],
            'ordinal': [0, 1],
            'snapped_location': [[8.773, 49.376], [8.773085, 49.376161]],
            'snapped_distance': [114.44, 114.44],
        },
    )
    distances: list[float] = [134.2]

    result = match_ors_distance_to_cells(spur, distances)

    verify(result)


def test_match_ors_distance_to_cells_edge_cases():
    spur = pd.DataFrame(
        data={
            'id': ['0', '1', '2'],
            'spur_id': ['a', 'a', 'a'],
            'ordinal': [0, 1, 2],
            'snapped_location': [None, [8.773085, 49.376161], [8.773085, 49.376161]],
            'snapped_distance': [np.nan, 114.44, 20.0],
        },
    )
    distances: list[float] = [134.2]

    result = match_ors_distance_to_cells(spur, distances)

    assert result.shape == (2, 2)

    verify(result)


def test_generate_waypoint_pairs():
    spur = pd.DataFrame(
        data={
            'id': ['0', '1', '2', '3', '4', '5', '6'],
            'spur_id': ['a', 'a', 'a', 'a', 'a', 'a', 'a'],
            'ordinal': [2, 3, 4, 5, 6, 7, 8],
            'snapped_location': [
                None,
                [8.773085, 49.376161],
                [8.773085, 49.376161],
                None,
                None,
                [8.773085, 49.376161],
                None,
            ],
            'snapped_distance': [np.nan, 114.44, 20.0, np.nan, np.nan, 10, np.nan],
        },
    )
    expected_result = [(3, 4), (4, 7)]
    result = generate_waypoint_pairs(spur)

    assert result == expected_result
