import geopandas as gpd
import pandas as pd
import pytest
import responses
import shapely
from pandas.testing import assert_frame_equal
from requests.exceptions import HTTPError, RetryError
from vcr import use_cassette

from mobility_tools.utils.snapping import snap_batched_records


@pytest.fixture
def small_ors_snapping_response():
    return [
        {'locations': [None]},
        {'locations': [{'location': [8.773085, 49.376161], 'name': 'Schulstraße', 'snapped_distance': 114.44}]},
    ]


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
