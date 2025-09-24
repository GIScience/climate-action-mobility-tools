import json

import pandas as pd
import pytest
import shapely

from mobility_tools.ors_settings import ORSSettings


@pytest.fixture
def default_ors_settings() -> ORSSettings:
    return ORSSettings(ors_base_url='http://localhost:8080/ors', ors_api_key='test-key')


@pytest.fixture
def default_aoi() -> shapely.MultiPolygon:
    return shapely.MultiPolygon(
        polygons=[
            [
                [
                    [12.29, 48.20],
                    [12.29, 48.34],
                    [12.48, 48.34],
                    [12.48, 48.20],
                    [12.29, 48.20],
                ]
            ]  # type: ignore
        ]
    )


@pytest.fixture
def small_aoi() -> shapely.Polygon:
    return shapely.box(8.689282, 49.416193, 8.693186, 49.419123)


@pytest.fixture
def expected_detour_factors() -> pd.DataFrame:
    detour_factors = pd.DataFrame(
        data={
            'detour_factor': [
                1.3995538900828162,
                1.219719961221372,
                1.454343083874761,
                1.7969363677141994,
                1.4832090368368422,
                1.8521635465676833,
                1.3880294081510607,
            ],
            'id': [
                '8a1faa996847fff',
                '8a1faa99684ffff',
                '8a1faa996857fff',
                '8a1faa99685ffff',
                '8a1faa9968c7fff',
                '8a1faa9968effff',
                '8a1faa996bb7fff',
            ],
        }
    ).set_index('id')
    return detour_factors.h3.h3_to_geo_boundary()


@pytest.fixture
def ors_directions_responses() -> dict:
    with open('test/resources/ors_directions_responses.json', 'r') as file:
        return json.load(file)


@pytest.fixture
def snapping_response():
    return {
        'locations': [
            {'location': [8.690396, 49.418228], 'snapped_distance': 19.2},
            {'location': [8.690353, 49.417188], 'snapped_distance': 18.76},
            {'location': [8.689776, 49.416313], 'snapped_distance': 29.88},
            {'location': [8.692346, 49.416992], 'snapped_distance': 0.43},
            {'location': [8.693091, 49.418907], 'snapped_distance': 1.28},
            {'location': [8.692439, 49.418159], 'snapped_distance': 43.83},
            {'location': [8.691352, 49.419144], 'snapped_distance': 6.04},
        ]
    }


@pytest.fixture
def large_mock_ors_directions_api(ordered_responses_mock, ors_directions_responses):
    for response in ors_directions_responses['responses']:
        ordered_responses_mock.add(
            method='POST', url='https://api.openrouteservice.org/v2/directions/foot-walking', json=response
        )
