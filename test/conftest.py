import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import responses
import shapely
from responses.registries import OrderedRegistry
from shapely import LineString

from mobility_tools.settings import ORSSettings, S3Settings
from mobility_tools.slope.pmtiles_utils import BoundingBox


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
                np.nan,
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
def snapping_response():
    return {
        'locations': [
            {'location': [8.690396, 49.418228], 'snapped_distance': 19.2},
            {'location': [8.690353, 49.417188], 'snapped_distance': 18.76},
            {'location': [8.689776, 49.416313], 'snapped_distance': 29.88},
            {'location': [8.692346, 49.416992], 'snapped_distance': 0.43},
            {'location': [8.693091, 49.418907], 'snapped_distance': 1.28},
            {'location': [8.692439, 49.418159], 'snapped_distance': 43.83},
            None,
        ]
    }


@pytest.fixture
def responses_mock():
    with responses.RequestsMock(registry=OrderedRegistry) as rsps:
        yield rsps


@pytest.fixture
def default_s3_settings() -> S3Settings:
    return S3Settings(
        s3_endpoint='test.s3.endpoint',
        s3_access_key='test',
        s3_secret_key='test-key',
        s3_bucket='test-bucket',
        s3_dem_version='0.0.7',
        s3_default_filename='planet.pmtiles',
    )


@pytest.fixture
def slope_geo_bbox():
    # across tiles (6, 32, 29) and (6, 32, 30)
    return BoundingBox(min_lon=0, min_lat=10, max_lon=5, max_lat=15)


@pytest.fixture
def default_rgb_img():
    # elev=[0, 9.9]
    return np.array([[[128, 0, 0], [128, 9, 230]]])


@pytest.fixture
def default_test_slope_path():
    path = gpd.GeoDataFrame(
        index=[1],
        data={'@osmId': ['way/a']},
        geometry=[
            LineString([[0, 0], [10, 0], [10, 10], [20, 10], [20, 30]]),
        ],
        crs='EPSG:25829',
    ).to_crs('EPSG:4326')

    return path
