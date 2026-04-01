import numpy as np
import pytest

from mobility_tools.slope.pmtiles_utils import (
    BoundingBox,
    TileCoordinate,
    get_pixel_coordinates,
    get_smoothed_elevation,
    get_tiles_for_bbox,
    rgb_to_elevation,
)


def test_tilecoordinate_from_lon_lat():
    lon, lat = -6.297456, 49.915093
    zoom = 6
    tile_result = TileCoordinate.from_lon_lat(lon, lat, zoom)

    expected_tile_x = 30
    expected_tile_y = 21

    assert isinstance(tile_result, TileCoordinate)
    assert tile_result.tile_x == expected_tile_x
    assert tile_result.tile_y == expected_tile_y


def test_tilecoordinate_to_lon_lat():
    tile_x, tile_y = 30, 21
    zoom = 6
    upperleft_corner_result = TileCoordinate.to_lon_lat(TileCoordinate(zoom, tile_x, tile_y))
    bottomright_corner_result = TileCoordinate.to_lon_lat(TileCoordinate(zoom, tile_x + 1, tile_y + 1))

    expected_lon, expected_lat = -6.297456, 49.915093

    assert upperleft_corner_result[0] <= expected_lon <= bottomright_corner_result[0]
    assert upperleft_corner_result[1] >= expected_lat >= bottomright_corner_result[1]


def test_get_tiles_for_bbox():
    bbox = BoundingBox(min_lon=-11.25, min_lat=48.93, max_lon=-5.62, max_lat=52.49)
    zoom = 6

    tiles_xy_result = get_tiles_for_bbox(bbox, zoom)

    expected_tiles = []
    for tile_x in range(30, 31 + 1):
        for tile_y in range(20, 21 + 1):
            expected_tiles.append(TileCoordinate(zoom, tile_x, tile_y))

    assert expected_tiles == tiles_xy_result


def test_get_pixel_coordinates(slope_geo_bbox):
    slope_elev_arr = np.arange(0, 100, 1).reshape((10, 10)) / 10

    img_h, img_w = slope_elev_arr.shape

    fake_lonlats = [
        [slope_geo_bbox.min_lon, slope_geo_bbox.max_lat],  # top-left corner in the img
        [slope_geo_bbox.max_lon, slope_geo_bbox.max_lat],  # top-right corner in the img
        [slope_geo_bbox.min_lon, slope_geo_bbox.min_lat],  # bottom-left corner in the img
        [slope_geo_bbox.max_lon, slope_geo_bbox.min_lat],  # bottom-right corner in the img
        [slope_geo_bbox.max_lon - 1e-5, slope_geo_bbox.max_lat],  # close to top-right
        [
            (slope_geo_bbox.min_lon + slope_geo_bbox.max_lon) / 2,
            (slope_geo_bbox.min_lat + slope_geo_bbox.max_lat) / 2,
        ],  # center in the img
    ]
    expected_pixel_xys = [
        [0, 0],
        [9, 0],
        [0, 9],
        [9, 9],
        [9, 0],
        [5, 5],  # note: here it's 5,5, as we discussed yesterday for the margin value
    ]

    calculated_pixel_xys = []
    for i in range(len(fake_lonlats)):
        fake_lon, fake_lat = fake_lonlats[i]
        pixel_x, pixel_y = get_pixel_coordinates(
            bbox=slope_geo_bbox, img_h=img_h, img_w=img_w, lon=fake_lon, lat=fake_lat
        )
        calculated_pixel_xys.append([pixel_x, pixel_y])

    assert calculated_pixel_xys == expected_pixel_xys


def test_rgb_to_elevation():
    rgb_img = np.array([[[128, 0, 0], [128, 9, 230]]])
    elevations_result = rgb_to_elevation(rgb_img)

    expected_elevations = [0, 9.9]

    assert np.allclose(elevations_result.flatten(), expected_elevations, 1e-2)


@pytest.fixture
def fake_elevation_array():
    return np.array([[0, 0, 0, 1, 1], [0, 0, 0, 1, 1], [0, 0, 0, 1, 1], [2, 2, 2, 3, 3], [2, 2, 2, 3, 3]])


VALIDATION_OBJECT_SMOOTHED_ELEVATIONS = {
    'top_left_corner': {'lonlat': [[0.0, 15.0], [0.5, 14.5], [1.0, 14.0]], 'expected': [0, 0, 0]},
    'top_right_corner': {'lonlat': [[5.0, 15.0], [4.5, 14.5], [4.0, 14.0]], 'expected': [1, 1, 1]},
    'bottom_left_corner': {'lonlat': [[0.0, 10.0], [0.5, 10.5], [1.0, 11.0]], 'expected': [2, 2, 2]},
    'bottom_right_corner': {'lonlat': [[5.0, 10.0], [4.5, 10.5], [4.0, 11.0]], 'expected': [3, 3, 3]},
    'top_row': {
        'lonlat': [
            [0.5, 15.0],
            [1.0, 15.0],
            [1.5, 15.0],
            [2.0, 15.0],
            [2.5, 15.0],
            [3.0, 15.0],
            [3.5, 15.0],
            [4.0, 15.0],
            [4.5, 15.0],
        ],
        'expected': [0, 0, 0, 0, 0, 0.5, 1.0, 1.0, 1.0],
    },
    'left_column': {
        'lonlat': [
            [0.0, 14.5],
            [0.0, 14.0],
            [0.0, 13.5],
            [0.0, 13.0],
            [0.0, 12.5],
            [0.0, 12.0],
            [0.0, 11.5],
            [0.0, 11.0],
            [0.0, 10.5],
        ],
        'expected': [0, 0, 0, 0, 0, 1.0, 2.0, 2.0, 2.0],
    },
    'middle_column': {
        'lonlat': [
            [3.0, 14.5],
            [3.0, 14.0],
            [3.0, 13.5],
            [3.0, 13.0],
            [3.0, 12.5],
            [3.0, 12.0],
            [3.0, 11.5],
            [3.0, 11.0],
            [3.0, 10.5],
            [3.0, 10.0],
        ],
        'expected': [0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 2.5, 2.5, 2.5, 2.5],
    },
    'center': {
        'lonlat': [[2.50, 12.50], [2.25, 12.75], [2.75, 12.75], [2.25, 12.25], [2.75, 12.25]],
        'expected': [0.0, 0.0, 0.25, 0.50, 0.75],
    },
}


@pytest.mark.parametrize(
    'coordinate_groups',
    VALIDATION_OBJECT_SMOOTHED_ELEVATIONS.values(),
    ids=VALIDATION_OBJECT_SMOOTHED_ELEVATIONS.keys(),
)
def test_get_smoothed_elevation(slope_geo_bbox, fake_elevation_array, coordinate_groups):
    results = [
        get_smoothed_elevation(slope_geo_bbox, fake_elevation_array, lon, lat)
        for lon, lat in coordinate_groups['lonlat']
    ]
    assert results == coordinate_groups['expected']
