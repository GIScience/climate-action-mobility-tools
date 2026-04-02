# Assuming these are the exception types your code uses
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import geopandas as gpd
import numpy as np
import pytest
from obstore.store import LocalStore
from PIL import Image
from pmtiles.tile import Entry

from mobility_tools.slope.pmtiles_io import (
    find_entry_from_all_potential_tiles,
    get_l6_source,
    get_pmtile_source,
    get_point_elevations,
    load_sub_pmtiles,
    match_points_to_tiles,
)
from mobility_tools.slope.pmtiles_utils import BoundingBox, TileCoordinate, TileKey


@pytest.fixture
def default_local_store() -> LocalStore:
    store = MagicMock()
    store.s3store = LocalStore(prefix='test/resources/')
    store.obs_planet_source = 'planet.pmtiles'
    return store


@pytest.fixture
def default_local_pmtile_object() -> str:
    return '6-50-33.pmtiles'  # Christmas island, Australia


@pytest.fixture
def default_pathway() -> gpd.GeoDataFrame:
    return gpd.read_file('test/resources/way565149324.geojson')


@pytest.fixture
def default_points(slope_geo_bbox, default_pathway):
    dafault_way_points = default_pathway.geometry.get_coordinates()
    supplement_points = np.array(
        [
            [slope_geo_bbox.min_lon, slope_geo_bbox.min_lat],  # at 6-32-30
            [slope_geo_bbox.max_lon, slope_geo_bbox.max_lat],  # at 6-32-29
            [slope_geo_bbox.min_lon + 0.1, slope_geo_bbox.min_lat + 0.1],  # at 6-32-30
            [-5.69, 52.4],  # at 6-30-21
        ]
    )

    return np.vstack((supplement_points, dafault_way_points))


@pytest.fixture
def default_points_tiles_bounds(default_points):
    tile_xys = [[32, 30], [32, 29], [30, 21]]
    tiles_bounds = {}
    for i in range(len(tile_xys)):
        lon_min, lat_max = TileCoordinate.to_lon_lat(
            TileCoordinate(zoom=6, tile_x=tile_xys[i][0], tile_y=tile_xys[i][1])
        )
        lon_max, lat_min = TileCoordinate.to_lon_lat(
            TileCoordinate(zoom=6, tile_x=tile_xys[i][0] + 1, tile_y=tile_xys[i][1] + 1)
        )

        tile_bounds = BoundingBox(min_lon=lon_min, min_lat=lat_min, max_lon=lon_max, max_lat=lat_max)
        tiles_bounds[f'6-{tile_xys[i][0]}-{tile_xys[i][1]}'] = tile_bounds

    return tiles_bounds


@pytest.fixture
def mock_pmtile_reader(default_local_store):
    fake_root_entries = [
        Entry(tile_id=79385941, offset=0, length=9993, run_length=0),
        Entry(tile_id=79390041, offset=9993, length=10170, run_length=0),
        Entry(tile_id=79394137, offset=9993, length=10170, run_length=0),
        Entry(tile_id=317543771, offset=100, length=200, run_length=0),  # z14 of first entry
    ]

    mock_reader = AsyncMock()
    mock_reader.load_leaf_entries = AsyncMock()

    mock_reader.minzoom = 6
    mock_reader.maxzoom = 16
    mock_reader._root_entries = fake_root_entries
    mock_reader._load_leaf_entries = {
        79385941: [
            Entry(tile_id=79385941, offset=0, length=9993, run_length=1),
            Entry(tile_id=79385942, offset=0, length=9993, run_length=1),
        ],
        79390041: [
            Entry(tile_id=79390041, offset=9993, length=10170, run_length=1),
            Entry(tile_id=79390042, offset=9993, length=10170, run_length=1),
        ],
        79394137: [
            Entry(tile_id=79394137, offset=9993, length=10170, run_length=1),
            Entry(tile_id=79394138, offset=9993, length=10170, run_length=1),
        ],
        317543771: [
            Entry(tile_id=317543771, offset=0, length=9993, run_length=1),
        ],
    }

    async def fake_load_leaf_entries(entry: Entry):
        # Look up by tile_id in the dict
        result = mock_reader._load_leaf_entries.get(entry.tile_id, [])
        return result

    # Attach the fake async function
    mock_reader.load_leaf_entries = AsyncMock(side_effect=fake_load_leaf_entries)

    return mock_reader


@pytest.fixture
def default_points_tilexy_l6():
    return {
        TileKey(zoom=6, tile_x=32, tile_y=30): [0, 2],
        TileKey(zoom=6, tile_x=32, tile_y=29): [1],
        TileKey(zoom=6, tile_x=30, tile_y=21): [3],
        TileKey(zoom=6, tile_x=50, tile_y=33): list(np.arange(0, 23, 1) + 4),  # 23=point number of default_pathway
    }


@pytest.fixture
def mock_get_pmtile_source(monkeypatch):
    def mock_object_name(s3settings, x, y, z):
        return f'{z}-{x}-{y}.pmtiles'

    monkeypatch.setattr('mobility_tools.slope.pmtiles_io.get_pmtile_source', mock_object_name)


@pytest.fixture
def mock_values_for_get_grouped_points_elevations(
    monkeypatch, default_rgb_img, default_points_tilexy_l6, default_local_pmtile_object, mock_get_pmtile_source
):
    rgb_img_2x2 = np.concatenate([default_rgb_img, default_rgb_img[:, ::-1, :]], axis=0).astype(np.uint8)

    def match_points_to_entries(points, tilename_l6, s3settings):
        return {TileKey(zoom=12, tile_x=0, tile_y=0): range(len(points))}

    def mock_rgb_img(file_pointer):
        return Image.fromarray(rgb_img_2x2)

    monkeypatch.setattr('mobility_tools.slope.pmtiles_io.match_points_to_entries', match_points_to_entries)
    monkeypatch.setattr('PIL.Image.open', mock_rgb_img)


def test_get_pmtile_source_success(default_s3_settings):
    # Coordinates for zoom 8: 6- (8 >> 2) - (12 >> 2) = 6-2-3
    tile_x, tile_y, zoom = 30, 21, 6
    expected_name = f'mapterhorn/0.0.7/{zoom}-{tile_x}-{tile_y}.pmtiles'

    # Mock obs.head to do nothing (simulating file exists)
    with patch('mobility_tools.slope.pmtiles_io.obs.head') as mock_head:
        result = get_pmtile_source(default_s3_settings, tile_x, tile_y, zoom)

        assert result == expected_name
        mock_head.assert_called_once_with(default_s3_settings.s3store, expected_name)


def test_get_pmtile_source_fallback(default_s3_settings):
    # Mock obs.head to raise a FileNotFoundError wrapped in S3Error
    with patch('mobility_tools.slope.pmtiles_io.obs.head') as mock_head:
        # Simulate the specific error logic in your try-except block
        mock_head.side_effect = FileNotFoundError('File not found')

        result = get_pmtile_source(default_s3_settings, 29, 20, 6)

        assert result == default_s3_settings.obs_planet_source


def test_load_sub_pmtiles_tilexy(default_local_store, default_local_pmtile_object):
    tile_x, tile_y, zoom = 6501, 4334, 13

    data, tile_bounds = asyncio.run(
        load_sub_pmtiles(
            default_local_store.s3store, default_local_pmtile_object, maxzoom=zoom, tile_x=tile_x, tile_y=tile_y
        )
    )

    upper_right_coords = [tile_bounds.max_lon, tile_bounds.max_lat]
    expected_coords = [105.7324218, -10.4013775]

    assert len(data) == 66442
    assert np.allclose(upper_right_coords, expected_coords, rtol=1e-6)


def test_load_sub_pmtiles_lonlat(default_local_store, default_local_pmtile_object):
    lonlat = (105.72948, -10.40439)  # first point of default_pathway

    data, tile_bounds = asyncio.run(
        load_sub_pmtiles(
            default_local_store.s3store,
            default_local_pmtile_object,
            maxzoom=None,
            lonlat=lonlat,
        )
    )

    upper_right_coords = [tile_bounds.max_lon, tile_bounds.max_lat]
    expected_coords = [105.7324218, -10.4013775]

    assert len(data) == 52
    assert np.allclose(upper_right_coords, expected_coords, rtol=1e-6)


def test_match_points_to_tiles(default_points, default_points_tilexy_l6):
    pois_tiles_xy = match_points_to_tiles(default_points, zoom=6)

    assert pois_tiles_xy == default_points_tilexy_l6


def test_find_entry_from_all_potential_tiles(mock_pmtile_reader):
    pending_tile_ids = [
        TileKey(zoom=zoom, tile_x=x, tile_y=y)
        for zoom, x, y in [
            [15, 17404, 11258],  # zoom 13-79385942 (1), with efficient leaf entry
            [15, 17404, 11256],  # zoom 14-317543771 (4), with efficient leaf entry
            [15, 17404, 10992],  # zoom 13-79390042 (2), with efficient leaf entry,
            [15, 17144, 10992],  # zoom 13-79394138 (3), no efficient leaf entry
        ]
    ]  # corresponds to 6-33-21

    input_pending_tiles = {
        pending_tile_ids[0]: [0, 2],
        pending_tile_ids[1]: [3, 4],
        pending_tile_ids[2]: [1],
        pending_tile_ids[3]: [5],
    }

    result_tile_info = find_entry_from_all_potential_tiles(
        input_pending_tiles,
        from_zoom=15,
        to_zoom=13,
        root_entries=mock_pmtile_reader._root_entries,
        pmtile_src=mock_pmtile_reader,
    )

    expected_tile_info = {
        TileKey(zoom=14, tile_x=8702, tile_y=5628): [3, 4],
        TileKey(zoom=13, tile_x=4351, tile_y=2814): [0, 2],
        TileKey(zoom=13, tile_x=4351, tile_y=2748): [1],
        'unassigned_points': [5],
    }

    assert result_tile_info == expected_tile_info


def test_get_l6_source(default_s3_settings, default_points_tilexy_l6, mock_get_pmtile_source):
    pmtile_sources_result = get_l6_source(default_s3_settings, default_points_tilexy_l6)

    expected_sources = {
        TileKey(zoom=6, tile_x=32, tile_y=30): '6-32-30.pmtiles',
        TileKey(zoom=6, tile_x=32, tile_y=29): '6-32-29.pmtiles',
        TileKey(zoom=6, tile_x=30, tile_y=21): '6-30-21.pmtiles',
        TileKey(zoom=6, tile_x=50, tile_y=33): '6-50-33.pmtiles',
    }

    assert pmtile_sources_result == expected_sources


def test_get_point_elevations_tile_exist(
    default_local_store,
    default_points,
    default_points_tiles_bounds,
    mock_get_pmtile_source,
    monkeypatch,
):
    add_points_christmas_island = [[105.6893, -10.42792]]
    points_christmas_island = np.vstack([default_points[4:], add_points_christmas_island])
    smooth_elevations_result = get_point_elevations(
        default_local_store, points_christmas_island
    )  # points from default_pathway

    expected_start_end_elevations = [155.2, 182.8, 278.55]  # Ground truth.

    assert np.allclose(smooth_elevations_result[[0, -2, -1]], expected_start_end_elevations, 1e-1)


def test_get_point_elevations_tile_not_exist(
    default_local_store,
    default_points,
    default_points_tiles_bounds,
    mock_values_for_get_grouped_points_elevations,
    monkeypatch,
):
    async def mock_load_sub_pmtiles_w_nodata(client_store, object_name, maxzoom, tile_x=None, tile_y=None, lonlat=None):
        if 'planet' in object_name:
            bbox = BoundingBox(min_lon=-5, min_lat=10, max_lon=5, max_lat=52.5)
            return b'mocked_tile_data', bbox
        else:
            return None, default_points_tiles_bounds[object_name.split('.')[0]]

    monkeypatch.setattr('mobility_tools.slope.pmtiles_io.load_sub_pmtiles', mock_load_sub_pmtiles_w_nodata)

    smth_elevs_result = get_point_elevations(default_local_store, default_points[:4])

    expected_smth_elevs = [4.95, 0, 4.75, 0]

    assert np.allclose(smth_elevs_result, expected_smth_elevs, 1e-2)
