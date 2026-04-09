import asyncio
import logging
from collections import defaultdict
from collections.abc import Buffer
from io import BytesIO

import numpy as np
import obstore as obs
from obstore.store import S3Store
from PIL import Image
from pmtiles.tile import Entry, find_tile, zxy_to_tileid
from tqdm import tqdm

from mobility_tools.settings import S3Settings
from mobility_tools.slope.pmtiles_utils import (
    BoundingBox,
    ExtendedPMTilesReader,
    TileCoordinate,
    TileKey,
    get_smoothed_elevation,
    rgb_to_elevation,
)
from mobility_tools.utils import LonLat
from mobility_tools.utils.exceptions import NoTileDataError, TileNotFoundError

log = logging.getLogger(__name__)


def get_point_elevations(
    s3settings: S3Settings,
    points: np.ndarray,
) -> np.ndarray:
    """
    Input a series of (lon, lat) points (CRS=WGS84,4326), and return their smoothed elevations.
    :param s3settings: settings of the s3 bucket
    :param points: 2d-array, each row is (lon, lat)
    :param is_smooth: bool, whether to smooth the elevation
    :return:
        point_elevations: 1d-array, the smoothed/raw elevations of input points.
    """
    # get groups at zoom level 6
    tile_xys_l6 = match_points_to_tiles(points, zoom=6)
    tile_names_l6 = get_l6_source(s3settings, tile_xys_l6)

    point_smooth_elevations_w_ids = []
    log.debug('Processing groups of points by PMTiles at zoom level 6')
    for tile_key, tilename_l6 in tqdm(tile_names_l6.items()):
        # pick up points belonging to the same pmtile file at zoom level 6
        group_point_ids = tile_xys_l6[tile_key]  # e.g. [0,2,3,4]
        group_points = points[group_point_ids]

        # match points to corresponding entries with highest zoom level
        subtiles_xy = asyncio.run(match_points_to_entries(group_points, tilename_l6, s3settings))

        # for every sub-group, get their corresponding smaller tiles
        log.debug('Processing groups of points by PMTiles at HIGHER zoom levels')
        for subtile_key, subgroup_point_ids in subtiles_xy.items():
            tile_object_name = tilename_l6 if subtile_key.zoom > 12 else s3settings.obs_planet_source

            data, tile_bounds = asyncio.run(
                load_sub_pmtiles(
                    s3settings.s3store,
                    tile_object_name,
                    maxzoom=subtile_key.zoom,
                    tile_x=subtile_key.tile_x,
                    tile_y=subtile_key.tile_y,
                )
            )

            img = Image.open(BytesIO(data))
            img = img.convert('RGB')
            img = np.asarray(img)  # [height, width, channel]

            # get elev
            elevations = rgb_to_elevation(img)
            for point_id in subgroup_point_ids:
                point_lon, point_lat = group_points[point_id]
                point_smoothed_elevation = get_smoothed_elevation(tile_bounds, elevations, point_lon, point_lat)
                point_smooth_elevations_w_ids.append(
                    [group_point_ids[point_id], point_smoothed_elevation]
                )  # todo: Emily wants to implement it by BTreeMap (in RUST)

    point_smooth_elevations_w_ids = np.asarray(point_smooth_elevations_w_ids)
    point_smooth_elevations = point_smooth_elevations_w_ids[np.argsort(point_smooth_elevations_w_ids[:, 0]), 1]

    return point_smooth_elevations


def match_points_to_tiles(points: np.ndarray, zoom: int, minzoom: int | None = None) -> dict[TileKey, list]:
    """
    Match/group points to corresponding tiles at specified zoom level.
    Returns:
        tiles_xy: dict[TileKey, list].
            key: TileKey including zoom, tile_x, tile_y, and minzoom information.
            value: list of point indices belonging to corresponding tile.
    """
    tiles_xy: dict[TileKey, list[int]] = defaultdict(list)
    log.debug(f'Matching points to tiles at zoom level {zoom}')
    for index, point in enumerate(points):
        tile_zxy = TileCoordinate.from_lon_lat(lon=point[0], lat=point[1], zoom=zoom)
        tile_key = TileKey(zoom=zoom, tile_x=tile_zxy.tile_x, tile_y=tile_zxy.tile_y, minzoom=minzoom)
        tiles_xy[tile_key].append(index)

    return tiles_xy


def get_l6_source(s3settings: S3Settings, tilexys_l6: dict[TileKey, list[int]]) -> dict[TileKey, str]:
    pmtile_sources = {}
    for tile_key in tilexys_l6:
        pmtile_source = get_pmtile_source(s3settings, tile_key.tile_x, tile_key.tile_y, 6)
        pmtile_sources[tile_key] = pmtile_source

    return pmtile_sources


def get_pmtile_source(s3settings: S3Settings, tile_x: int, tile_y: int, zoom: int = 6) -> str:
    top_tile_name = f'6-{tile_x >> (zoom - 6)}-{tile_y >> (zoom - 6)}'
    object_dir = f'mapterhorn/{s3settings.s3_dem_version}'
    object_name = f'{object_dir}/{top_tile_name}.pmtiles'

    try:
        # check whether target high-resolution tile exists
        obs.head(s3settings.s3store, object_name)
        # if exists, return objectname
    except FileNotFoundError:
        log.warning('high-resolution dem is not reachable, will use 30 m planet dem.')
        object_name = s3settings.obs_planet_source

    return object_name


async def match_points_to_entries(
    points: np.ndarray,
    tilename_l6: str,
    s3settings: S3Settings,
) -> dict[TileKey, list]:
    pmtile_src = await ExtendedPMTilesReader.open(tilename_l6, store=s3settings.s3store)

    # get root_entries (all leaf directories)
    minzoom, maxzoom, root_entries = await get_subtile_info(pmtile_src)
    log.debug(f'Matching points to tiles from maxzoom level {maxzoom} to minzoom level {minzoom}')

    initial_tiles_xy: dict[TileKey, list[int]] = defaultdict(list)

    # Step 1: group point indices by their tile at maxzoom
    for index, point in enumerate(points):
        tile_zxy = TileCoordinate.from_lon_lat(lon=point[0], lat=point[1], zoom=maxzoom)
        initial_tiles_xy[TileKey(**tile_zxy.__dict__)].append(index)

    # Step 2: walk zoom levels downward, merging unmatched groups each level
    tiles_xy = await find_entry_from_all_potential_tiles(
        initial_tiles_xy, from_zoom=maxzoom, to_zoom=minzoom, root_entries=root_entries, pmtile_src=pmtile_src
    )

    # Step 3: update the tile info of points at planet.pmtiles
    #   Note: 12 is the maxzoom level of planet.pmtiles
    if 'unassigned_points' in tiles_xy:
        planet_points_idx: list = tiles_xy.pop('unassigned_points')
        planet_tiles_xy = match_points_to_tiles(points[planet_points_idx], zoom=12)

        tiles_xy.update(planet_tiles_xy)  # type: ignore

    return tiles_xy  # type: ignore this will have been converted to a dict with no string entries


async def find_entry_from_all_potential_tiles(
    pending_tiles: dict[TileKey, list],
    from_zoom: int,
    to_zoom: int,
    root_entries: list[Entry],
    pmtile_src: ExtendedPMTilesReader,
) -> dict[TileKey | str, list]:
    result: dict[TileKey | str, list[int]] = defaultdict(list)
    fallback_indices: list[int] = []  # points that never matched

    for temp_zoom in range(from_zoom, to_zoom - 1, -1):
        if not pending_tiles:
            break

        next_pending: dict[TileKey, list[int]] = defaultdict(list)

        # compute tile coords at current z from original maxzoom tile
        for tile, indices in pending_tiles.items():
            tile_id = zxy_to_tileid(tile.zoom, tile.tile_x, tile.tile_y)

            matched_root_entry = find_tile(root_entries, tile_id)
            if matched_root_entry is None:
                raise TileNotFoundError()
            if matched_root_entry.run_length > 0:  # no leaf directory
                tile_exists = tile_id == matched_root_entry.tile_id
            else:  # search leaf dicrectory
                _, leaf_entries_tile_ids = await get_leaf_entries(pmtile_src, query_entry=matched_root_entry)
                tile_exists = tile_id in leaf_entries_tile_ids

            if tile_exists:
                result[tile].extend(indices)
            else:
                if temp_zoom == to_zoom:  # exhausted all zoom levels — falls back to planet.pmtiles
                    fallback_indices.extend(indices)
                else:  # merge into parent tile group for next iteration
                    parent = TileKey(zoom=temp_zoom - 1, tile_x=tile.tile_x >> 1, tile_y=tile.tile_y >> 1)
                    next_pending[parent].extend(indices)

        pending_tiles = next_pending

    if fallback_indices:  # points cannot match any high-resolution pmtiles, need to use planet.pmtiles' elevation data.
        result['unassigned_points'] = fallback_indices

    return result


async def get_subtile_info(src: ExtendedPMTilesReader) -> tuple[int, int, list[Entry]]:
    entries = await src.load_root_entries()
    return src.minzoom, src.maxzoom, entries


async def get_leaf_entries(src: ExtendedPMTilesReader, query_entry: Entry) -> tuple[list[Entry], list[int]]:
    leaf_entries = await src.load_leaf_entries(query_entry)
    leaf_entries_tile_ids = [entry.tile_id for entry in leaf_entries]
    return leaf_entries, leaf_entries_tile_ids


async def load_sub_pmtiles(
    client_store: S3Store,
    object_name: str,
    maxzoom: int | None,
    tile_x: int | None = None,
    tile_y: int | None = None,
    coordinate: LonLat | None = None,
) -> tuple[Buffer, BoundingBox]:
    """
    Load a tile from a PMTiles file, and return the tile data and its geographical bounds.
    Args:
    - client_store: S3Store object for accessing the PMTiles file.
    - object_name: The name of the PMTiles file in the S3 store.
    - maxzoom: The zoom level of the tile to load. If None, it will be determined from the PMTiles file's maxzoom.
    - tile_x, tile_y: The x and y indices of the tile to load. Used when maxzoom is not None.
    - lon, lat: longitude and latitude/ If maxzoom is None,  the corresponding tile indices at maxzoom will be calculated.
    """

    src = await ExtendedPMTilesReader.open(object_name, store=client_store)

    # todo: return dict instead of tuple
    if maxzoom is not None:
        assert tile_x is not None and tile_y is not None, ValueError(
            'When maxzoom is provided, tile_x and tile_y must also be provided'
        )
    else:
        assert coordinate is not None, ValueError(
            'When maxzoom is not provided, lon and lat must be provided to calculate tile indices'
        )

        # if maxzoom is None we get the source from planet tile
        maxzoom = src.maxzoom
        tile_zxy = TileCoordinate.from_lon_lat(lon=coordinate[0], lat=coordinate[1], zoom=maxzoom)
        tile_x, tile_y = tile_zxy.tile_x, tile_zxy.tile_y

    data = await src.get_tile(z=maxzoom, x=tile_x, y=tile_y)
    if data is None:
        raise NoTileDataError(f'No data in Tile at {maxzoom}')

    # Get tile bounds
    lon_min, lat_max = TileCoordinate.to_lon_lat(TileCoordinate(zoom=maxzoom, tile_x=tile_x, tile_y=tile_y))
    lon_max, lat_min = TileCoordinate.to_lon_lat(TileCoordinate(zoom=maxzoom, tile_x=tile_x + 1, tile_y=tile_y + 1))

    tile_bounds = BoundingBox(min_lon=lon_min, min_lat=lat_min, max_lon=lon_max, max_lat=lat_max)

    return data, tile_bounds
