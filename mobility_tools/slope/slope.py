import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS

from mobility_tools.settings import S3Settings
from mobility_tools.slope.pmtiles_io import get_point_elevations
from mobility_tools.slope.utils import paths_lines_to_points, points_to_lines, segmentize_paths


def get_paths_slopes(
    paths: gpd.GeoDataFrame, s3settings: S3Settings, segment_length: float | None = 10.0
) -> gpd.GeoDataFrame:
    """
    Calculate slope for paths segments.
    Args:
    - paths: GeoDataFrame with LineString geometries and '@osmId'
    - s3settings: S3Settings object with S3 connection settings and PMTiles info.
    - segment_length: Non-negative, non-zero length in meters for segmentizing paths. If no segmentizing is desired, supply `None`.
    Returns:
    - GeoDataFrame with LineString segments and their slope values.
        columns = ['@osmId', 'segment_length', 'slope', 'segment_id', 'geometry']
    """
    estimated_utm = paths.estimate_utm_crs()

    # project and re-project paths for segmentizing
    if segment_length is not None:
        if segment_length <= 0:
            raise ValueError('segment_length cannot be smaller or equal than 0')
        paths = segmentize_paths(paths, estimated_utm, segment_length)

    # convert paths to points to get elevations.
    paths_pts_wid = paths_lines_to_points(paths)

    # get elevations for points in PMTiles
    path_smoothed_elevations = get_point_elevations(
        s3settings, points=paths_pts_wid[['x', 'y']].values
    )  # todo: think again input it as pd.Series or array

    # reconstruct segments and calculate slope
    paths_pts_elevations = pd.DataFrame(
        {
            '@osmId': paths_pts_wid['id'],
            'smoothed_elevation': path_smoothed_elevations,
            'x': paths_pts_wid['x'],
            'y': paths_pts_wid['y'],
        }
    )
    paths_segments = get_segments_slopes_from_points(paths_pts_elevations, estimated_utm)

    return paths_segments


def get_segments_slopes_from_points(paths_pts_elevations: pd.DataFrame, estimated_utm: CRS) -> gpd.GeoDataFrame:
    """
    re-group path points to path segments, to calculate their slopes
    :param paths_pts_elevations: DataFrame.columns = ['@osmId', 'smoothed_elevation', 'x', 'y']
    :param estimated_utm: projected crs to calculate segment length correctly (unit: meter)
    :return:
        GeoDataFrame with LineString segments and their slope values.
    """
    paths_segments = []
    for path_id, path_pts_info in paths_pts_elevations.groupby('@osmId'):
        # Create LineString segments and calculate lengths
        segments = points_to_lines(path_pts_info[['x', 'y']].values, estimated_utm, crs='EPSG:4326', length=True)
        segments.insert(0, '@osmId', path_id)
        # Calculate slope
        segments['slope'] = calc_slope(path_pts_info['smoothed_elevation'].values, segments['segment_length'].values)

        segments = segments[segments['segment_length'] > 0].reset_index(drop=True)
        segments['segment_id'] = segments.index.values
        paths_segments.append(segments)

    paths_segments = pd.concat(paths_segments, ignore_index=True)

    return paths_segments


def calc_slope(path_elev: list | np.ndarray, segment_lengths: np.ndarray) -> np.ndarray:
    if isinstance(path_elev, list):
        path_elev = np.asarray(path_elev)

    segment_slopes = 100 * np.divide(
        (path_elev[1:] - path_elev[:-1]),
        segment_lengths,
        out=np.full(segment_lengths.shape, fill_value=np.nan, dtype=float),
        where=segment_lengths != 0,
    )

    return segment_slopes
