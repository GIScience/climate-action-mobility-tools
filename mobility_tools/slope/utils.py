import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import CRS


def segmentize_paths(paths: gpd.GeoDataFrame, estimated_utm: CRS, segment_length: float = 10.0) -> gpd.GeoDataFrame:
    paths = paths.to_crs(estimated_utm)
    paths['geometry'] = paths.segmentize(segment_length)
    paths = paths.to_crs('EPSG:4326')

    return paths


def paths_lines_to_points(paths: gpd.GeoDataFrame) -> pd.DataFrame:
    path_points = []
    for _, path_row in paths.iterrows():
        x, y = path_row.geometry.xy
        path_points.append(pd.DataFrame({'id': path_row['@osmId'], 'x': x, 'y': y}))

    return pd.concat(path_points, ignore_index=True)


def points_to_lines(
    points: np.ndarray[float], estimated_utm: CRS, crs: str = 'EPSG:4326', length: bool = True
) -> gpd.GeoDataFrame:
    segments = np.hstack([points[:-1], points[1:]]).reshape(-1, 2, 2)
    segments = shapely.linestrings(segments)
    segments = gpd.GeoDataFrame(geometry=segments, crs=crs)

    if length:
        segments['segment_length'] = segments.to_crs(estimated_utm).length

    return segments
