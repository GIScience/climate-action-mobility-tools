import logging

import geopandas as gpd
import h3pandas
import numpy as np
import openrouteservice.directions as directions
import pandas as pd
import shapely
from PIL.ImageChops import offset

from mobility_tools.ors_settings import ORSSettings

log = logging.getLogger(__name__)


def calculate_detour_factors(chunk_distances):
    chunk_detour_factors = []
    for actual_distances, linear_distances in chunk_distances:
        for index, linear_distance in enumerate(linear_distances):
            if linear_distance == 0.0:
                linear_distances.pop(index)
                actual_distances.pop(index)

        detour_ratio = np.array(actual_distances) / np.array(linear_distances)
        detour_factor = detour_ratio.mean()
        chunk_detour_factors.append(detour_factor)

    return chunk_detour_factors


def get_detour_factors_batched(
    aoi: shapely.MultiPolygon,
    # paths: gpd.GeoDataFrame,
    ors_settings: ORSSettings,
    profile: str,
    resolution: int = 10,
) -> gpd.GeoDataFrame:
    """
    Get detour factors calculates detour factors for the aoi in a hexgrid.
    :param: aoi: `shapely.MultiPolygon` area to calculate the detour factors for.
    :param: ors_settings: ORSSettings that contain the relevant settings for the ORS.
    :param: profile: Specifies the mode of transport to use when calculating.
        detour factors. One of ["driving-car", "driving-hgv", "foot-walking",
        "foot-hiking", "cycling-regular", "cycling-road",
        "cycling-safe", "cycling-mountain", "cycling-tour",
        "cycling-electric",].
    :param: resolution: int setting the hexgrid resolution. Defaults to 10.
    """
    log.info('Computing detour factors')

    log.debug(f'Using h3pandas v{h3pandas.version} to get hexgrid for aoi.')  # need to use h3pandas import
    full_hexgrid = gpd.GeoDataFrame(geometry=[aoi], crs='EPSG:4326').h3.polyfill_resample(resolution)

    # get cell centers and geometry
    hexgrid = full_hexgrid.h3.h3_to_geo().rename(columns={'geometry': 'cell_center'}).set_geometry('cell_center')
    hexgrid = hexgrid.h3.h3_to_geo_boundary()

    # Process hexgrid in batches to avoid applying row-by-row for the entire frame
    batch_size = 3 #should be set dynamically depending on the size of the hexgrid and ORS rate limits
    detour_factors = []
    for start in range(0, len(hexgrid), batch_size):
        end = min(start + batch_size, len(hexgrid))
        chunk = hexgrid.iloc[start:end]
        log.debug(f'Processing hexgrid batch {start}:{end}')
        chunk_coordinates = chunk.apply(extract_coordinates, axis=1).tolist()
        chunk_distances = compute_distances(chunk_coordinates, ors_settings, profile)
        batch_detour_factors = calculate_detour_factors(chunk_distances)
        detour_factors += batch_detour_factors
    full_hexgrid['detour_factor'] = detour_factors

    return full_hexgrid


def extract_points(row):
    center: shapely.Point = row['cell_center']
    boundary: shapely.Polygon = row['geometry']

    center_lon, center_lat = center.coords.xy
    boundary_lon, boundary_lat = boundary.exterior.coords.xy
    center_point = (center_lon[0], center_lat[0])
    vertices = list(zip(boundary_lon, boundary_lat))
    for i in range(len(vertices)):
        vertices.insert(2 * i, center_point)
    vertices.pop()
    return vertices

def extract_coordinates(row):
    center_point: shapely.Point = row['cell_center']
    boundary: shapely.Polygon = row['geometry']

    center_lon, center_lat = center_point.coords.xy
    boundary_lon, boundary_lat = boundary.exterior.coords.xy
    center = (center_lon[0], center_lat[0])
    corners = list(zip(boundary_lon, boundary_lat))
    corners.pop()

    return (center, corners)

def compute_distances(
        chunk_coordinates, ors_settings: ORSSettings, profile: str
) -> list[(list[float], list[float])]:
    coordinates = []
    skip_segments = []
    segment_indices = []
    for center, vertices in chunk_coordinates:
        if len(coordinates):
            # skip connection from last cell center to this cell center
            skip_segments.append(len(coordinates))
        for i in range(len(vertices) + 1):
            vertices.insert(2 * i, center)
        n = len(vertices)
        offset = len(coordinates)
        linear_indices = [i + offset for i in list(range(2, n, 2))]
        coordinates += vertices
        skip_segments += linear_indices
        segment_indices.append([i - 2 for i in linear_indices])

    result = directions.directions(
        client=ors_settings.client,
        coordinates=coordinates,
        profile=profile,
        geometry=False,
        skip_segments=skip_segments
    )

    segment_distances = [segment['distance'] for segment in result['routes'][0]['segments']]
    distances = []
    for indices in segment_indices:
        routes_distances = [segment_distances[i] for i in indices]
        linear_distances = [segment_distances[i+1] for i in indices]
        distances.append((routes_distances, linear_distances))

    return distances