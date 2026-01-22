import logging

import geopandas as gpd
import h3pandas
import numpy as np
import openrouteservice.directions as directions
import pandas as pd
import shapely

from mobility_tools.ors_settings import ORSSettings

log = logging.getLogger(__name__)


def get_detour_factors_simplified(
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

    full_hexgrid['detour_factor'] = hexgrid.apply(
        extract_routing_coordinates, axis=1, ors_settings=ors_settings, profile=profile
    )

    return full_hexgrid


def extract_routing_coordinates(row: pd.Series, ors_settings: ORSSettings, profile: str) -> float:
    vertices = extract_points(row)

    skip_segments = range(2, len(vertices), 2)
    result = directions.directions(
        client=ors_settings.client,
        coordinates=vertices,
        profile=profile,
        geometry=False,
        skip_segments=list(skip_segments),
    )

    detour_factor = calculate_detour_factor(result)

    return detour_factor


def calculate_detour_factor(result):
    segments = result['routes'][0]['segments']
    distances = [segment['distance'] for segment in segments]
    indices = list(range(0, len(distances), 2))
    actual_distances = [distances[i] for i in indices]
    linear_distances = [distances[i + 1] for i in indices]

    for index, linear_distance in enumerate(linear_distances):
        if linear_distance == 0.0:
            linear_distances.pop(index)
            actual_distances.pop(index)

    detour_ratio = np.array(actual_distances) / np.array(linear_distances)

    detour_factor = detour_ratio.mean()
    return detour_factor


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
