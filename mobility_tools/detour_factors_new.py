import logging
from time import time_ns

import geopandas as gpd
import h3pandas
import numpy as np
import openrouteservice.distance_matrix as ors
import pandas as pd
import shapely
from pyproj import Transformer

from mobility_tools.ors_settings import ORSSettings

log = logging.getLogger(__name__)

TIMES = []


def get_detour_factors_new(
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

    crs = full_hexgrid.estimate_utm_crs()
    transform = Transformer.from_crs(crs_from='EPSG:4326', crs_to=crs)

    # get cell centers and geometry
    hexgrid = full_hexgrid.h3.h3_to_geo().rename(columns={'geometry': 'cell_center'}).set_geometry('cell_center')
    hexgrid = hexgrid.h3.h3_to_geo_boundary()

    full_hexgrid['detour_factor'] = hexgrid.apply(
        extract_routing_coordinates, axis=1, ors_settings=ors_settings, profile=profile, transform=transform
    )

    return full_hexgrid


def extract_routing_coordinates(
    row: pd.Series, ors_settings: ORSSettings, profile: str, transform: Transformer
) -> float:
    start = time_ns()
    vertices = extract_points(row)
    end = time_ns()
    extract_time = end - start

    start = time_ns()
    result = ors_request(ors_settings, profile, vertices)
    end = time_ns()
    request_time = end - start

    start = time_ns()
    detour_factor = calculate_detour_factor(transform, result)
    end = time_ns()
    result_time = end - start

    TIMES.append({'extraction': extract_time, 'request': request_time, 'result': result_time})
    return detour_factor


def ors_request(ors_settings: ORSSettings, profile: str, vertices: list[tuple]) -> dict:
    return ors.distance_matrix(
        client=ors_settings.client,
        locations=vertices,
        profile=profile,
        sources=[0],
        destinations=list(range(1, len(vertices))),
        metrics=['distance'],
    )


def calculate_detour_factor(transform, result):
    distances = result['distances'][0]
    snapped_vertices = [destination.get('location') for destination in result['destinations']]

    snapped_sources = result['sources'][0].get('location')
    snapped_vertices.insert(0, snapped_sources)

    utm_lon, utm_lat = transform.transform(
        xx=[location[0] for location in snapped_vertices], yy=[location[1] for location in snapped_vertices]
    )

    utm_points = [shapely.Point(utm_lon[i], utm_lat[i]) for i in range(0, len(utm_lon))]
    source = utm_points[0]

    linear_distances = []
    for destination in utm_points[1:]:
        distance = shapely.distance(source, destination)
        linear_distances.append(distance)

    for index, linear_distance in enumerate(linear_distances):
        if linear_distance == 0.0:
            linear_distances.pop(index)
            distances.pop(index)

    detour_ratio = np.array(distances) / np.array(linear_distances)

    detour_factor = detour_ratio.mean()
    return detour_factor


def extract_points(row):
    center: shapely.Point = row['cell_center']
    boundary: shapely.Polygon = row['geometry']

    center_lon, center_lat = center.coords.xy
    boundary_lon, boundary_lat = boundary.exterior.coords.xy
    center_point = (center_lon[0], center_lat[0])
    vertices = list(zip(boundary_lon, boundary_lat))

    vertices.insert(0, center_point)
    vertices.pop()
    return vertices
