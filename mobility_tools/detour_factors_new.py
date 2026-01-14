import logging
from time import time_ns

import geopandas as gpd
import h3pandas
import numpy as np
import openrouteservice.directions as ors
import pandas as pd
import shapely
import urllib3
from pyproj import Transformer
from urllib3 import Retry

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
    vertices = extract_points(row, ors_settings=ors_settings, profile=profile)
    end = time_ns()
    extract_time = end - start

    start = time_ns()
    result = ors_request(ors_settings, profile, vertices)
    end = time_ns()
    request_time = end - start

    start = time_ns()
    detour_factor = calculate_detour_factor(transform, result, vertices)
    end = time_ns()
    result_time = end - start

    TIMES.append({'extraction': extract_time, 'request': request_time, 'result': result_time})
    return detour_factor


def ors_request(ors_settings: ORSSettings, profile: str, vertices: list[tuple]) -> dict:
    return ors.directions(
        client=ors_settings.client,
        coordinates=vertices,
        profile=profile,
        geometry=False,
        options={'avoid_features': ['ferries']},
    )


def calculate_detour_factor(transform: Transformer, result: dict, waypoints: list[tuple[float, float]]) -> float:
    segments = result['routes'][0]['segments']

    center_segments = [segments[0], segments[2], segments[3], segments[5], segments[6], segments[8]]
    distances = [segment.get('distance') for segment in center_segments]

    # remove all but first instance of center point as source
    waypoints.pop(9)
    waypoints.pop(6)
    waypoints.pop(3)

    utm_lon, utm_lat = transform.transform(
        xx=[location[0] for location in waypoints], yy=[location[1] for location in waypoints]
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


def extract_points(row: pd.Series, ors_settings: ORSSettings, profile: str):
    center: shapely.Point = row['cell_center']
    boundary: shapely.Polygon = row['geometry']

    center_lon, center_lat = center.coords.xy
    boundary_lon, boundary_lat = boundary.exterior.coords.xy
    center_point = (center_lon[0], center_lat[0])
    vertices = list(zip(boundary_lon, boundary_lat))

    vertices.pop()
    vertices.append(center_point)

    # snapped_vertices = snap_vertices(vertices, ors_settings=ors_settings, profile=profile)

    # #remove last point from linear ring because it closes the ring
    snapped_vertices = vertices
    snapped_center = snapped_vertices.pop()

    # insert center point every 3 positions so the center point is adjacent to each point
    snapped_vertices.insert(0, snapped_center)
    snapped_vertices.insert(3, snapped_center)
    snapped_vertices.insert(6, snapped_center)
    snapped_vertices.insert(9, snapped_center)

    return snapped_vertices


def snap_vertices(
    vertices: list[tuple[float, float]], ors_settings: ORSSettings, profile: str
) -> list[tuple[float, float]]:
    headers = {
        'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
        'Authorization': ors_settings.ors_api_key,
        'Content-Type': 'application/json; charset=utf-8',
    }

    body = {'locations': vertices}

    retries = Retry(
        total=3,
        backoff_factor=0.1,
        status_forcelist=[502, 503, 504],
        allowed_methods={'POST'},
    )

    response = urllib3.request(
        method='post',
        url=f'{ors_settings.client._base_url}/v2/snap/{profile}',
        headers=headers,
        json=body,
        retries=retries,
    )

    result = response.json()
    snapped_vertices = [location.get('location') for location in result['locations']]
    return snapped_vertices
