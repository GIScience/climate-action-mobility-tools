import json
import logging

import geopandas as gpd
import h3pandas
import numpy as np
import openrouteservice.directions as directions
import shapely
from pyproj import Transformer

from mobility_tools.ors_settings import ORSSettings

log = logging.getLogger(__name__)


def calculate_detour_factors(chunk_distances, transform: Transformer):
    chunk_detour_factors = []
    for actual_distances, coordinates in chunk_distances:
        center, waypoints = coordinates
        waypoints.insert(0, center)

        utm_lon, utm_lat = transform.transform(
            xx=[location[1] for location in waypoints], yy=[location[0] for location in waypoints]
        )
        utm_points = [shapely.Point(utm_lon[i], utm_lat[i]) for i in range(0, len(utm_lon))]
        source = utm_points.pop(0)

        linear_distances = []
        for destination in utm_points:
            distance = shapely.distance(source, destination)
            linear_distances.append(distance)

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

    crs = full_hexgrid.estimate_utm_crs()
    transform = Transformer.from_crs(crs_from='EPSG:4326', crs_to=crs)

    # get cell centers and geometry
    hexgrid = full_hexgrid.h3.h3_to_geo().rename(columns={'geometry': 'cell_center'}).set_geometry('cell_center')
    hexgrid = hexgrid.h3.h3_to_geo_boundary()

    # Process hexgrid in batches to avoid applying row-by-row for the entire frame
    batch_size = 3  # should be set dynamically depending on the size of the hexgrid and ORS rate limits
    detour_factors = []
    for start in range(0, len(hexgrid), batch_size):
        end = min(start + batch_size, len(hexgrid))
        chunk = hexgrid.iloc[start:end]
        log.debug(f'Processing hexgrid batch {start}:{end}')
        chunk_coordinates = chunk.apply(extract_coordinates, axis=1).tolist()
        chunk_distances = compute_distances(chunk_coordinates, ors_settings, profile)
        batch_detour_factors = calculate_detour_factors(chunk_distances, transform)
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


def compute_distances(chunk_coordinates, ors_settings: ORSSettings, profile: str) -> list[(list[float], list[float])]:
    coordinates = []
    skip_segments = []
    segment_indices = []
    for center, waypoints in chunk_coordinates:
        if len(coordinates):
            # skip connection from last cell center to this cell center
            skip_segments.append(len(coordinates))
        for i in range(len(waypoints) // 2 + 1):
            waypoints.insert(3 * i, center)
        offset = len(coordinates)
        connecting_segments = [i + offset for i in list(range(2, len(waypoints), 3))]
        route_segments = []
        for i in connecting_segments:
            route_segments.append(i - 1)
            route_segments.append(i + 1)
        coordinates += waypoints
        skip_segments += connecting_segments
        segment_indices.append([i - 1 for i in route_segments])

    result = directions.directions(
        client=ors_settings.client,
        coordinates=coordinates,
        profile=profile,
        # geometry=False,
        # skip_segments=skip_segments,
        format='geojson',
    )

    with open('test.json', 'w+') as file:
        json.dump(result, indent=2, fp=file)

    segment_distances = [segment['distance'] for segment in result['features'][0]['properties']['segments']]
    snapped_coordinates = [
        result['features'][0]['geometry']['coordinates'][i] for i in result['features'][0]['properties']['way_points']
    ]

    distances = []
    for indices in segment_indices:
        routes_distances = [segment_distances[i] for i in indices]
        snapped_center = snapped_coordinates[indices[0]]
        corner_indices = [i + indices[0] for i in [1, 2, 4, 5, 7, 8]]
        snapped_corners = [snapped_coordinates[i] for i in corner_indices]
        distances.append((routes_distances, (snapped_center, snapped_corners)))

    return distances
