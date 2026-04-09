import logging
import re
from dataclasses import dataclass

import geopandas as gpd
import h3pandas
import numpy as np
import openrouteservice.directions as directions
import openrouteservice.exceptions as ors_exceptions
import pandas as pd
import shapely
from pyproj import Transformer
from tqdm import tqdm

from mobility_tools.detour_factors.batching import batching
from mobility_tools.detour_factors.snapping import snap_batched_records
from mobility_tools.settings import ORSSettings
from mobility_tools.utils import LonLat

log = logging.getLogger(__name__)


@dataclass
class _PathResponse:
    """Coordinates for waypoints along a path and the distances between each sequential pair of coordinates
    (len(coordinates)==len(distances)+1). A coordinate value of `None` represents an unroutable point.
    """

    coordinates: list[LonLat | None]
    distances: list[float]


def get_detour_factors(
    aoi: shapely.MultiPolygon,
    paths: gpd.GeoDataFrame,
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
    full_hexgrid = (
        gpd.GeoDataFrame(geometry=[aoi], crs='EPSG:4326').h3.polyfill_resample(resolution).sort_values(by='h3_polyfill')
    )

    crs = full_hexgrid.estimate_utm_crs()
    transform = Transformer.from_crs(crs_from='EPSG:4326', crs_to=crs)

    # get cell centers and geometry
    hexgrid = full_hexgrid.h3.h3_to_geo().rename(columns={'geometry': 'cell_center'}).set_geometry('cell_center')
    hexgrid = hexgrid.h3.h3_to_geo_boundary()

    hexgrid['snapped'] = snap_centers(hexgrid['cell_center'], ors_settings=ors_settings, profile=profile)
    hexgrid = hexgrid[~hexgrid['snapped'].isna()]
    hexgrid['snapped_centers'] = hexgrid['snapped'].apply(lambda x: shapely.Point(x[0], x[1]))
    hexgrid = hexgrid.drop(columns=['cell_center', 'snapped']).rename(columns={'snapped_centers': 'cell_center'})
    hexgrid = exclude_ferries(hexgrid, paths)

    hexgrid['coordinates'] = hexgrid.apply(extract_coordinates, axis=1)

    # Process hexgrid in batches to avoid applying row-by-row for the entire frame
    batch_size = 7  # should be set dynamically depending on the size of the hexgrid and ORS rate limits
    detour_factors = []

    for start in tqdm(range(0, len(hexgrid), batch_size), desc='Processing batches'):
        end = min(start + batch_size, len(hexgrid))
        chunk = hexgrid.iloc[start:end]
        log.debug(f'Processing hexgrid batch {start}:{end}')
        chunk_coordinates = chunk['coordinates'].to_list()
        chunk_distances = compute_distances(chunk_coordinates, ors_settings, profile)
        batch_detour_factors = calculate_detour_factors(chunk_distances, transform)
        detour_factors += batch_detour_factors
    hexgrid['detour_factor'] = detour_factors

    return hexgrid


def snap_centers(centers: gpd.GeoSeries, ors_settings: ORSSettings, profile: str):
    batched_centers = batching(series=centers, batch_size=ors_settings.ors_snapping_request_size_limit)

    # snapping radius is set based on cell resolution of 10
    snapped_records = snap_batched_records(ors_settings, batched_centers, profile=profile, snapping_radius=70)

    return snapped_records['snapped_location']


def extract_coordinates(row: pd.Series) -> dict[str, tuple | list[tuple]]:
    center_point: shapely.Point = row['cell_center']
    boundary: shapely.Polygon = row['geometry']

    center_lon, center_lat = center_point.coords.xy
    boundary_lon, boundary_lat = boundary.exterior.coords.xy
    center = (center_lon[0], center_lat[0])
    corners = list(zip(boundary_lon, boundary_lat))
    corners.pop()

    return {'center': center, 'corners': corners}


def compute_distances(
    chunk_coordinates: list[dict], ors_settings: ORSSettings, profile: str
) -> list[dict[str, list[float] | dict]]:
    coordinates = create_waypoint_path(chunk_coordinates)

    path_response = ors_request(
        coordinates=coordinates,
        profile=profile,
        ors_settings=ors_settings,
    )

    distances = extract_data_from_ors_result(path_response)

    return distances


def create_waypoint_path(chunk_coordinates: list[dict]) -> list[LonLat]:
    coordinates = []
    for chunk in chunk_coordinates:
        center = chunk['center']
        waypoints = chunk['corners'].copy()
        for i in range(len(waypoints) // 2 + 1):
            waypoints.insert(3 * i, center)
        coordinates += waypoints
    return coordinates


def ors_request(coordinates: list[LonLat], profile: str, ors_settings: ORSSettings) -> _PathResponse:
    """
    Request distances between sequential pairs of coordinates from ORS. If a point is disconnected in the network, set
    the corresponding coordinates to `None` and the distances to `np.inf`.

    :param coordinates: list of Coordinates (in EPSG:4326) to route between.
    :param profile: string as described in get_detour_factors.
    :param ors_settings: initialised ORSSettings instance.
    :return: _PathResponse containing the snapped coordinates and distances between them. Coordinates that could not be
    snapped or routed return as None. Distances for which no route could be found return as numpy.inf.
    """
    try:
        result = directions.directions(
            client=ors_settings.client, coordinates=coordinates, profile=profile, format='geojson'
        )

        snapped_coordinates = [
            result['features'][0]['geometry']['coordinates'][i]
            for i in result['features'][0]['properties']['way_points']
        ]
        segment_distances = [segment['distance'] for segment in result['features'][0]['properties']['segments']]

    except ors_exceptions.ApiError as e:
        error_message = (e.message or {}).get('error', {})
        message_text = error_message.get('message')

        if error_message.get('code') == 2009 and message_text:
            log.debug(message_text)

            capture = re.search(
                '^Route could not be found - Unable to find a route between points ([0-9]*) \(.*\) and ([0-9]*) \(.*\)\.$',
                message_text,
            )
            error_start_index = int(capture.group(1))
            error_end_index = int(capture.group(2))

            route_to_error_start = coordinates[: error_start_index + 1]
            if len(route_to_error_start) > 1:
                path_response = ors_request(
                    ors_settings=ors_settings, profile=profile, coordinates=route_to_error_start
                )
                coordinates_to_error = path_response.coordinates
                distances_to_error = path_response.distances
            else:
                distances_to_error = []
                coordinates_to_error: list[LonLat | None] = [None]

            route_from_error_end = coordinates[error_end_index:]
            if len(route_from_error_end) > 1:
                path_response = ors_request(
                    ors_settings=ors_settings, profile=profile, coordinates=route_from_error_end
                )
                coordinates_from_error = path_response.coordinates
                distances_from_error = path_response.distances
            else:
                distances_from_error = []
                coordinates_from_error: list[LonLat | None] = [None]

            segment_distances = distances_to_error + [np.inf] + distances_from_error
            snapped_coordinates = coordinates_to_error + coordinates_from_error

        else:
            raise e

    return _PathResponse(coordinates=snapped_coordinates, distances=segment_distances)


def extract_data_from_ors_result(path_response: _PathResponse) -> list[dict]:
    """
    Decompose the ORS routing response back into results by hex-cell and extract the snapped corners and distances
    that were routable (i.e. exclude corner points that were snapped to the center point or that are part of a
    disconnected network).

    :param path_response: an ORS routing response containing point coordinates and the distances between them
    :return: a dictionary in the following form:
        {
            'snapped_coordinates': {
                'center': Coordinate # the snapped center point
                'corners': list[Coordinate] # the snapped corner points that were routable
            },
            'distances': list[float] # the routed distance to each of the corners
        }
    """
    # TODO try out if this needs more error handling

    coordinates_per_cell = 10
    distances = []
    for cell_offset in range(0, len(path_response.coordinates), coordinates_per_cell):
        cell_coordinates = path_response.coordinates[cell_offset : cell_offset + coordinates_per_cell]
        cell_distances = path_response.distances[cell_offset : cell_offset + coordinates_per_cell]

        snapped_center = cell_coordinates[0]
        center_indices = list(range(0, len(cell_coordinates), 3))
        route_distances = []
        snapped_coordinates = []
        for i in center_indices[:-1]:
            route_distances.extend([cell_distances[i], cell_distances[i + 2]])
            snapped_coordinates.extend(cell_coordinates[i + 1 : i + 3])

        valid_indices = [i for i, dist in enumerate(route_distances) if dist > 0]
        route_distances = [route_distances[i] for i in valid_indices]
        snapped_coordinates = [snapped_coordinates[i] for i in valid_indices]

        distances.append(
            {
                'distances': route_distances,
                'snapped_coordinates': {'center': snapped_center, 'corners': snapped_coordinates},
            }
        )

    return distances


def calculate_detour_factors(chunk_distances: list[dict], transform: Transformer) -> list[float]:
    chunk_detour_factors = []
    for chunk_distance in chunk_distances:
        if np.inf == max(chunk_distance['distances']):
            chunk_detour_factors.append(np.inf)
            continue

        actual_distances = chunk_distance['distances']
        center = chunk_distance['snapped_coordinates']['center']
        waypoints = chunk_distance['snapped_coordinates']['corners']

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


def exclude_ferries(snapped_destinations: pd.DataFrame, paths: gpd.GeoDataFrame) -> pd.DataFrame:
    if paths is None or len(paths) == 0:
        return snapped_destinations
    boundaries = snapped_destinations.h3.h3_to_geo_boundary()
    snapped_destinations['contains_paths'] = boundaries.intersects(paths.union_all())
    snapped_destinations = snapped_destinations[snapped_destinations['contains_paths']]
    return snapped_destinations.drop(columns=['contains_paths'])
