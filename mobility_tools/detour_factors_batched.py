import logging

import geopandas as gpd
import h3pandas
import numpy as np
import openrouteservice.directions as directions
import pandas as pd
import shapely
from pyproj import Transformer
from tqdm import tqdm

from mobility_tools.ors_settings import ORSSettings
from mobility_tools.utils.batching import batching
from mobility_tools.utils.snapping import snap_batched_records

log = logging.getLogger(__name__)


def get_detour_factors_batched(
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

    result = directions.directions(
        client=ors_settings.client,
        coordinates=coordinates,
        profile=profile,
        format='geojson',
    )

    distances = extract_data_from_ors_result(result)

    return distances


def create_waypoint_path(chunk_coordinates: list[dict]) -> list[list[float]]:
    coordinates = []
    for chunk in chunk_coordinates:
        center = chunk['center']
        waypoints = chunk['corners']
        for i in range(len(waypoints) // 2 + 1):
            waypoints.insert(3 * i, center)
        coordinates += waypoints
    return coordinates


def extract_data_from_ors_result(result: dict) -> list[dict]:
    # TODO try out if this needs more error handling
    segment_distances = [segment['distance'] for segment in result['features'][0]['properties']['segments']]
    snapped_coordinates = [
        result['features'][0]['geometry']['coordinates'][i] for i in result['features'][0]['properties']['way_points']
    ]
    coordinates_per_cell = 10
    distances = []
    for cell_offset in range(0, len(snapped_coordinates), coordinates_per_cell):
        cell_coordinates = snapped_coordinates[cell_offset : cell_offset + coordinates_per_cell]
        cell_distances = segment_distances[cell_offset : cell_offset + coordinates_per_cell]

        center_indices = list(range(0, len(cell_coordinates), 3))
        snapped_center = cell_coordinates.pop()
        route_distances = []
        for i in reversed(center_indices[:-1]):
            snapped_center = cell_coordinates.pop(i)
            route_distances[0:0] = [cell_distances[i], cell_distances[i + 2]]

        valid_indices = [i for i, snapped_corner in enumerate(cell_coordinates) if snapped_corner != snapped_center]
        route_distances = [route_distances[i] for i in valid_indices]
        cell_coordinates = [cell_coordinates[i] for i in valid_indices]

        distances.append(
            {
                'distances': route_distances,
                'snapped_coordinates': {'center': snapped_center, 'corners': cell_coordinates},
            }
        )

    return distances


def calculate_detour_factors(chunk_distances: list[dict], transform: Transformer) -> list[float]:
    chunk_detour_factors = []
    for chunk_distance in chunk_distances:
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
