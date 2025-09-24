import logging
import shapely
import math
import pandas as pd
import geopandas as gpd
import numpy as np
from ors_settings import ORSSettings

log = logging.getLogger(__name__)


def get_detour_factors(
    aoi: shapely.MultiPolygon, ors_settings: ORSSettings, profile: str
) -> gpd.GeoDataFrame:
    log.info("Computing detour factors")

    destinations = create_destinations(
        aoi, max_waypoint_number=ors_settings.ors_directions_waypoint_limit
    )
    distance_between_cells = get_cell_distance(destinations)

    # This following calculation gives the distance from the cell center point to one of the corners.
    # It's based on the normal distance from the center point of the cell to one of its sides,
    # which is half the distance to its neighbouring cell.
    distance_center_corner = int(
        np.ceil((distance_between_cells / 2) * math.cos(math.pi / 6))
    )

    snapped_destinations = snap_destinations(
        destinations,
        ors_settings=ors_settings,
        snapping_radius=distance_center_corner,
        profile=profile,
    )

    destinations_with_snapping = pd.merge(
        left=destinations,
        right=snapped_destinations,
        how="left",
        left_on="id",
        right_on="id",
    )

    mean_walking_distances = get_ors_walking_distances(
        ors_settings,
        distance_between_cells,
        destinations_with_snapping,
        profile=profile,
    )
    return mean_walking_distances.h3.h3_to_geo_boundary().drop(columns="distance")


def create_destinations(
    aoi: shapely.MultiPolygon, max_waypoint_number: int
) -> gpd.GeoDataFrame:
    pass


def get_cell_distance(destinations: gpd.GeoDataFrame) -> float:
    pass


def snap_destinations(
    destinations: gpd.GeoDataFrame,
    ors_settings: ORSSettings,
    snapping_radius: float,
    profile: str,
) -> gpd.GeoDataFrame:
    pass


def get_ors_walking_distances(
    ors_settings: ORSSettings,
    distance_between_cells: float,
    destinations: gpd.GeoDataFrame,
    profile: str,
) -> gpd.GeoDataFrame:
    pass
