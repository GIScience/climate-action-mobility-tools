import math
from dataclasses import dataclass
from typing import Self, TypeAlias

import numpy as np
from async_pmtiles import PMTilesReader as aiopmReader
from pmtiles.tile import Entry, deserialize_directory

Coordinate: TypeAlias = tuple[float, float]


class ExtendedPMTilesReader(aiopmReader):
    _root_entries: list[Entry] = None
    _leaf_entries_cache: dict[int, list[Entry]] = dict()  # cache leaf entries by their offset for efficiency

    async def _load_directory(self, offset: int, length: int):
        data = await self.store.get_range_async(
            self.path,
            start=offset,
            length=length,
        )
        return deserialize_directory(data)

    async def load_leaf_entries(self, entry: Entry) -> list[Entry]:
        if entry.tile_id in self._leaf_entries_cache:
            return self._leaf_entries_cache[entry.tile_id]

        child_offset = self.header['leaf_directory_offset'] + entry.offset
        child_length = entry.length
        leaf_entries = await self._load_directory(child_offset, child_length)
        self._leaf_entries_cache[entry.tile_id] = leaf_entries

        return leaf_entries

    async def load_root_entries(self) -> list[Entry]:
        """
        Fetch the root directory bytes via obstore and deserialize them
        into a list of EntryV3 using pmtiles' own deserialize_directory().
        """
        if self._root_entries is not None:
            return self._root_entries

        dir_offset = self.header['root_offset']
        dir_length = self.header['root_length']

        # Fetch raw root-directory bytes from the store using obstore.
        # self._store and self._path are set by PMTilesReader.open().
        directory_values = await self.store.get_range_async(
            self.path,
            start=dir_offset,
            length=dir_length,
        )
        self._root_entries = deserialize_directory(directory_values)
        return self._root_entries


@dataclass
class BoundingBox:
    """Geographic bounding box"""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


@dataclass(frozen=True)
class TileCoordinate:
    """Web Mercator tile coordinate system"""

    zoom: int
    tile_x: int
    tile_y: int

    @classmethod
    def from_lon_lat(cls, lon: float, lat: float, zoom: int) -> Self:
        """
        Convert longitude/latitude to tile coordinates

        Args:
            lon: Longitude in degrees
            lat: Latitude in degrees
            zoom: Zoom level

        Returns:
            (zoom, tile_x, tile_y)
        """
        lat_rad = math.radians(lat)
        n = 2.0**zoom

        tile_x = int((lon + 180.0) / 360.0 * n)
        tile_y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)

        return cls(zoom, tile_x, tile_y)

    def to_lon_lat(self) -> Coordinate:
        """
        Convert tile coordinates to longitude/latitude (top-left corner)

        Args:
            tile_x: Tile X coordinate
            tile_y: Tile Y coordinate
            zoom: Zoom level

        Returns:
            (longitude, latitude)
        """
        n = 2.0**self.zoom

        lon = self.tile_x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * self.tile_y / n)))
        lat = math.degrees(lat_rad)

        return lon, lat


@dataclass(frozen=True)
class TileKey(TileCoordinate):
    """
    This dataclass is only used as the dict key for clarifying the tile metadata,
    no method from TileCoordinate should be called.
    """

    minzoom: int | None = None


def get_tiles_for_bbox(bbox: BoundingBox, zoom: int) -> list[TileCoordinate]:
    """
    Get all tiles that intersect with a bounding box

    Args:
        bbox: Bounding box
        zoom: Zoom level

    Returns:
        List of (tile_x, tile_y) tuples
    """
    # Get tile coordinates for corners
    upperleft_tile_xy = TileCoordinate.from_lon_lat(bbox.min_lon, bbox.min_lat, zoom)
    min_tile_x, max_tile_y = upperleft_tile_xy.tile_x, upperleft_tile_xy.tile_y
    bottomright_tile_xy = TileCoordinate.from_lon_lat(bbox.max_lon, bbox.max_lat, zoom)
    max_tile_x, min_tile_y = bottomright_tile_xy.tile_x, bottomright_tile_xy.tile_y

    # Generate all tiles in range
    tiles = []
    for x in range(min_tile_x, max_tile_x + 1):
        for y in range(min_tile_y, max_tile_y + 1):
            tiles.append(TileCoordinate(zoom, x, y))

    return tiles


def get_pixel_coordinates(bbox: BoundingBox, img_w: int, img_h: int, lon: float, lat: float) -> tuple[int, int]:
    """Calculate pixel coordinates for a given longitude/latitude"""
    pixel_width = (bbox.max_lon - bbox.min_lon) / img_w
    pixel_height = (bbox.max_lat - bbox.min_lat) / img_h

    pixel_x = min(int((lon - bbox.min_lon) / pixel_width), img_w - 1)
    pixel_y = min(int((bbox.max_lat - lat) / pixel_height), img_h - 1)
    return pixel_x, pixel_y


def rgb_to_elevation(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return (r * 256 + g + b / 256) - 32768


def get_point_elevation(bbox: BoundingBox, elev: np.ndarray, lon: float, lat: float) -> float:
    img_h, img_w = elev.shape
    pixel_x, pixel_y = get_pixel_coordinates(bbox, img_w, img_h, lon=lon, lat=lat)

    return elev[pixel_y, pixel_x]


def get_smoothed_elevation(bbox: BoundingBox, elev: np.ndarray, lon: float, lat: float) -> float:
    """Calculate elevation using bilinear interpolation over the 2x2 neighborhood."""
    img_h, img_w = elev.shape[:2]

    pixel_width = (bbox.max_lon - bbox.min_lon) / img_w
    pixel_height = (bbox.max_lat - bbox.min_lat) / img_h

    # Continuous pixel coordinates (0..w, 0..h), origin at top-left
    x = (lon - bbox.min_lon) / pixel_width
    y = (bbox.max_lat - lat) / pixel_height

    # Clamp to valid continuous range so x1/y1 are in-bounds
    x = float(np.clip(x - 0.5, 0.0, img_w - 1.0))
    y = float(np.clip(y - 0.5, 0.0, img_h - 1.0))

    # Identify the 4 surrounding integer pixel coordinates
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, img_w - 1)
    y1 = min(y0 + 1, img_h - 1)

    # Calculate fractional weights
    dx = x - x0
    dy = y - y0

    # Get the values at these 4 points
    q00 = float(elev[y0, x0])
    q10 = float(elev[y0, x1])
    q01 = float(elev[y1, x0])
    q11 = float(elev[y1, x1])

    # 4. Interpolate across x, then y
    top = q00 * (1.0 - dx) + q10 * dx
    bottom = q01 * (1.0 - dx) + q11 * dx

    return top * (1.0 - dy) + bottom * dy
