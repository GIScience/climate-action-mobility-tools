class SizeLimitExceededError(Exception):
    """Exceeded Timeout limits"""

    pass


class TileNotFoundError(Exception):
    """Couldn't Find Matching PMTile"""

    pass


class NoTileDataError(Exception):
    """Matched Tile does not contain elevation data"""
