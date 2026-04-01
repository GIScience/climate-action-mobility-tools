import geopandas as gpd
import shapely
from geopandas.testing import assert_geodataframe_equal
from shapely import LineString

from mobility_tools.slope.utils import segmentize_paths


def test_segmentize_paths(default_test_slope_path):
    # Segmentize the paths with a segment length of 5
    estimated_utm = default_test_slope_path.estimate_utm_crs()
    segmented_gdf_result = segmentize_paths(
        default_test_slope_path, estimated_utm=estimated_utm, segment_length=5
    ).to_crs('EPSG:25829')
    segmented_gdf_result['geometry'] = shapely.set_precision(segmented_gdf_result.geometry.array, grid_size=1e-2)

    # Check that the segments are correctly created
    expected_segments = gpd.GeoDataFrame(
        index=default_test_slope_path.index,
        data=default_test_slope_path,
        geometry=[
            LineString(
                [(0, 0), (5, 0), (10, 0), (10, 5), (10, 10), (15, 10), (20, 10), (20, 15), (20, 20), (20, 25), (20, 30)]
            ),
        ],
        crs='EPSG:25829',
    )

    assert_geodataframe_equal(segmented_gdf_result, expected_segments)
