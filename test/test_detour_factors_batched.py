import shapely
from vcr import use_cassette

from mobility_tools.detour_factors_batched import get_detour_factors_batched


@use_cassette
def test_get_detour_factors_batched(default_ors_settings):
    aoi = shapely.box(8.671217, 49.408404, 8.6800658, 49.410400)

    result = get_detour_factors_batched(aoi, default_ors_settings, profile='foot-walking')

    # approval test here is hard because batching happens in non-deterministic order,
    # so recorded request returns static data for varied orders of cells
    assert 'detour_factor' in result.columns
    for detour_factor in result.detour_factor:
        assert isinstance(detour_factor, float)
    assert result.active_geometry_name is not None
