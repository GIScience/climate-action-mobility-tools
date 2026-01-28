from time import time_ns

import shapely
from detour_factors_batched import get_detour_factors_batched
from matplotlib import pyplot as plt
from ors_settings import ORSSettings

from mobility_tools.detour_factors import get_detour_factors

aoi = shapely.box(8.671217, 49.408404, 8.7000658, 49.420400)
# aoi = shapely.MultiPolygon(
#     polygons=[
#         [
#             [
#                 [12.29, 48.20],
#                 [12.29, 48.34],
#                 [12.48, 48.34],
#                 [12.48, 48.20],
#                 [12.29, 48.20],
#             ]
#         ]  # type: ignore
#     ]
# )
ors_settings = ORSSettings()

print('Running detour factor calculations...')

start = time_ns()
detour_factors_batched = get_detour_factors_batched(aoi=aoi, paths=None, ors_settings=ors_settings, profile='foot-walking')
end = time_ns()
print(f'Batched: {(end - start) / 10**9} s')

start = time_ns()
detour_factors_old = get_detour_factors(aoi=aoi, paths=None, ors_settings=ors_settings, profile='foot-walking')
end = time_ns()
print(f'Old: {(end - start) / 10**9} s')

fig, axes = plt.subplots(1, 2)
detour_factors_old.plot(column='detour_factor', ax=axes[0])
detour_factors_batched.plot(column='detour_factor', ax=axes[1])
fig.savefig('test.png')
