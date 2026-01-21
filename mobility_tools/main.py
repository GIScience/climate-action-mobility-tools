from time import time_ns

import shapely
from detour_factors_batched import get_detour_factors_batched
from matplotlib import pyplot as plt
from ors_settings import ORSSettings

aoi = shapely.box(8.671217, 49.408404, 8.7000658, 49.420400)
ors_settings = ORSSettings()

print('Running detour factor calculations...')

start = time_ns()
detour_factors_batched = get_detour_factors_batched(aoi=aoi, ors_settings=ors_settings, profile='foot-walking')
end = time_ns()
print(f'Batched: {(end - start) / 10**9} s')

# start = time_ns()
# detour_factors_new = get_detour_factors_new(aoi=aoi, ors_settings=ors_settings, profile='foot-walking')
# end = time_ns()
# print(f'New: {(end - start) / 10**9} s')

# # start = time_ns()
# # detour_factors_simplified = get_detour_factors_simplified(aoi=aoi, ors_settings=ors_settings, profile='foot-walking')
# # end = time_ns()
# # print(f'Simplified: {(end - start) / 10**9} s')

# start = time_ns()
# detour_factors_old = get_detour_factors(aoi=aoi, ors_settings=ors_settings, profile='foot-walking')
# end = time_ns()
# print(f'Old: {(end - start) / 10**9} s')

fig, ax = plt.subplots(1, 1)
# detour_factors_old.plot(column='detour_factor', ax=axes[0])
# detour_factors_simplified.plot(column='detour_factor', ax=axes[1])
detour_factors_batched.plot(column='detour_factor', ax=ax)
# detour_factors_new.plot(column='detour_factor', ax=axes[2])

fig.savefig('test.png')
