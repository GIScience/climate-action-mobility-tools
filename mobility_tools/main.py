from time import time_ns

import shapely
from detour_factors import get_detour_factors
from detour_factors_new import get_detour_factors_new
from ors_settings import ORSSettings

aoi = shapely.box(8.671217, 49.408404, 8.7000658, 49.420400)
ors_settings = ORSSettings()

start = time_ns()
detour_factors_new = get_detour_factors_new(aoi=aoi, ors_settings=ors_settings, profile='foot-walking')
end = time_ns()
print(f'New: {end - start} ns')

start = time_ns()
detour_factors_old = get_detour_factors(aoi=aoi, ors_settings=ors_settings, profile='foot-walking')
end = time_ns()
print(f'Old: {end - start} ns')

# from detour_factors_new import TIMES
#
# times = pd.DataFrame.from_records(TIMES)
# print(times.sum(axis=0))
# detour_factors_new = get_detour_factors_new(aoi=aoi, ors_settings=ors_settings, profile='foot-walking')


# fig, axes = plt.subplots(1, 2)
# detour_factors_old.plot(column='detour_factor', ax=axes[0])
# detour_factors_new.plot(column='detour_factor', ax=axes[1])
#
# fig.savefig('test.png')
