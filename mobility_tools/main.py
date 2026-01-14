import pandas as pd
import shapely
from detour_factors_new import get_detour_factors_new
from ors_settings import ORSSettings

aoi = shapely.box(8.671217, 49.408404, 8.680658, 49.418400)
ors_settings = ORSSettings()

detour_factors_new = get_detour_factors_new(aoi=aoi, ors_settings=ors_settings, profile='foot-walking')

from detour_factors_new import TIMES

times = pd.DataFrame.from_records(TIMES)
print(times.sum(axis=0))
# detour_factors_new = get_detour_factors_new(aoi=aoi, ors_settings=ors_settings, profile='foot-walking')


# fig, axes = plt.subplots(1, 2)
# detour_factors_old.plot(column='detour_factor', ax=axes[0])
# detour_factors_new.plot(column='detour_factor', ax=axes[1])
#
# fig.savefig('test.png')
