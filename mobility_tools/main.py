import matplotlib.pyplot as plt
import shapely
from detour_factors_new import get_detour_factors_new as get_detour_factors
from ors_settings import ORSSettings

aoi = shapely.box(8.671217, 49.408404, 8.680658, 49.418400)
ors_settings = ORSSettings()
detour_factors = get_detour_factors(aoi=aoi, ors_settings=ors_settings, profile='foot-walking')

fig, ax = plt.subplots(1, 1)
detour_factors.plot(column='detour_factor', ax=ax)

fig.savefig('test.png')
