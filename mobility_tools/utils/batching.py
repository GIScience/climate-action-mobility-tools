import geopandas as gpd
import numpy as np


def batching(series: gpd.GeoSeries, batch_size: int) -> list[gpd.GeoSeries]:
    # TODO maybe replace with python 3.13 update to make use of inbuilt batch function
    num_batches = int(np.ceil(len(series) / batch_size))

    batches = []
    for i in range(num_batches):
        start = batch_size * i
        end = start + batch_size
        batches.append(series.iloc[start:end])
    return batches
