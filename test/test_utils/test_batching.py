import pandas as pd
from pandas.testing import assert_series_equal

from mobility_tools.utils.batching import batching


def test_batching():
    origins = pd.Series([i for i in range(0, 6)])
    batch_size = 2

    expected_result = [pd.Series([0, 1]), pd.Series([2, 3]), pd.Series([4, 5])]
    result = batching(origins, batch_size)  # type: ignore

    for index, batch in enumerate(result):
        assert len(batch) == batch_size
        assert_series_equal(batch, expected_result[index], check_index=False)
