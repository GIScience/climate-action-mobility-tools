import logging

import geopandas as gpd
import pandas as pd
from requests.adapters import HTTPAdapter
from requests_ratelimiter import LimiterSession
from urllib3.util import Retry

from mobility_tools.ors_settings import ORSSettings

log = logging.getLogger(__name__)


def snap_batched_records(
    ors_settings: ORSSettings,
    batched_locations: list[gpd.GeoSeries],
    profile: str,
    snapping_radius: int = 150,
) -> pd.DataFrame:
    log.debug('Snapping Destinations')
    # snapping unfortunately does not have a wrapper in openrouteservice-py
    headers = {
        'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
        'Authorization': ors_settings.client._key,
        'Content-Type': 'application/json; charset=utf-8',
    }

    retries = Retry(
        total=3,
        backoff_factor=0.1,
        status_forcelist=[502, 503, 504],
        allowed_methods={'POST'},
    )

    request_session = LimiterSession(per_minute=ors_settings.ors_snapping_rate_limit)

    request_session.mount('https://', HTTPAdapter(max_retries=retries))
    request_session.mount('http://', HTTPAdapter(max_retries=retries))

    snapped_df_list = []
    for i, batch in enumerate(batched_locations):
        locations = [[round(x, 5), round(y, 5)] for x, y in zip(batch.x, batch.y)]
        body = {'locations': locations, 'radius': snapping_radius}

        call = request_session.post(f'{ors_settings.client._base_url}/v2/snap/{profile}', json=body, headers=headers)

        call.raise_for_status()
        json_result = call.json()

        snapping_response = pd.Series(index=batch.index, data=json_result['locations'])
        snapped_response_extracted = snapping_response.apply(extract_ors_snapping_results)
        snapped_df = pd.DataFrame()
        snapped_df['snapping_results'] = snapped_response_extracted
        snapped_df['snapped_location'], snapped_df['snapped_distance'] = zip(*snapped_df.snapping_results)

        snapped_df_list.append(snapped_df)

    return pd.concat(snapped_df_list).drop(columns=['snapping_results'])


def extract_ors_snapping_results(result: None | dict) -> tuple[None, None] | tuple[list[float], float]:
    if result is None:
        return None, None
    else:
        snapped_distance = result.get('snapped_distance')
        if snapped_distance is None:
            snapped_distance = 0
        return result['location'], snapped_distance
