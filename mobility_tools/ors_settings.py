from functools import cached_property
from typing import Optional

import openrouteservice
from pydantic_settings import BaseSettings, SettingsConfigDict


class ORSSettings(BaseSettings):
    ors_base_url: Optional[str] = None
    ors_api_key: Optional[str] = None

    ors_snapping_rate_limit: int = 100
    ors_snapping_request_size_limit: int = 4999

    ors_directions_rate_limit: int = 40
    ors_directions_waypoint_limit: int = 50

    ors_isochrone_max_request_number: int = 500
    ors_isochrone_max_batch_size: int = 5

    ors_coordinate_precision: float = 0.000001

    model_config = SettingsConfigDict(env_file='.env.ors')  # dead: disable

    @cached_property
    def client(self) -> openrouteservice.Client:
        # For future reference maybe check this suggestion: https://gitlab.heigit.org/climate-action/plugins/walkability/-/merge_requests/82#note_61406
        if self.ors_base_url is None:
            client = openrouteservice.Client(key=self.ors_api_key)
        else:
            client = openrouteservice.Client(base_url=self.ors_base_url, key=self.ors_api_key)

        openrouteservice.client._RETRIABLE_STATUSES = {502, 503}

        return client
