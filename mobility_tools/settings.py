from functools import cached_property

import openrouteservice
from obstore.store import S3Store
from pydantic_settings import BaseSettings, SettingsConfigDict


class ORSSettings(BaseSettings):
    ors_base_url: str | None = None
    ors_api_key: str | None = None

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


class S3Settings(BaseSettings):
    s3_endpoint: str
    s3_access_key: str
    s3_secret_key: str
    s3_secure: bool | None = True
    s3_bucket: str
    s3_dem_version: str
    s3_default_filename: str

    model_config = SettingsConfigDict(env_file='.env.s3')

    @cached_property
    def s3store(self) -> S3Store:
        obstore_endpoint = self.s3_endpoint
        if not obstore_endpoint.startswith(('http://', 'https://')):
            scheme = 'https' if self.s3_secure else 'http'
            obstore_endpoint = f'{scheme}://{obstore_endpoint}'

        s3store = S3Store(
            bucket=self.s3_bucket,
            endpoint=obstore_endpoint,
            access_key_id=self.s3_access_key,
            secret_access_key=self.s3_secret_key,
            virtual_hosted_style_request=False,
        )

        return s3store

    @cached_property
    def obs_planet_source(self) -> str:
        return f'mapterhorn/{self.s3_dem_version}/{self.s3_default_filename}'
