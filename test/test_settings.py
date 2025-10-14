from datetime import timedelta

import responses


def test_custom_ors_client_retires_502(default_ors_settings):
    ors_settings = default_ors_settings.model_copy(deep=True)
    client = ors_settings.client
    client._retry_timeout = timedelta(seconds=1)

    with responses.RequestsMock() as rsps:
        resp1 = rsps.get(default_ors_settings.ors_base_url, json='{}', status=502)
        try:
            client.request(url='')
        except Exception:
            pass
        assert resp1.call_count > 1
