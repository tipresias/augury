"""Base module for fetching data from afl_data service."""

from typing import Dict, Any, List
import os
import time

import requests


LOCAL_AFL_DATA_SERVICE = "http://afl_data:8080"
AFL_DATA_SERVICE = os.getenv("AFL_DATA_SERVICE", default="")


def _handle_response_data(response: requests.Response) -> List[Dict[str, Any]]:
    parsed_response = response.json()

    if isinstance(parsed_response, dict) and "error" in parsed_response.keys():
        raise RuntimeError(parsed_response["error"])

    data = parsed_response.get("data")

    if any(data):
        return data

    return []


def _make_request(
    url: str, params: Dict[str, Any] = {}, headers: Dict[str, str] = {}, retry=True
) -> requests.Response:
    response = requests.get(url, params=params, headers=headers)

    if response.status_code != 200:
        # If it's the first call to afl_data service in awhile, the response takes
        # longer due to the container getting started, and it sometimes times out,
        # so we'll retry once just in case
        if retry:
            print(f"Received an error response from {url}, retrying...")
            time.sleep(5)
            _make_request(url, params=params, headers=headers, retry=False)

        raise Exception(
            "Bad response from application: "
            f"{response.status_code} / {response.headers} / {response.text}"
        )

    return response


def fetch_afl_data(path: str, params: Dict[str, Any] = {}) -> List[Dict[str, Any]]:
    """
    Fetch data from the afl_data service.

    Params
    ------
    path (string): API endpoint to call.
    params (dict): Query parameters to include in the API request.

    Returns
    -------
    list of dicts, representing the AFL data requested.
    """
    if os.getenv("PYTHON_ENV") == "production":
        service_host = AFL_DATA_SERVICE
        headers = {"Authorization": f'Bearer {os.getenv("AFL_DATA_SERVICE_TOKEN")}'}
    else:
        service_host = LOCAL_AFL_DATA_SERVICE
        headers = {}

    service_url = service_host + path
    response = _make_request(service_url, params=params, headers=headers)

    return _handle_response_data(response)
