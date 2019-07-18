"""Base module for fetching data from afl_data service"""

from typing import Dict, Any, List
import json
from urllib.parse import urljoin
import os

import requests


LOCAL_AFL_DATA_SERVICE = "http://afl_data:8080"
AFL_DATA_SERVICE = os.getenv("AFL_DATA_SERVICE", default=LOCAL_AFL_DATA_SERVICE)


def _handle_response_data(response: requests.Response) -> List[Dict[str, Any]]:
    data = response.json()

    if isinstance(data, dict) and "error" in data.keys():
        raise RuntimeError(data["error"])

    if len(data) == 1:
        # For some reason, when returning match data with fetch_data=False,
        # plumber returns JSON as a big string inside a list, so we have to parse
        # the first element
        return json.loads(data[0])

    if any(data):
        return data

    return []


def _make_request(
    url: str, params: Dict[str, Any] = {}, headers: Dict[str, str] = {}
) -> requests.Response:
    response = requests.get(url, params=params, headers=headers)

    if response.status_code != 200:
        raise Exception(
            "Bad response from application: "
            f"{response.status_code} / {response.headers} / {response.text}"
        )

    return response


def fetch_afl_data(path: str, params: Dict[str, Any] = {}) -> List[Dict[str, Any]]:
    """
    Fetch data from the afl_data service

    Args:
        path (string): API endpoint to call.
        params (dict): Query parameters to include in the API request.

    Returns:
        list of dicts, representing the AFL data requested.
    """

    if os.getenv("PYTHON_ENV") == "production":
        service_host = AFL_DATA_SERVICE
        headers = {"Authorization": f'Bearer {os.getenv("GCR_TOKEN")}'}
    else:
        service_host = LOCAL_AFL_DATA_SERVICE
        headers = {}

    service_url = urljoin(service_host, path)
    response = _make_request(service_url, params=params, headers=headers)

    return _handle_response_data(response)
