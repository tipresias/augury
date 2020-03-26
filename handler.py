"""Collection of serverless functions.

All functions must take an HTTP request as a parameter and return an HTTP response
per Serverless Framework conventions.
"""

import os
import sys
from datetime import date
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
SRC_PATH = os.path.join(BASE_DIR, "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from augury import api


TRUE = "true"
FALSE = "false"


def _response(body, status=200, headers={}):
    response_headers = {**{"Content-Type": "application/json"}, **headers}

    return {
        "statusCode": status,
        "headers": response_headers,
        "body": json.dumps(body, indent=2),
    }


def _unauthorized_response():
    return _response("Not authorized", status=401)


def _request_is_authorized(event) -> bool:
    auth_token = event["headers"].get("Authorization")

    if auth_token == f"Bearer {os.getenv('GCPF_TOKEN')}":
        return True

    return False


def predictions(event, _context):
    """Generate predictions for the given year and round number.

    Params
    ------
    event (dict): A dict provided by AWS with information about the HTTP request.
        It can have the following query params:

        year_range (str, optional): Year range for which you want prediction data.
            Format = yyyy-yyyy.
            Default = current year only.
        round_number (int, optional): Round number for which you want prediction data.
            Default = All rounds for given year.
        ml_models (str, optional): Comma-separated list of names of ML model to use
            for making predictions.
            Default = All available models
    context (dict): A dict provided by AWS with information about the environment
        in which the function is running. Currently unused.

    Returns
    -------
    Dict with a body field that has a JSON of prediction data.
    """
    if not _request_is_authorized(event):
        return _unauthorized_response()

    this_year = date.today().year
    year_range_param = event.get("year_range", f"{this_year}-{this_year + 1}")
    year_range = tuple([int(year) for year in year_range_param.split("-")])

    round_number = event.get("round_number", None)
    round_number = int(round_number) if round_number is not None else None

    ml_models_param = event.get("ml_models", None)
    ml_models_param = (
        ml_models_param.split(",")
        if ml_models_param is not None and ml_models_param != ""
        else None
    )

    return _response(
        api.make_predictions(
            year_range, round_number=round_number, ml_model_names=ml_models_param
        )
    )


def fixtures(event, _context):
    """Fetch fixture data for the given date range.

    Params
    ------
    event (dict): A dict provided by AWS with information about the HTTP request.
        It can have the following query params:

        start_date (string of form 'yyyy-mm-dd', required): Start of date range
            (inclusive) for which you want data.
        start_date (string of form 'yyyy-mm-dd', required): End of date range
            (inclusive) for which you want data.
    context (dict): A dict provided by AWS with information about the environment
        in which the function is running. Currently unused.

    Returns
    -------
    Dict with a body field that has a JSON of prediction data.
    """
    if not _request_is_authorized(event):
        return _unauthorized_response()

    start_date = event.get("start_date")
    end_date = event.get("end_date")

    return _response(api.fetch_fixture_data(start_date, end_date))


def match_results(event, _context):
    """Fetch match results data for the given date range.

    Params
    ------
    event (dict): A dict provided by AWS with information about the HTTP request.
        It can have the following query params:

        start_date (string of form 'yyyy-mm-dd', required): Start of date range
            (inclusive) for which you want data.
        start_date (string of form 'yyyy-mm-dd', required): End of date range
            (inclusive) for which you want data.
    context (dict): A dict provided by AWS with information about the environment
        in which the function is running. Currently unused.

    Returns
    -------
    Dict with a body field that has a JSON of prediction data.
    """
    if not _request_is_authorized(event):
        return _unauthorized_response()

    start_date = event.get("start_date")
    end_date = event.get("end_date")

    return _response(api.fetch_match_results_data(start_date, end_date))


def ml_models(event, _context):
    """Fetch info for all available ML models.

    Params
    ------
    event (dict): A dict provided by AWS with information about the HTTP request.
    context (dict): A dict provided by AWS with information about the environment
        in which the function is running. Currently unused.

    Returns
    -------
    Dict with a body field that has a JSON of prediction data.
    """
    if not _request_is_authorized(event):
        return _unauthorized_response()

    return _response(api.fetch_ml_model_info())
