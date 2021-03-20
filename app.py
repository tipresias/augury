"""API routes and request resolvers for a Bottle app."""

from typing import Dict, Any
import os
import sys
from datetime import date

from kedro.framework.session import KedroSession
from bottle import Bottle, run, request, response


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
SRC_PATH = os.path.join(BASE_DIR, "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from augury import api


PYTHON_ENV = os.getenv("PYTHON_ENV", "development").lower()
IS_PRODUCTION = PYTHON_ENV == "production"
TIPRESIAS_HOST = (
    "http://www.tipresias.net" if IS_PRODUCTION else "http://host.docker.internal:8000"
)
PACKAGE_NAME = "augury"

app = Bottle()

if IS_PRODUCTION:
    from rollbar.contrib.bottle import RollbarBottleReporter

    rbr = RollbarBottleReporter(
        access_token=os.getenv("ROLLBAR_TOKEN", ""),
        environment=PYTHON_ENV,
    )
    app.install(rbr)


def _run_kwargs():
    run_kwargs: Dict[str, Any] = {
        "port": int(os.getenv("PORT", "8008")),
        "reloader": not IS_PRODUCTION,
        "host": "0.0.0.0",
        "server": "gunicorn",
        "accesslog": "-",
        "timeout": 1200,
        "workers": 1,
    }

    return run_kwargs


def _unauthorized_response():
    response.status = 401
    return "Unauthorized"


def _request_is_authorized(http_request) -> bool:
    auth_token = http_request.headers.get("Authorization")

    if (
        IS_PRODUCTION
        and auth_token != f"Bearer {os.environ['DATA_SCIENCE_SERVICE_TOKEN']}"
    ):
        return False

    return True


@app.route("/predictions")
def predictions():
    """
    Generate predictions for the given year and round number.

    Params
    ------
    Request with the following URL params:
        year_range (str, optional): Year range for which you want prediction data.
            Format = yyyy-yyyy.
            Default = current year only.
        round_number (int, optional): Round number for which you want prediction data.
            Default = All rounds for given year.
        ml_models (str, optional): Comma-separated list of names of ML model to use
            for making predictions.
            Default = All available models.
        train_models (bool, optional): Whether to train each model
            on earlier seasons' data before generating predictions
            for a given season/round.
            Default = False.

    Returns
    -------
    Response with a body that has a JSON of prediction data.
    """
    if not _request_is_authorized(request):
        return _unauthorized_response()

    this_year = date.today().year
    year_range_param = (
        f"{this_year}-{this_year + 1}"
        if request.query.year_range in [None, ""]
        else request.query.year_range
    )
    year_range = tuple([int(year) for year in year_range_param.split("-")])

    round_number = request.query.round_number
    round_number = None if round_number in [None, ""] else int(round_number)

    ml_models_param = request.query.ml_models
    ml_models_param = (
        None if ml_models_param in [None, ""] else ml_models_param.split(",")
    )

    train_models_param = request.query.train_models
    train_models = train_models_param.lower() == "true"

    with KedroSession.create(
        settings.PACKAGE_NAME, extra_params={"round_number": round_number}
    ):
        return api.make_predictions(
            year_range,
            round_number=round_number,
            ml_model_names=ml_models_param,
            train=train_models,
        )


@app.route("/fixtures")
def fixtures():
    """
    Fetch fixture data for the given date range.

    Params
    ------
    Request with the following URL params:
        start_date (string of form 'yyyy-mm-dd', required): Start of date range
            (inclusive) for which you want data.
        end_date (string of form 'yyyy-mm-dd', required): End of date range
            (inclusive) for which you want data.

    Returns
    -------
    Response with a body that has a JSON of fixture data.
    """
    if not _request_is_authorized(request):
        return _unauthorized_response()

    start_date = request.query.start_date
    end_date = request.query.end_date

    with KedroSession.create(settings.PACKAGE_NAME):
        return api.fetch_fixture_data(start_date, end_date)


@app.route("/match_results")
def match_results():
    """
    Fetch match results data for the given round.

    Params
    ------
    Request with the following URL params:
        round_number (int): Fetch data for the given round. If missing, will fetch
            all match results for the current year.

    Returns
    -------
    Response with a body that has a JSON of match results data.
    """
    if not _request_is_authorized(request):
        return _unauthorized_response()

    round_number = request.query.round_number

    with KedroSession.create(settings.PACKAGE_NAME):
        return api.fetch_match_results_data(round_number)


@app.route("/matches")
def matches():
    """
    Fetch match data for the given date range.

    Params
    ------
    Request with the following URL params:
        start_date (string of form 'yyyy-mm-dd', required): Start of date range
            (inclusive) for which you want data.
        end_date (string of form 'yyyy-mm-dd', required): End of date range
            (inclusive) for which you want data.

    Returns
    -------
    Response with a body that has a JSON of match data.
    """
    if not _request_is_authorized(request):
        return _unauthorized_response()

    start_date = request.query.start_date
    end_date = request.query.end_date

    with KedroSession.create(settings.PACKAGE_NAME):
        return api.fetch_match_data(start_date, end_date)


@app.route("/ml_models")
def ml_models():
    """
    Fetch info for all available ML models.

    Returns
    -------
    Response with a body that has a JSON of ML model data.
    """
    if not _request_is_authorized(request):
        return _unauthorized_response()

    with KedroSession.create(settings.PACKAGE_NAME):
        return api.fetch_ml_model_info()


run(app, **_run_kwargs())
