"""Bottle server routes for mimicking serverless HTTP API.

See the serverless entrypoint main.py for documentation.
"""

import os
import sys
from datetime import date

from bottle import Bottle, run, request

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
SRC_PATH = os.path.join(BASE_DIR, "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from augury import api


app = Bottle()


@app.route("/predictions")
def predictions():
    """
    Generate predictions for the given year and round number.

    Accepts the following query params:
    year_range: Range of years to predict, with the format `yyyy-yyyy`.
        First year inclusive, second year exclusive per Python's `range` function.
        Predicts all seasons if omitted.
    round_number: Round number to predict. Predicts all rounds if omitted.
    ml_models: Comma separated names of ML models to use for predictions.
        Uses all ML models if omitted.
    train_models (bool, optional): Whether to train each model
        on earlier seasons' data before generating predictions
        for a given season/round.
        Default = False.
    """
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

    Accepts the following query params:
    start_date: The earliest date (inclusive) for which to fetch matches.
    end_date: The latest date (inclusive) for which to fetch matches.
    """
    start_date = request.query.start_date
    end_date = request.query.end_date

    return api.fetch_fixture_data(start_date, end_date)


@app.route("/match_results")
def match_results():
    """
    Fetch match results data for the given date range.

    Accepts the following query params:
    start_date: The earliest date (inclusive) for which to fetch matches.
    end_date: The latest date (inclusive) for which to fetch matches.
    """
    start_date = request.query.start_date
    end_date = request.query.end_date

    return api.fetch_match_results_data(start_date, end_date)


@app.route("/ml_models")
def ml_models():
    """Fetch info for all available ML models."""
    return api.fetch_ml_model_info()


run(app, host="0.0.0.0", port=8008, reloader=True)
