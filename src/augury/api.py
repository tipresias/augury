"""The public API for the Augury app."""

from typing import List, Optional, Dict, Union, Any
from pathlib import Path
from datetime import date

import pandas as pd
from mypy_extensions import TypedDict
from kedro.framework.context import KedroContext
from kedro.framework.session import KedroSession
import simplejson

from augury.data_import import match_data
from augury.pipelines.match import nodes as match
from augury.predictions import Predictor
from augury.types import YearRange, MLModelDict
from augury import settings


ApiResponse = TypedDict(
    "ApiResponse", {"data": Union[List[Dict[str, Any]], Dict[str, Any]]}
)

PIPELINE_NAMES = {"full_data": "full", "legacy_data": "legacy"}


def _clean_data_frame_for_json(data_frame: pd.DataFrame) -> List[Dict[str, Any]]:
    # I don't feel great about this, but there isn't a good way of converting np.nan
    # to null for JSON. Since GCF expects dicts that it converts to JSON for us,
    # we call dumps then loads to avoid nested stringified weirdness.
    return simplejson.loads(
        simplejson.dumps(data_frame.to_dict("records"), ignore_nan=True, default=str)
    )


def _api_response(data: Union[pd.DataFrame, Dict[str, Any]]) -> ApiResponse:
    response_data = (
        _clean_data_frame_for_json(data) if isinstance(data, pd.DataFrame) else data
    )

    return {"data": response_data}


def _run_pipelines(context: KedroContext, ml_models: List[MLModelDict]):
    data_set_names = {ml_model["data_set"] for ml_model in ml_models}

    for data_set_name in data_set_names:
        context.run(pipeline_name=PIPELINE_NAMES[data_set_name])


def make_predictions(
    year_range: YearRange,
    round_number: Optional[int] = None,
    ml_model_names: Optional[List[str]] = None,
    train=False,
) -> ApiResponse:
    """Generate predictions for the given year and round number.

    Params
    ------
    year_range: Year range for which you want prediction data. Format = yyyy-yyyy.
    round_number: Round number for which you want prediction data.
    ml_models: Comma-separated list of names of ML model to use for making predictions.
    train: Whether to train the model before predicting.

    Returns
    -------
    List of prediction data dictionaries.
    """
    package_name = Path(__file__).resolve().parent.name
    extra_params = {
        "round_number": round_number,
        "start_date": "1897-01-01",
        "end_date": f"{date.today().year}-12-31",
    }
    with KedroSession.create(
        package_name, env=settings.ENV, extra_params=extra_params
    ) as session:
        context = session.load_context()

        if ml_model_names is None:
            ml_models = settings.ML_MODELS
        else:
            ml_models = [
                ml_model
                for ml_model in settings.ML_MODELS
                if ml_model["name"] in ml_model_names
            ]

        _run_pipelines(context, ml_models)

        predictor = Predictor(
            year_range,
            context,
            # Ignoring, because ProjectContext has project-specific attributes,
            # and importing it to use as a type tends to create circular dependencies
            round_number=round_number,  # type: ignore
            train=train,
            verbose=1,
        )

        predictions = predictor.make_predictions(ml_models)

        return _api_response(predictions)


def fetch_fixture_data(
    start_date: str, end_date: str, data_import=match_data, verbose: int = 1
) -> ApiResponse:
    """
    Fetch fixture data (doesn't include match results) from afl_data service.

    Params
    ------
        start_date (str): Stringified date of form yyy-mm-dd that determines
            the earliest date for which to fetch data.
        end_date (str): Stringified date of form yyy-mm-dd that determines
            the latest date for which to fetch data.
        verbose (0 or 1): Whether to print info messages while fetching data.

    Returns
    -------
    List of fixture data dictionaries.
    """
    return _api_response(
        pd.DataFrame(
            data_import.fetch_fixture_data(
                start_date=start_date, end_date=end_date, verbose=verbose
            )
        ).pipe(match.clean_fixture_data)
    )


def fetch_match_data(
    start_date: str, end_date: str, data_import=match_data, verbose: int = 1
) -> ApiResponse:
    """
    Fetch data for past matches from afl_data service.

    Params
    ------
    start_date (str): Stringified date of form yyy-mm-dd that determines
        the earliest date for which to fetch data.
    end_date (str): Stringified date of form yyy-mm-dd that determines
        the latest date for which to fetch data.
    verbose (0 or 1): Whether to print info messages while fetching data.

    Returns
    -------
    List of match data dictionaries.
    """
    return _api_response(
        pd.DataFrame(
            data_import.fetch_match_data(
                start_date=start_date, end_date=end_date, verbose=verbose
            )
        ).pipe(match.clean_match_data)
    )


def fetch_match_results_data(
    round_number: int, data_import=match_data, verbose: int = 1
) -> ApiResponse:
    """
    Fetch data for past matches from afl_data service.

    Params
    ------
    round_number: Fetch results for the given round.
    verbose (0 or 1): Whether to print info messages while fetching data.

    Returns
    -------
    List of match results data dictionaries.
    """
    return _api_response(
        pd.DataFrame(
            data_import.fetch_match_results_data(round_number, verbose=verbose)
        ).pipe(match.clean_match_results_data)
    )


def fetch_ml_model_info() -> ApiResponse:
    """Fetch general info about all saved ML models."""
    return _api_response(settings.ML_MODELS)
