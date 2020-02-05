"""The public API for the Augury app."""

from typing import List, Optional, Dict, Union, Any
from datetime import date

import pandas as pd
from mypy_extensions import TypedDict
from kedro.context import load_context
import simplejson

from augury.data_import import match_data
from augury.nodes import match
from augury.predictions import Predictor
from augury.types import YearRange
from augury.settings import ML_MODELS, PREDICTION_DATA_START_DATE, BASE_DIR


ApiResponse = TypedDict(
    "ApiResponse", {"data": Union[List[Dict[str, Any]], Dict[str, Any]]}
)

END_OF_YEAR = f"{date.today().year}-12-31"


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


def make_predictions(
    year_range: YearRange,
    round_number: Optional[int] = None,
    ml_model_names: Optional[List[str]] = None,
    train=False,
) -> ApiResponse:
    """Generate match predictions with the given models for the given seasons."""
    context = load_context(
        BASE_DIR,
        start_date=PREDICTION_DATA_START_DATE,
        end_date=END_OF_YEAR,
        round_number=round_number,
    )

    predictor = Predictor(
        year_range,
        context,
        # Ignoring, because ProjectContext has project-specific attributes,
        # and importing it to use as a type tends to create circular dependencies
        round_number=context.round_number,  # type: ignore
        train=train,
        verbose=1,
    )

    if ml_model_names is None:
        ml_models = ML_MODELS
    else:
        ml_models = [
            ml_model for ml_model in ML_MODELS if ml_model["name"] in ml_model_names
        ]

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


def fetch_match_results_data(
    start_date: str, end_date: str, data_import=match_data, verbose: int = 1
) -> ApiResponse:
    """
    Fetch results data for past matches from afl_data service.

    Params
    ------
    start_date (str): Stringified date of form yyy-mm-dd that determines
        the earliest date for which to fetch data.
    end_date (str): Stringified date of form yyy-mm-dd that determines
        the latest date for which to fetch data.
    verbose (0 or 1): Whether to print info messages while fetching data.

    Returns
    -------
    List of match results data dictionaries.
    """
    return _api_response(
        pd.DataFrame(
            data_import.fetch_match_data(
                start_date=start_date, end_date=end_date, verbose=verbose
            )
        ).pipe(match.clean_match_data)
    )


def fetch_ml_model_info() -> ApiResponse:
    """Fetch general info about all saved ML models."""
    return _api_response(ML_MODELS)
