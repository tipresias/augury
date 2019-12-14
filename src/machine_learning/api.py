from typing import List, Optional, Dict, Tuple, Union, Any
from datetime import date

import pandas as pd
from mypy_extensions import TypedDict

from machine_learning.data_import import match_data
from machine_learning.nodes import match
from machine_learning.predictions import Predictor
from machine_learning.settings import ML_MODELS, PREDICTION_DATA_START_DATE


PredictionData = TypedDict(
    "PredictionData",
    {
        "team": str,
        "year": int,
        "round_number": int,
        "at_home": int,
        "oppo_team": str,
        "ml_model": str,
        "predicted_margin": float,
    },
)

DataConfig = TypedDict(
    "DataConfig",
    {"team_names": List[str], "defunct_team_names": List[str], "venues": List[str]},
)

ApiResponse = TypedDict(
    "ApiResponse", {"data": Union[List[Dict[str, Any]], Dict[str, Any]]}
)

END_OF_YEAR = f"{date.today().year}-12-31"


def _clean_data_frame_for_json(data_frame: pd.DataFrame) -> List[Dict[str, Any]]:
    clean_data_frame = (
        data_frame.astype({"date": str})
        if "date" in data_frame.columns
        else data_frame.copy()
    )

    return clean_data_frame.to_dict("records")


def _api_response(data: Union[pd.DataFrame, Dict[str, Any]]) -> ApiResponse:
    response_data = (
        _clean_data_frame_for_json(data) if isinstance(data, pd.DataFrame) else data
    )

    return {"data": response_data}


def make_predictions(
    year_range: Tuple[int, int],
    round_number: Optional[int] = None,
    ml_model_names: Optional[List[str]] = None,
    train=False,
) -> ApiResponse:
    prediction_kwargs = {"train": train}

    if ml_model_names is not None:
        prediction_kwargs["ml_model_names"] = ml_model_names

    predictions = Predictor(
        year_range,
        round_number=round_number,
        verbose=1,
        start_date=PREDICTION_DATA_START_DATE,
        end_date=END_OF_YEAR,
    ).make_predictions(**prediction_kwargs)

    return _api_response(predictions)


def fetch_fixture_data(
    start_date: str, end_date: str, data_import=match_data, verbose: int = 1
) -> ApiResponse:
    """
    Fetch fixture data (doesn't include match results) from afl_data service.

    Args:
        start_date (str): Stringified date of form yyy-mm-dd that determines
            the earliest date for which to fetch data.
        end_date (str): Stringified date of form yyy-mm-dd that determines
            the latest date for which to fetch data.
        verbose (0 or 1): Whether to print info messages while fetching data.

    Returns:
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

    Args:
        start_date (str): Stringified date of form yyy-mm-dd that determines
            the earliest date for which to fetch data.
        end_date (str): Stringified date of form yyy-mm-dd that determines
            the latest date for which to fetch data.
        verbose (0 or 1): Whether to print info messages while fetching data.

    Returns:
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
    """Fetch general info about all saved ML models"""

    return _api_response(ML_MODELS)
