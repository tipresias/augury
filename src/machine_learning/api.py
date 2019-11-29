from typing import List, Optional, Dict, Tuple, Union, Any, Callable
from datetime import date
from functools import partial
import itertools

import pandas as pd
from mypy_extensions import TypedDict
import numpy as np
from kedro.context import load_context, KedroContext

from machine_learning.ml_data import MLData
from machine_learning.ml_estimators.base_ml_estimator import BaseMLEstimator
from machine_learning.data_import import match_data
from machine_learning.nodes import match
from machine_learning.settings import (
    SEED,
    ML_MODELS,
    BASE_DIR,
    PREDICTION_DATA_START_DATE,
)


np.random.seed(SEED)

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


def _train_model(ml_model: BaseMLEstimator, data: MLData) -> BaseMLEstimator:
    X_train, y_train = data.train_data()

    # On the off chance that we try to run predictions for years that have no relevant
    # prediction data
    if X_train.empty or y_train.empty:
        raise ValueError(
            "Some required data was missing for training for year range "
            f"{data.train_years}.\n"
            f"{'X_train is empty' if X_train.empty else ''}"
            f"{'and ' if X_train.empty and y_train.empty else ''}"
            f"{'y_train is empty' if y_train.empty else ''}"
        )

    ml_model.fit(X_train, y_train)

    return ml_model


def _make_model_predictions(
    year: int,
    data: MLData,
    context: KedroContext,
    ml_model: Dict[str, str],
    round_number: Optional[int] = None,
    verbose=1,
    train=False,
) -> pd.DataFrame:
    if verbose == 1:
        print(f"Making predictions with {ml_model['name']}")

    loaded_model = context.catalog.load(ml_model["name"])
    data.data_set = ml_model["data_set"]
    data.train_years = (None, year - 1)
    data.test_years = (year, year)

    trained_model = _train_model(loaded_model, data) if train else loaded_model

    X_test, _ = data.test_data()

    assert X_test.any().any(), (
        "X_test doesn't have any rows, likely due to no data being available for "
        f"{year}."
    )

    y_pred = trained_model.predict(X_test)
    data_row_slice = (slice(None), year, slice(round_number, round_number))

    model_predictions = (
        X_test.assign(predicted_margin=y_pred, ml_model=ml_model["name"])
        .set_index("ml_model", append=True, drop=False)
        .loc[
            data_row_slice,
            [
                "team",
                "year",
                "round_number",
                "oppo_team",
                "at_home",
                "ml_model",
                "predicted_margin",
            ],
        ]
    )

    assert model_predictions.any().any(), (
        "Model predictions data frame is empty, possibly due to a bad row slice:\n"
        f"{data_row_slice}"
    )

    return model_predictions


def _make_predictions_by_year(
    data: MLData,
    context: KedroContext,
    ml_model_names: Optional[List[str]],
    year: int,
    round_number: Optional[int] = None,
    verbose=1,
    train=False,
) -> pd.DataFrame:
    partial_make_model_predictions = partial(
        _make_model_predictions,
        year,
        data,
        context,
        round_number=round_number,
        verbose=verbose,
        train=train,
    )

    if ml_model_names is None:
        ml_models = ML_MODELS
    else:
        ml_models = [
            ml_model for ml_model in ML_MODELS if ml_model["name"] in ml_model_names
        ]

    assert any(ml_models), (
        "Couldn't find any ML models, check that at least one "
        f"{ml_model_names} is in ML_MODELS."
    )

    return [partial_make_model_predictions(ml_model) for ml_model in ml_models]


def make_predictions(
    year_range: Tuple[int, int],
    round_number: Optional[int] = None,
    data: MLData = MLData(start_date=PREDICTION_DATA_START_DATE, end_date=END_OF_YEAR),
    ml_model_names: Optional[List[str]] = None,
    verbose=1,
    train=False,
    context_func: Callable = load_context,
) -> ApiResponse:
    data.round_number = round_number

    context = context_func(
        BASE_DIR,
        start_date=f"{year_range[0]}-01-01",
        end_date=f"{year_range[1]}-12-31",
        round_number=round_number,
    )

    partial_make_predictions_by_year = partial(
        _make_predictions_by_year,
        data,
        context,
        ml_model_names,
        round_number=round_number,
        verbose=verbose,
        train=train,
    )

    predictions = [
        partial_make_predictions_by_year(year) for year in range(*year_range)
    ]

    return _api_response(pd.concat(list(itertools.chain.from_iterable(predictions))))


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
