"""Pipeline nodes for transforming betting data"""

from typing import List

import pandas as pd

from machine_learning.data_config import INDEX_COLS
from .base import _parse_dates, _translate_team_column


def _validate_required_columns(columns: pd.Index, required_columns: List[str]) -> None:
    req_col_set = {*required_columns}
    column_set = {*columns}

    if req_col_set == req_col_set & column_set:
        return None

    missing_columns = req_col_set - column_set

    raise ValueError(
        f"Required columns {missing_columns} are missing from the data frame."
    )


def clean_data(betting_data: pd.DataFrame) -> pd.DataFrame:
    """
    Basic data cleaning, translation, and dropping in preparation for ML-specific
    transformations

    Args:
        betting_data (pandas.DataFrame): Raw betting data

    Returns:
        pandas.DataFrame
    """
    return (
        betting_data.rename(columns={"season": "year"})
        .drop(
            [
                "home_win_paid",
                "home_line_paid",
                "away_win_paid",
                "away_line_paid",
                "venue",
                "home_margin",
                "away_margin",
            ],
            axis=1,
        )
        .assign(
            home_team=_translate_team_column("home_team"),
            away_team=_translate_team_column("away_team"),
            date=_parse_dates,
        )
        .drop("round", axis=1)
    )


def add_betting_pred_win(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Add whether a team is predicted to win per the betting odds

    Args:
        data_frame (pandas.DataFrame): A data frame with betting data.

    Returns:
        pandas.DataFrame with a 'betting_pred_win' column
    """

    REQUIRED_COLS: List[str] = [
        "win_odds",
        "oppo_win_odds",
        "line_odds",
        "oppo_line_odds",
    ]
    _validate_required_columns(data_frame.columns, REQUIRED_COLS)

    is_favoured = (
        (data_frame["win_odds"] < data_frame["oppo_win_odds"])
        | (data_frame["line_odds"] < data_frame["oppo_line_odds"])
    ).astype(int)
    odds_are_even = (
        (data_frame["win_odds"] == data_frame["oppo_win_odds"])
        & (data_frame["line_odds"] == data_frame["oppo_line_odds"])
    ).astype(int)

    # Give half point for predicted draws
    predicted_results = is_favoured + (odds_are_even * 0.5)

    return data_frame.assign(betting_pred_win=predicted_results)


def finalize_data(
    data_frame: pd.DataFrame, index_cols: List[str] = INDEX_COLS
) -> pd.DataFrame:
    """
    Perform final data cleaning after all the data transformations and feature
    building steps.

    Args:
        data_frame (pandas.DataFrame): Data frame that has been cleaned & transformed.

    Returns:
        pandas.DataFrame that's ready to be fed into a machine-learning model.
    """

    return (
        data_frame.astype({"year": int})
        .fillna(0)
        .set_index(index_cols, drop=False)
        .rename_axis([None] * len(index_cols))
        .sort_index()
    )
