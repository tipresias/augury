"""Pipeline nodes for transforming betting data"""

from typing import List

import pandas as pd

from .base import (
    _parse_dates,
    _translate_team_column,
    _validate_required_columns,
    _validate_unique_team_index_columns,
    _filter_out_dodgy_data,
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
    clean_betting_data = (
        betting_data.rename(columns={"season": "year"})
        .pipe(
            _filter_out_dodgy_data(
                duplicate_subset=["year", "round_number", "home_team", "away_team"]
            )
        )
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

    _validate_unique_team_index_columns(clean_betting_data)

    return clean_betting_data


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
    _validate_required_columns(REQUIRED_COLS, data_frame.columns)

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
