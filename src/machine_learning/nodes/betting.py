"""Pipeline nodes for transforming betting data"""

from typing import Callable, List, Sequence, Dict, Any, cast
from functools import reduce

import pandas as pd
import numpy as np

from machine_learning.data_config import TEAM_TRANSLATIONS, INDEX_COLS
from machine_learning.settings import MELBOURNE_TIMEZONE


def convert_to_data_frame(
    *data: Sequence[List[Dict[str, Any]]]
) -> Sequence[pd.DataFrame]:
    """
    Converts JSON data in the form of a list of dictionaries into a data frame

    Args:
        data (sequence of list of dictionaries): Data received from a JSON data set.

    Returns:
        Sequence of pandas.DataFrame
    """

    return [pd.DataFrame(datum) for datum in data]


def _combine_data_vertically(
    acc_data_frame: pd.DataFrame, curr_data_frame: pd.DataFrame
) -> pd.DataFrame:
    """
    Assumes the following:
        - All data frames have a date column
        - Data frames are sorted by date in ascending order
        - Data frames have all data for a given date (i.e. all matches played
            on a date, not 1 of 3, which would result in missing data)
    """

    max_accumulated_date = acc_data_frame[  # pylint: disable=unused-variable
        "date"
    ].max()
    sliced_current_data_frame = curr_data_frame.query("date > @max_accumulated_date")

    return acc_data_frame.append(sliced_current_data_frame)


def combine_data(*data_frames: Sequence[pd.DataFrame], axis=0) -> pd.DataFrame:
    """
    Concatenate data frames from multiple sources into one data frame

    Args:
        data_frames (list of pandas.DataFrame): Data frames to be concatenated.
        axis (0 or 1, defaults to 0): Whether to concatenate by rows (0) or columns (1).

    Returns:
        Concatenated data frame.
    """

    if len(data_frames) == 1:
        return data_frames[0]

    if axis == 0:
        sorted_data_frames = sorted(
            cast(Sequence[pd.DataFrame], data_frames), key=lambda df: df["date"].min()
        )
        return reduce(_combine_data_vertically, sorted_data_frames)

    if axis == 1:
        return pd.concat(data_frames, axis=axis)

    raise ValueError(f"Expected axis to be 0 or 1, but recieved {axis}")


def _validate_required_columns(columns: pd.Index, required_columns: List[str]) -> None:
    req_col_set = {*required_columns}
    column_set = {*columns}

    if req_col_set == req_col_set & column_set:
        return None

    missing_columns = req_col_set - column_set

    raise ValueError(
        f"Required columns {missing_columns} are missing from the data frame."
    )


def _parse_dates(data_frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(data_frame["date"]).dt.tz_localize(MELBOURNE_TIMEZONE)


def _translate_team_name(team_name: str) -> str:
    return TEAM_TRANSLATIONS[team_name] if team_name in TEAM_TRANSLATIONS else team_name


def _translate_team_column(col_name: str) -> Callable[[pd.DataFrame], str]:
    return lambda data_frame: data_frame[col_name].map(_translate_team_name)


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


def _replace_col_names(team_type: str, oppo_team_type: str) -> Callable[[str], str]:
    return lambda col_name: (
        col_name.replace(f"{team_type}_", "").replace(f"{oppo_team_type}_", "oppo_")
    )


def _team_data_frame(data_frame: pd.DataFrame, team_type: str) -> pd.DataFrame:
    is_home_team = team_type == "home"

    if is_home_team:
        oppo_team_type = "away"
        at_home_col = np.ones(len(data_frame))
    else:
        oppo_team_type = "home"
        at_home_col = np.zeros(len(data_frame))

    return (
        data_frame.rename(columns=_replace_col_names(team_type, oppo_team_type))
        .assign(at_home=at_home_col)
        .set_index(INDEX_COLS, drop=False)
        # Renaming index cols 'None' to avoid warnings about duplicate column names
        .rename_axis([None] * len(INDEX_COLS))
    )


def convert_match_rows_to_teammatch_rows(
    match_row_data_frame: pd.DataFrame
) -> pd.DataFrame:
    """
    Reshape data frame from one match per row, with home_team and away_team columns,
    to one team-match combination per row (i.e. two rows per match), with team and
    oppo_team columns.


    Args:
        match_row_data_frame (pandas.DataFrame): Data frame to be transformed.

    Returns:
        pandas.DataFrame
    """

    REQUIRED_COLS: List[str] = ["home_team", "year", "round_number", "date"]

    _validate_required_columns(match_row_data_frame.columns, REQUIRED_COLS)

    team_data_frames = [
        _team_data_frame(match_row_data_frame, "home"),
        _team_data_frame(match_row_data_frame, "away"),
    ]

    return (
        pd.concat(team_data_frames, join="inner")
        .sort_values("date", ascending=True)
        # Various finals matches have been draws and replayed,
        # and sometimes home/away is switched requiring us to drop duplicates
        # at the end.
        # This eliminates some matches from Round 15 in 1897, because they
        # played some sort of round-robin tournament for finals, but I'm
        # not too worried about the loss of that data.
        .drop_duplicates(subset=INDEX_COLS, keep="last")
        .sort_index()
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
