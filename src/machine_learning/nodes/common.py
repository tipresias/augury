"""Pipeline nodes for transforming data"""

from typing import Sequence, List, Dict, Any, cast, Callable
from functools import reduce

import pandas as pd
import numpy as np


from machine_learning.data_config import INDEX_COLS
from .base import _validate_required_columns


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

    if len(data) == 1:
        return pd.DataFrame(data[0])

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

    return acc_data_frame.append(sliced_current_data_frame, sort=False)


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
        return reduce(_combine_data_vertically, sorted_data_frames).fillna(0)

    if axis == 1:
        return pd.concat(data_frames, axis=axis, sort=False).fillna(0)

    raise ValueError(f"Expected axis to be 0 or 1, but recieved {axis}")


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

    _validate_required_columns(REQUIRED_COLS, match_row_data_frame.columns)

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
