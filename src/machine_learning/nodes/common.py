"""Pipeline nodes for transforming data"""

from typing import Sequence, List, Dict, Any, cast, Callable, Optional
from functools import reduce, partial

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


def _validate_no_duplicated_columns(data_frame: pd.DataFrame) -> None:
    are_duplicate_columns = data_frame.columns.duplicated()

    assert not are_duplicate_columns.any(), (
        "The data frame with 'oppo' features added has duplicate columns."
        "The offending columns are: "
        f"{data_frame.columns[are_duplicate_columns]}"
    )


def _oppo_features(data_frame: pd.DataFrame, cols_to_convert) -> Optional[pd.DataFrame]:
    if not any(cols_to_convert):
        return None

    oppo_cols = {col_name: f"oppo_{col_name}" for col_name in cols_to_convert}
    oppo_col_names = oppo_cols.values()
    column_translations = {**{"oppo_team": "team"}, **oppo_cols}

    return (
        data_frame.reset_index(drop=True)
        .loc[:, ["year", "round_number", "oppo_team"] + list(cols_to_convert)]
        # We switch out oppo_team for team in the index,
        # then assign feature as oppo_{feature_column}
        .rename(columns=column_translations)
        .set_index(INDEX_COLS)
        .sort_index()
        .loc[:, list(oppo_col_names)]
    )


def _cols_to_convert_to_oppo(
    data_frame: pd.DataFrame,
    match_cols: List[str] = [],
    oppo_feature_cols: List[str] = [],
) -> List[str]:
    if any(oppo_feature_cols):
        return oppo_feature_cols

    return [col for col in data_frame.columns if col not in match_cols]


def _add_oppo_features_node(
    data_frame: pd.DataFrame,
    match_cols: List[str] = [],
    oppo_feature_cols: List[str] = [],
) -> pd.DataFrame:
    cols_to_convert = _cols_to_convert_to_oppo(
        data_frame, match_cols=match_cols, oppo_feature_cols=oppo_feature_cols
    )

    REQUIRED_COLS: List[str] = (INDEX_COLS + ["oppo_team"] + cols_to_convert)

    _validate_required_columns(REQUIRED_COLS, data_frame.columns)

    transform_data_frame = (
        data_frame.copy()
        .set_index(INDEX_COLS, drop=False)
        .rename_axis([None] * len(INDEX_COLS))
        .sort_index()
    )

    concated_data_frame = pd.concat(
        [transform_data_frame, _oppo_features(transform_data_frame, cols_to_convert)],
        axis=1,
    )

    _validate_no_duplicated_columns(concated_data_frame)

    return concated_data_frame


def add_oppo_features(
    match_cols: List[str] = [], oppo_feature_cols: List[str] = []
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Add oppo_team equivalents for team features based on the oppo_team for that
    match based on match_cols (non-team-specific columns to ignore) or
    oppo_feature_cols (team-specific features to add 'oppo' versions of). Including both
    column arguments will raise an error.

    Args:
        match_cols (list of strings): Names of columns to ignore (calculates oppo
            features for all columns not listed).
        oppo_feature_cols (list of strings): Names of columns to add oppo features of
            (ignores all columns not listed)

    Returns:
        Function that takes pandas.DataFrame and returns another pandas.DataFrame
        with 'oppo_' columns added.
    """

    if any(match_cols) and any(oppo_feature_cols):
        raise ValueError(
            "To avoid conflicts, you can't include both match_cols "
            "and oppo_feature_cols. Choose the shorter list to determine which "
            "columns to skip and which to turn into opposition features."
        )

    return partial(
        _add_oppo_features_node,
        match_cols=match_cols,
        oppo_feature_cols=oppo_feature_cols,
    )
