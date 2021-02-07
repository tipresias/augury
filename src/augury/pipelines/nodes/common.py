"""Pipeline nodes for transforming data. Used across different pipelines."""

from typing import Sequence, List, Dict, Any, cast, Callable, Optional, Union
from functools import reduce, partial, update_wrapper
import re
from datetime import timezone

import pandas as pd
import numpy as np
from dateutil import parser

from augury.settings import INDEX_COLS
from .base import _validate_required_columns, _validate_no_dodgy_zeros


DATE_STRING_REGEX = re.compile(r"\d{4}\-\d{2}\-\d{2}")


def convert_to_data_frame(
    *data: List[Dict[str, Any]]
) -> Union[List[pd.DataFrame], pd.DataFrame]:
    """
    Convert JSON data in the form of a list of dictionaries into a data frame.

    Params
    ------
    data (sequence of list of dictionaries): Data received from a JSON data set.

    Returns
    -------
    Sequence of pandas.DataFrame
    """
    if len(data) == 0:
        return pd.DataFrame()

    data_frames = [pd.DataFrame(datum) for datum in data]

    # TODO: Quick fix for empty roster data not having a 'date' column.
    # We should figure out a better way to handle empty data such that
    # its shape/columns are consistent, but one doesn't occur to me at the moment.
    for df in data_frames:
        if "date" in df.columns:
            df.loc[:, "date"] = pd.to_datetime(df["date"])

    return data_frames if len(data_frames) > 1 else data_frames[0]


def _combine_data_horizontally(*data_frames: Sequence[pd.DataFrame]):
    # We need to sort by length (going from longest to shortest), then keeping first
    # duplicated column to make sure we don't lose earlier values of shared columns
    # (e.g. dropping match data's 'date' column in favor of the betting data 'date'
    # column results in lots of NaT values, because betting data only goes back to 2010)
    sorted_data_frames: List[pd.DataFrame] = sorted(data_frames, key=len, reverse=True)
    joined_data_frame = pd.concat(sorted_data_frames, axis=1, sort=False)
    duplicate_columns = joined_data_frame.columns.duplicated(keep="first")

    combined_data_frame = joined_data_frame.loc[:, ~duplicate_columns].copy()

    # Due to vagaries in data updates, sometimes the longest data set is missing
    # the most-recent round which results in blank rows for required columns
    # like data and round_number. So, we make sure to fill those values
    # with the data set that has the most-recent data just to be sure.
    max_date_data_frame = max(sorted_data_frames, key=lambda df: df["date"].max())

    for column in set(joined_data_frame.columns[duplicate_columns]):
        combined_data_frame.loc[:, [column]] = combined_data_frame[column].fillna(
            max_date_data_frame[column]
        )

    filled_data_frame = combined_data_frame.fillna(0)

    _validate_no_dodgy_zeros(filled_data_frame)

    return filled_data_frame


def _append_data_frames(
    acc_data_frame: pd.DataFrame, curr_data_frame: pd.DataFrame
) -> pd.DataFrame:
    # Assumes the following:
    #     - All data frames have a date column
    #     - Data frames are sorted by date in ascending order
    #       (the data frames themselves, not their rows)
    #     - Data frames have all data for a given date (i.e. all matches played
    #         on a date, not 1 of 3, which would result in missing data)
    max_accumulated_date = acc_data_frame[  # pylint: disable=unused-variable
        "date"
    ].max()
    sliced_current_data_frame = curr_data_frame.query("date > @max_accumulated_date")

    return pd.concat([acc_data_frame, sliced_current_data_frame], axis=0, sort=False)


def _combine_data_vertically(*data_frames: Sequence[pd.DataFrame]):
    if len(data_frames) == 1:
        return data_frames[0]

    valid_data_frames = [
        df for df in cast(Sequence[pd.DataFrame], data_frames) if df.any().any()
    ]
    sorted_data_frames = sorted(valid_data_frames, key=lambda df: df["date"].min())
    combined_data_frame = reduce(_append_data_frames, sorted_data_frames).fillna(0)

    _validate_no_dodgy_zeros(combined_data_frame)

    return combined_data_frame


# TODO: Combining vertically vs horizontally are so different that it doesn't make
# sense for both to be contained within a single node. Make combine_data_vertically
# and combine_data_horizontally their own nodes.
def combine_data(axis: int = 0) -> pd.DataFrame:
    """Concatenate data frames from multiple sources into one data frame.

    Params
    ------
    data_frames (list of pandas.DataFrame): Data frames to be concatenated.
    axis (0 or 1, defaults to 0): Whether to concatenate by rows (0) or columns (1).

    Returns
    -------
    Concatenated data frame.
    """
    assert axis in [0, 1], f"Expected axis to be 0 or 1, but recieved {axis}"

    if axis == 0:
        return _combine_data_vertically

    return _combine_data_horizontally


def _filter_by_date(
    start_date: str, end_date: str, data_frame: pd.DataFrame
) -> pd.DataFrame:
    REQUIRED_COLS = {"date"}
    _validate_required_columns(REQUIRED_COLS, data_frame.columns)

    start_datetime = parser.parse(  # pylint: disable=unused-variable
        start_date
    ).replace(tzinfo=timezone.utc)
    end_datetime = parser.parse(end_date).replace(  # pylint: disable=unused-variable
        hour=11, minute=59, second=59, tzinfo=timezone.utc
    )

    return data_frame.query("date >= @start_datetime & date <= @end_datetime")


def filter_by_date(
    start_date: str, end_date: str
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Filter data frames by the given start and end_dates.

    Params
    ------
    start_date (str, YYYY-MM-DD): Earliest date (inclusive) for match data.
    end_date (str, YYYY-MM-DD): Latest date (inclusive) for match data.

    Returns
    -------
    Callable function that filters data frames by the given dates.
    """
    if not DATE_STRING_REGEX.match(start_date) or not DATE_STRING_REGEX.match(end_date):
        raise ValueError(
            f"Expected date string parameters start_date ({start_date}) and "
            f"end_date ({end_date}) to be of the form YYYY-MM-DD."
        )

    return update_wrapper(
        partial(_filter_by_date, start_date, end_date), _filter_by_date
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
    match_row_data_frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reshape data frame from one match per row to one team-match combination per row.

    This results in a data frame with two rows per match, one each for the home
    and away team.

    Params
    ------
    match_row_data_frame (pandas.DataFrame): Data frame to be transformed.

    Returns
    -------
    pandas.DataFrame
    """
    REQUIRED_COLS: List[str] = [
        "home_team",
        "away_team",
        "year",
        "round_number",
        "date",
    ]

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
        .drop("match_id", axis=1, errors="ignore")
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

    REQUIRED_COLS: List[str] = INDEX_COLS + ["oppo_team"] + cols_to_convert

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
    """Calculate team-based features for opposition teams and append them to the data.

    The team columns to use for creating oppo columns are based on match_cols
    (non-team-specific columns to ignore) or oppo_feature_cols
    (team-specific features to add 'oppo' versions of). Including both column arguments
    will raise an error.

    Params
    ------
    match_cols (list of strings): Names of columns to ignore (calculates oppo
        features for all columns not listed).
    oppo_feature_cols (list of strings): Names of columns to add oppo features of
        (ignores all columns not listed)

    Returns
    -------
    Function that takes pandas.DataFrame and returns another pandas.DataFrame
        with 'oppo_' columns added.
    """
    if any(match_cols) and any(oppo_feature_cols):
        raise ValueError(
            "To avoid conflicts, you can't include both match_cols "
            "and oppo_feature_cols. Choose the shorter list to determine which "
            "columns to skip and which to turn into opposition features."
        )

    return update_wrapper(
        partial(
            _add_oppo_features_node,
            match_cols=match_cols,
            oppo_feature_cols=oppo_feature_cols,
        ),
        _add_oppo_features_node,
    )


def finalize_data(
    data_frame: pd.DataFrame, index_cols: List[str] = INDEX_COLS
) -> pd.DataFrame:
    """Perform final data cleaning after all other data transformation steps.

    Params
    ------
    data_frame (pandas.DataFrame): Data frame that has been cleaned & transformed.

    Returns
    -------
    pandas.DataFrame that's ready to be fed into a machine-learning model.
    """
    final_data_frame = (
        data_frame.astype({"year": int})
        .fillna(0)
        .set_index(index_cols, drop=False)
        .rename_axis([None] * len(index_cols))
        .sort_index()
    )

    _validate_no_dodgy_zeros(final_data_frame)

    return final_data_frame


def _sort_data_frame_columns_node(
    category_cols: List[str], data_frame: pd.DataFrame
) -> pd.DataFrame:
    if category_cols is None:
        numeric_data_frame = data_frame.select_dtypes(
            include=["number", "datetimetz", "datetime"]
        ).fillna(0)
        category_data_frame = data_frame.drop(numeric_data_frame.columns, axis=1)
    else:
        category_data_frame = data_frame[category_cols]
        numeric_data_frame = data_frame.drop(category_data_frame.columns, axis=1)

    sorted_data_frame = pd.concat([category_data_frame, numeric_data_frame], axis=1)

    assert set(data_frame.columns) == set(
        sorted_data_frame.columns
    ), "Sorting data frames should not add or subtract columns."

    return sorted_data_frame


# TODO: This is for sorting columns in preparation for the ML pipeline.
# It should probably be at the start of that pipeline instead of at the end of the data
# pipeline
def sort_data_frame_columns(
    category_cols: Optional[List[str]] = None,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Sort data frame columns such that categories are grouped together on the left.

    Params
    ------
    category_cols: Explicitly define category columns to put on the left.
        If omitted, category columns are identified by their dtype not being
        'number' or a type of 'datetime'.
    """
    return update_wrapper(
        partial(_sort_data_frame_columns_node, category_cols),
        _sort_data_frame_columns_node,
    )


def clean_full_data(*data_frames: pd.DataFrame) -> pd.DataFrame:
    """Clean up data frames created from 'final_<pipeline>_data' JSON files."""
    # Need to convert dates from string to datetime for later
    # calculations/comparisons
    return [
        data_frame.assign(date=pd.to_datetime(data_frame["date"])).set_index(
            INDEX_COLS, drop=False
        )
        for data_frame in data_frames
    ]
