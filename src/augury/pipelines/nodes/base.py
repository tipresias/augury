"""Utility functions used in various kedro node functions."""

from typing import Callable, Union, Set, Sequence
from datetime import datetime, time
from dateutil import parser
import pytz

import pandas as pd
from mypy_extensions import TypedDict

from augury.settings import (
    TEAM_TRANSLATIONS,
    INDEX_COLS,
    VENUE_TIMEZONES,
    TEAM_NAMES,
)


ReplaceKwargs = TypedDict("ReplaceKwargs", {"hour": int, "minute": int})
# Some matches were played in Wellington, NZ from 2013 to 2015, and the earliest
# start time for those matches was 13:10
EARLIEST_NZ_START_TIME: ReplaceKwargs = {"hour": 13, "minute": 10}


def _localize_dates(row: pd.Series) -> datetime:
    # Defaulting to Melbourne time, when the datetime isn't location specific,
    # because the AFL has a pro-Melbourne bias, so why shouldn't we?
    venue_timezone_label = (
        VENUE_TIMEZONES.get(row["venue"])
        if "venue" in row.index
        else "Australia/Melbourne"
    )

    assert isinstance(
        venue_timezone_label, str
    ), f"Could not find timezone for {row['venue']}"

    match_date: datetime = parser.parse(row["date"])
    # For match dates without start times, we add the minimum start time that
    # (more-or-less) guarantees that converting times from local timezones to UTC
    # won't change the date as well. This should make joining data on dates
    # more consistent, because I couldn't find a case where converting a real match
    # start time to UTC changes the date.
    # 1. A quick check of player data found a minimum start time of 11:40
    #   at Manuka Oval (Canberra), which has a max UTC offset of +11, which is also
    #   the max offset in Australia.
    # 2. The max UTC offset for a match venue is Wellington (+13), but the earliest
    #   start time of matches played there is 13:10, and a match hasn't been played
    #   there since 2015 (meaning there's unlikely to be a future match there
    #   at an earlier time)
    # This may lead to specious start times, but so did defaulting to midnight.
    match_datetime = (
        match_date.replace(**EARLIEST_NZ_START_TIME)
        if match_date.time() == time()
        else match_date
    )

    return match_datetime.replace(tzinfo=pytz.timezone(venue_timezone_label))


def _format_time(unformatted_time: str):
    if not unformatted_time.isnumeric():
        return unformatted_time

    assert len(unformatted_time) in [3, 4], (
        "Time values are expected to be 3 or 4 digits, 1 or 2 for the hour "
        f"and 2 for the minute, but we received: {unformatted_time}."
    )

    return unformatted_time[:-2] + ":" + unformatted_time[2:]


def _parse_dates(data_frame: pd.DataFrame, time_col=None) -> pd.Series:
    localization_columns = list(
        set(["date", "venue", time_col]) & set(data_frame.columns)
    )
    localization_data = data_frame[localization_columns].astype(str)

    # Some data sources have separate date and local_start_time columns,
    # so we concat them to get consistent datetimes for matches
    if time_col is not None:
        localization_data.loc[:, "date"] = (
            localization_data["date"]
            + " "
            + localization_data[time_col].map(_format_time)
        )

    localized_dates = localization_data.apply(_localize_dates, axis=1)

    return pd.to_datetime(localized_dates, utc=True)


def _translate_team_name(team_name: str) -> str:
    return TEAM_TRANSLATIONS[team_name] if team_name in TEAM_TRANSLATIONS else team_name


def _translate_team_column(col_name: str) -> Callable[[pd.DataFrame], str]:
    return lambda data_frame: data_frame[col_name].map(_translate_team_name)


def _validate_required_columns(
    required_columns: Union[Set[str], Sequence[str]], data_frame_columns: pd.Index
):
    required_column_set = set(required_columns)
    data_frame_column_set = set(data_frame_columns)
    column_intersection = data_frame_column_set & required_column_set

    assert column_intersection == required_column_set, (
        f"{required_column_set} are required columns for this transformation, "
        "the missing columns are:\n"
        f"{required_column_set - column_intersection}"
    )


def _validate_unique_team_index_columns(data_frame: pd.DataFrame):
    duplicate_home_team_indices = data_frame[
        ["home_team", "year", "round_number"]
    ].duplicated(keep=False)
    assert not duplicate_home_team_indices.any().any(), (
        "Cleaning data resulted in rows with duplicate indices:\n"
        f"{data_frame[duplicate_home_team_indices]}"
    )

    duplicate_away_indices = data_frame[
        ["away_team", "year", "round_number"]
    ].duplicated(keep=False)
    assert not duplicate_away_indices.any().any(), (
        "Cleaning data resulted in rows with duplicate indices:\n"
        f"{data_frame[duplicate_away_indices]}"
    )


def _validate_no_dodgy_zeros(data_frame: pd.DataFrame):
    cols_to_check = list(set(data_frame.columns) & set(INDEX_COLS))

    if not any(cols_to_check):
        return None

    zero_value_query = " | ".join([f"{col} == 0" for col in cols_to_check])
    zeros_data_frame = data_frame.query(zero_value_query)

    assert (
        not zeros_data_frame.any().any()
    ), f"An invalid fillna produced index column values of 0:\n{zeros_data_frame}"


def _validate_canoncial_team_names(data_frame: pd.DataFrame):
    TEAM_NAME_COLS = ["team", "oppo_team", "home_team", "away_team", "playing_for"]

    cols_to_check = list(set(data_frame.columns) & set(TEAM_NAME_COLS))
    unique_team_names = set(data_frame[cols_to_check].to_numpy().flatten())
    non_canonical_team_names = unique_team_names - set(TEAM_NAMES)

    assert not any(non_canonical_team_names), (
        "All team names must be the canonical versions or table joins won't work. "
        "The non-canonical team names are:\n"
        f"{non_canonical_team_names}"
    )


def _filter_out_dodgy_data(keep="last", **kwargs) -> Callable:
    return lambda df: (
        df.sort_values("date", ascending=True)
        # Some early matches (1800s) have fully-duplicated rows.
        # Also, drawn finals get replayed, which screws up my indexing and a bunch of other
        # data munging, so we keep the 'last' finals played, which is the one
        # that didn't end in a tie.
        .drop_duplicates(keep=keep, **kwargs)
        # There were some weird round-robin rounds in the early days, and it's easier to
        # drop them rather than figure out how to split up the rounds.
        .query(
            "(year != 1897 | round_number != 15) "
            "& (year != 1924 | round_number != 19)"
        )
    )


# ID values are converted to floats automatically, making for awkward strings later.
# We want them as strings, because sometimes we have to use player names as replacement
# IDs, and we concatenate multiple ID values to create a unique index.
def _convert_id_to_string(id_label: str) -> Callable:
    return lambda df: df[id_label].astype(int).astype(str)
