"""Pipeline nodes for transforming match data"""

from typing import Callable
from functools import partial

import pandas as pd

from machine_learning.data_config import FOOTYWIRE_VENUE_TRANSLATIONS
from .base import _parse_dates, _translate_team_column


MATCH_COL_TRANSLATIONS = {
    "home_points": "home_score",
    "away_points": "away_score",
    "margin": "home_margin",
    "season": "year",
    "game": "match_id",
    "home_goals": "home_team_goals",
    "away_goals": "away_team_goals",
    "home_behinds": "home_team_behinds",
    "away_behinds": "away_team_behinds",
}

SHARED_MATCH_FIXTURE_COLS = [
    "date",
    "venue",
    "year",
    "round_number",
    "home_team",
    "away_team",
]


# ID values are converted to floats automatically, making for awkward strings later.
# We want them as strings, because sometimes we have to use player names as replacement
# IDs, and we concatenate multiple ID values to create a unique index.
def _convert_id_to_string(id_label: str) -> Callable:
    return lambda df: df[id_label].astype(int).astype(str)


def _filter_out_dodgy_data(duplicate_subset=None) -> Callable:
    return lambda df: (
        df.sort_values("date", ascending=True)
        # Some early matches (1800s) have fully-duplicated rows.
        # Also, drawn finals get replayed, which screws up my indexing and a bunch of other
        # data munging, so getting match_ids for the repeat matches, and filtering
        # them out of the data frame
        .drop_duplicates(subset=duplicate_subset, keep="last")
        # There were some weird round-robin rounds in the early days, and it's easier to
        # drop them rather than figure out how to split up the rounds.
        .query(
            "(year != 1897 | round_number != 15) "
            "& (year != 1924 | round_number != 19)"
        )
    )


def clean_match_data(match_data: pd.DataFrame) -> pd.DataFrame:
    """
    Basic data cleaning, translation, and dropping in preparation for ML-specific
    transformations

    Args:
        match_data (pandas.DataFrame): Raw match data

    Returns:
        Cleanish pandas.DataFrame
    """
    if any(match_data):
        return (
            match_data.rename(columns=MATCH_COL_TRANSLATIONS)
            .astype({"year": int, "round_number": int})
            .pipe(
                _filter_out_dodgy_data(
                    duplicate_subset=["year", "round_number", "home_team", "away_team"]
                )
            )
            .assign(
                match_id=_convert_id_to_string("match_id"),
                home_team=_translate_team_column("home_team"),
                away_team=_translate_team_column("away_team"),
                date=_parse_dates,
            )
            .drop("round", axis=1)
            .sort_values("date")
        )

    return pd.DataFrame()


def _map_footywire_venues(venue: str) -> str:
    return (
        FOOTYWIRE_VENUE_TRANSLATIONS[venue]
        if venue in FOOTYWIRE_VENUE_TRANSLATIONS
        else venue
    )


def _map_round_type(year: int, round_number: int) -> str:
    TWENTY_ELEVEN_AND_LATER_FINALS = year > 2011 and round_number > 23
    TWENTY_TEN_FINALS = year == 2010 and round_number > 24
    TWO_THOUSAND_NINE_AND_EARLIER_FINALS = (
        year > 1994 and year < 2010 and round_number > 22
    )

    if year <= 1994:
        raise ValueError(
            f"Tried to get fixtures for {year}, but fixture data is meant for "
            "upcoming matches, not historical match data. Try getting fixture data "
            "from 1995 or later, or fetch match results data for older matches."
        )

    if (
        TWENTY_ELEVEN_AND_LATER_FINALS
        or TWENTY_TEN_FINALS
        or TWO_THOUSAND_NINE_AND_EARLIER_FINALS
    ):
        return "Finals"

    return "Regular"


def _round_type_column(data_frame: pd.DataFrame) -> pd.DataFrame:
    years = data_frame["year"].drop_duplicates()

    if len(years) > 1:
        raise ValueError(
            "Fixture data should only include matches from the next round, but "
            f"fixture data for seasons {years} were given"
        )

    return data_frame["round_number"].map(partial(_map_round_type, years.iloc[0]))


def _match_id_column(data_frame: pd.DataFrame) -> pd.Series:
    # AFL Tables match IDs start at 1 and go up, so using 0 & negative numbers
    # for fixture matches will guarantee uniqueness if it ever becomes an issue
    return pd.Series(range(0, -len(data_frame), -1)).astype(str)


def clean_fixture_data(fixture_data: pd.DataFrame) -> pd.DataFrame:
    """
    Basic data cleaning, translation, and dropping in preparation for ML-specific
    transformations

    Args:
        fixture_data (pandas.DataFrame): Raw fixture data

    Returns:
        Cleanish pandas.DataFrame
    """

    if any(fixture_data):
        return (
            fixture_data.rename(columns={"round": "round_number", "season": "year"})
            .loc[:, SHARED_MATCH_FIXTURE_COLS]
            .assign(
                venue=lambda df: df["venue"].map(_map_footywire_venues),
                round_type=_round_type_column,
                home_team=_translate_team_column("home_team"),
                away_team=_translate_team_column("away_team"),
                date=_parse_dates,
                match_id=_match_id_column,
            )
            .fillna(0)
        )

    return pd.DataFrame()
