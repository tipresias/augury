"""Pipeline nodes for transforming match data"""

from typing import Callable, List, Tuple
from functools import partial, reduce

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from mypy_extensions import TypedDict

from machine_learning.data_config import FOOTYWIRE_VENUE_TRANSLATIONS
from .base import _parse_dates, _translate_team_column, _validate_required_columns


EloDictionary = TypedDict(
    "EloDictionary",
    {
        "home_away_elo_ratings": List[Tuple[float, float]],
        "current_team_elo_ratings": np.ndarray,
        "year": int,
    },
)


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

# Constants for ELO calculations
BASE_RATING = 1000
K = 35.6
X = 0.49
M = 130
HOME_GROUND_ADVANTAGE = 9
S = 250
SEASON_CARRYOVER = 0.575


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


# Basing ELO calculations on:
# http://www.matterofstats.com/mafl-stats-journal/2013/10/13/building-your-own-team-rating-system.html
def _elo_formula(
    prev_elo_rating: float, prev_oppo_elo_rating: float, margin: int, at_home: bool
) -> float:
    home_ground_advantage = (
        HOME_GROUND_ADVANTAGE if at_home else HOME_GROUND_ADVANTAGE * -1
    )
    expected_outcome = 1 / (
        1 + 10 ** ((prev_oppo_elo_rating - prev_elo_rating - home_ground_advantage) / S)
    )
    actual_outcome = X + 0.5 - X ** (1 + (margin / M))

    return prev_elo_rating + (K * (actual_outcome - expected_outcome))


# Assumes df sorted by year & round_number with ascending=True in order to calculate
# correct ELO ratings
def _calculate_match_elo_rating(
    elo_ratings: EloDictionary,
    # match_row = [year, home_team, away_team, home_margin]
    match_row: np.ndarray,
) -> EloDictionary:
    match_year = match_row[0]

    # It's typical for ELO models to do a small adjustment toward the baseline between
    # seasons
    if match_year != elo_ratings["year"]:
        prematch_team_elo_ratings = (
            elo_ratings["current_team_elo_ratings"] * SEASON_CARRYOVER
        ) + BASE_RATING * (1 - SEASON_CARRYOVER)
    else:
        prematch_team_elo_ratings = elo_ratings["current_team_elo_ratings"].copy()

    home_team = int(match_row[1])
    away_team = int(match_row[2])
    home_margin = match_row[3]

    prematch_home_elo_rating = prematch_team_elo_ratings[home_team]
    prematch_away_elo_rating = prematch_team_elo_ratings[away_team]

    home_elo_rating = _elo_formula(
        prematch_home_elo_rating, prematch_away_elo_rating, home_margin, True
    )
    away_elo_rating = _elo_formula(
        prematch_away_elo_rating, prematch_home_elo_rating, home_margin * -1, False
    )

    postmatch_team_elo_ratings = prematch_team_elo_ratings.copy()
    postmatch_team_elo_ratings[home_team] = home_elo_rating
    postmatch_team_elo_ratings[away_team] = away_elo_rating

    return {
        "home_away_elo_ratings": elo_ratings["home_away_elo_ratings"]
        + [(prematch_home_elo_rating, prematch_away_elo_rating)],
        "current_team_elo_ratings": postmatch_team_elo_ratings,
        "year": match_year,
    }


def add_elo_rating(data_frame_arg: pd.DataFrame) -> pd.DataFrame:
    """Add ELO rating of team prior to matches"""

    INDEX_COLS = {"home_team", "year", "round_number"}
    REQUIRED_COLS = INDEX_COLS | {"home_score", "away_score", "away_team", "date"}

    _validate_required_columns(REQUIRED_COLS, data_frame_arg.columns)

    data_frame = (
        data_frame_arg.set_index(list(INDEX_COLS), drop=False).rename_axis(
            [None] * len(INDEX_COLS)
        )
        if INDEX_COLS != {*data_frame_arg.index.names}
        else data_frame_arg.copy()
    )

    if not data_frame.index.is_monotonic:
        data_frame.sort_index(inplace=True)

    le = LabelEncoder()
    le.fit(data_frame["home_team"])
    time_sorted_data_frame = data_frame.sort_values("date")

    elo_matrix = (
        time_sorted_data_frame.assign(
            home_team=lambda df: le.transform(df["home_team"]),
            away_team=lambda df: le.transform(df["away_team"]),
        )
        .eval("home_margin = home_score - away_score")
        .loc[:, ["year", "home_team", "away_team", "home_margin"]]
    ).values
    current_team_elo_ratings = np.full(len(set(data_frame["home_team"])), BASE_RATING)
    starting_elo_dictionary: EloDictionary = {
        "home_away_elo_ratings": [],
        "current_team_elo_ratings": current_team_elo_ratings,
        "year": 0,
    }

    elo_columns = reduce(
        _calculate_match_elo_rating, elo_matrix, starting_elo_dictionary
    )["home_away_elo_ratings"]

    elo_data_frame = pd.DataFrame(
        elo_columns,
        columns=["home_elo_rating", "away_elo_rating"],
        index=time_sorted_data_frame.index,
    ).sort_index()

    return pd.concat([data_frame, elo_data_frame], axis=1)
