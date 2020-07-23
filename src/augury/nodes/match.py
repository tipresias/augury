"""Pipeline nodes for transforming match data."""

from typing import List, Tuple
from functools import partial, reduce, update_wrapper
import math
import re

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from mypy_extensions import TypedDict

from augury.settings import (
    FOOTYWIRE_VENUE_TRANSLATIONS,
    CITIES,
    VENUE_CITIES,
    TEAM_CITIES,
    INDEX_COLS,
)
from .base import (
    _parse_dates,
    _translate_team_column,
    _validate_required_columns,
    _filter_out_dodgy_data,
    _convert_id_to_string,
    _validate_unique_team_index_columns,
    _validate_canoncial_team_names,
)


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

MATCH_INDEX_COLS = ["year", "round_number"]

OPPO_REGEX = re.compile("^oppo_")

# Constants for Elo calculations
BASE_RATING = 1000
K = 35.6
X = 0.49
M = 130
HOME_GROUND_ADVANTAGE = 9
S = 250
SEASON_CARRYOVER = 0.575

EARTH_RADIUS = 6371
TEAM_LEVEL = 0
YEAR_LEVEL = 1
WIN_POINTS = 4


# AFLTables has some incorrect home/away team designations for finals 2019
def _correct_home_away_teams(match_data: pd.DataFrame) -> pd.DataFrame:
    round_24_2019 = (match_data["year"] == 2019) & (match_data["round_number"] == 24)
    incorrect_24_home_team = (match_data["home_team"] == "Collingwood") | (
        match_data["home_team"] == "Richmond"
    )

    round_25_2019 = (match_data["year"] == 2019) & (match_data["round_number"] == 25)
    incorrect_25_home_team = match_data["home_team"] == "GWS"

    round_26_2019 = (match_data["year"] == 2019) & (match_data["round_number"] == 26)
    incorrect_26_home_team = match_data["home_team"] == "GWS"

    reversed_home_away_teams = (
        (round_24_2019 & incorrect_24_home_team)
        | (round_25_2019 & incorrect_25_home_team)
        | (round_26_2019 & incorrect_26_home_team)
    )

    correct_match_data = match_data.loc[~reversed_home_away_teams, :]
    incorrect_match_data = match_data.loc[reversed_home_away_teams, :]

    renamed_match_data = incorrect_match_data.rename(
        columns=lambda col_name: (
            col_name.replace("home_", "away_")
            if "home_" in col_name
            else col_name.replace("away_", "home_")
        )
    )

    return correct_match_data.append(renamed_match_data, sort=False)


def clean_match_data(match_data: pd.DataFrame) -> pd.DataFrame:
    """Clean, translate, and drop data in preparation for ML-specific transformations.

    Params
    ------
    match_data (pandas.DataFrame): Raw match data

    Returns
    -------
    Cleanish pandas.DataFrame
    """
    if any(match_data):
        clean_data = (
            match_data.rename(columns=MATCH_COL_TRANSLATIONS)
            .astype({"year": int, "round_number": int})
            .pipe(
                _filter_out_dodgy_data(
                    subset=["year", "round_number", "home_team", "away_team"]
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
            .pipe(_correct_home_away_teams)
        )

        _validate_unique_team_index_columns(clean_data)
        _validate_canoncial_team_names(clean_data)

        return clean_data

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

    assert len(years) == 1, (
        "Fixture data should only include matches from the next round, but "
        f"fixture data for seasons {list(years.values)} were given. "
        f"The offending series is:\n{data_frame['year']}"
    )

    return data_frame["round_number"].map(
        update_wrapper(partial(_map_round_type, years.iloc[0]), _map_round_type)
    )


def _match_id_column(data_frame: pd.DataFrame) -> pd.Series:
    # AFL Tables match IDs start at 1 and go up, so using 0 & negative numbers
    # for fixture matches will guarantee uniqueness if it ever becomes an issue
    return pd.Series(range(0, -len(data_frame), -1)).astype(str)


def clean_fixture_data(fixture_data: pd.DataFrame) -> pd.DataFrame:
    """Clean, translate, and drop data in preparation for ML-specific transformations.

    Params
    ------
    fixture_data (pandas.DataFrame): Raw fixture data

    Returns
    -------
    Cleanish pandas.DataFrame
    """
    if not fixture_data.any().any():
        return pd.DataFrame()

    fixture_data_frame = (
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
        # TEMPORARY fix for bad round numbers in the fixture data due to the AFL
        # scheduling a constant run of matches to finish out the season as early as
        # possible. The upcoming round is still okay, but we'll need to fix
        # the data source before the next round.
        .pipe(
            _filter_out_dodgy_data(
                subset=["year", "round_number", "home_team"], keep="first"
            )
        )
        .pipe(
            _filter_out_dodgy_data(
                subset=["year", "round_number", "away_team"], keep="first"
            )
        )
    )

    _validate_unique_team_index_columns(fixture_data_frame)
    _validate_canoncial_team_names(fixture_data_frame)

    return fixture_data_frame


# Basing Elo calculations on:
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
# correct Elo ratings
def _calculate_match_elo_rating(
    elo_ratings: EloDictionary,
    # match_row = [year, home_team, away_team, home_margin]
    match_row: np.ndarray,
) -> EloDictionary:
    match_year = match_row[0]

    # It's typical for Elo models to do a small adjustment toward the baseline between
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
    """Append a column for teams' prematch Elo ratings."""
    ELO_INDEX_COLS = {"home_team", "year", "round_number"}
    REQUIRED_COLS = ELO_INDEX_COLS | {"home_score", "away_score", "away_team", "date"}

    _validate_required_columns(REQUIRED_COLS, data_frame_arg.columns)

    data_frame = (
        data_frame_arg.set_index(list(ELO_INDEX_COLS), drop=False).rename_axis(
            [None] * len(ELO_INDEX_COLS)
        )
        if ELO_INDEX_COLS != {*data_frame_arg.index.names}
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


def add_out_of_state(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Append a column for whether a team is playing out of their home state."""
    REQUIRED_COLS = {"venue", "team"}
    _validate_required_columns(REQUIRED_COLS, data_frame.columns)

    venue_state = data_frame["venue"].map(lambda x: CITIES[VENUE_CITIES[x]]["state"])
    team_state = data_frame["team"].map(lambda x: CITIES[TEAM_CITIES[x]]["state"])

    return data_frame.assign(out_of_state=(team_state != venue_state).astype(int))


# Got the formula from https://www.movable-type.co.uk/scripts/latlong.html
def _haversine_formula(
    lat_long1: Tuple[float, float], lat_long2: Tuple[float, float]
) -> float:
    """Calculate distance between two pairs of latitudes & longitudes."""
    lat1, long1 = lat_long1
    lat2, long2 = lat_long2
    # Latitude & longitude are in degrees, so have to convert to radians for
    # trigonometric functions
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = phi2 - phi1
    delta_lambda = math.radians(long2 - long1)
    a = math.sin(delta_phi / 2) ** 2 + (
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return EARTH_RADIUS * c


def add_travel_distance(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Append column for distances between teams' home cities and match venue cities."""
    required_cols = {"venue", "team"}
    _validate_required_columns(required_cols, data_frame.columns)

    venue_lat_long = data_frame["venue"].map(
        lambda x: (CITIES[VENUE_CITIES[x]]["lat"], CITIES[VENUE_CITIES[x]]["long"])
    )
    team_lat_long = data_frame["team"].map(
        lambda x: (CITIES[TEAM_CITIES[x]]["lat"], CITIES[TEAM_CITIES[x]]["long"])
    )

    return data_frame.assign(
        travel_distance=[
            _haversine_formula(*lats_longs)
            for lats_longs in zip(venue_lat_long, team_lat_long)
        ]
    )


def add_result(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Append a column for teamsa match results (win, draw, loss) as float."""
    REQUIRED_COLS = {"score", "oppo_score"}
    _validate_required_columns(REQUIRED_COLS, data_frame.columns)

    wins = (data_frame["score"] > data_frame["oppo_score"]).astype(int)
    draws = (data_frame["score"] == data_frame["oppo_score"]).astype(int) * 0.5

    return data_frame.assign(result=wins + draws)


def add_margin(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Append a column for teams' points margins from matches."""
    REQUIRED_COLS = {"score", "oppo_score"}
    _validate_required_columns(REQUIRED_COLS, data_frame.columns)

    return data_frame.assign(margin=data_frame["score"] - data_frame["oppo_score"])


def _shift_features(columns: List[str], shift: bool, data_frame: pd.DataFrame):
    if shift:
        columns_to_shift = columns
    else:
        columns_to_shift = [col for col in data_frame.columns if col not in columns]

    _validate_required_columns(columns_to_shift, data_frame.columns)

    shifted_col_names = {col: f"prev_match_{col}" for col in columns_to_shift}

    # Group by team (not team & year) to get final score from previous season for round 1.
    # This reduces number of rows that need to be dropped and prevents gaps
    # for cumulative features
    shifted_features = (
        data_frame.groupby("team")[columns_to_shift]
        .shift()
        .fillna(0)
        .rename(columns=shifted_col_names)
    )

    return pd.concat([data_frame, shifted_features], axis=1)


def add_shifted_team_features(
    shift_columns: List[str] = [], keep_columns: List[str] = []
):
    """Group features by team and shift by one to get previous match stats.

    Use shift_columns to indicate which features to shift or keep_columns for features
    to leave unshifted, but not both.
    """
    if any(shift_columns) and any(keep_columns):
        raise ValueError(
            "To avoid conflicts, you can't include both match_cols "
            "and oppo_feature_cols. Choose the shorter list to determine which "
            "columns to skip and which to turn into opposition features."
        )

    shift = any(shift_columns)
    columns = shift_columns if shift else keep_columns

    return update_wrapper(partial(_shift_features, columns, shift), _shift_features)


def add_cum_win_points(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Append a column for teams' cumulative win points per season."""
    REQUIRED_COLS = {"prev_match_result"}
    _validate_required_columns(REQUIRED_COLS, data_frame.columns)

    cum_win_points_col = (
        (data_frame["prev_match_result"] * WIN_POINTS)
        .groupby(level=[TEAM_LEVEL, YEAR_LEVEL])
        .cumsum()
    )

    return data_frame.assign(cum_win_points=cum_win_points_col)


def add_win_streak(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Append a column for teams' running win/loss streaks.

    Streaks calculated through the end of the current match. Positive result
    (win or draw) adds 1 (or 0.5); negative result subtracts 1. Changes in direction
    (i.e. broken streak) result in starting over at 1 or -1.
    """
    REQUIRED_COLS = {"prev_match_result"}
    _validate_required_columns(REQUIRED_COLS, data_frame.columns)

    win_groups = data_frame["prev_match_result"].groupby(
        level=TEAM_LEVEL, group_keys=False
    )
    streak_groups = []

    for team_group_key, team_group in win_groups:
        streaks: List = []

        for idx, result in enumerate(team_group):
            # 1 represents win, 0.5 represents draw
            if result > 0:
                if idx == 0 or streaks[idx - 1] <= 0:
                    streaks.append(result)
                else:
                    streaks.append(streaks[idx - 1] + result)
            # 0 represents loss
            elif result == 0:
                if idx == 0 or streaks[idx - 1] >= 0:
                    streaks.append(-1)
                else:
                    streaks.append(streaks[idx - 1] - 1)
            elif result < 0:
                raise ValueError(
                    f"No results should be negative, but {result} "
                    f"is at index {idx} of group {team_group_key}"
                )
            else:
                # For a team's first match in the data set or any rogue NaNs, we add 0
                streaks.append(0)

        streak_groups.extend(streaks)

    return data_frame.assign(
        win_streak=pd.Series(streak_groups, index=data_frame.index)
    )


def add_cum_percent(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Append a column for teams' cumulative percentages.

    This is an official stat used as a tie-breaker for AFL ladder positions
    and is calculated as cumulative score / cumulative opponents' score.
    """
    REQUIRED_COLS = {"prev_match_score", "prev_match_oppo_score"}
    _validate_required_columns(REQUIRED_COLS, data_frame.columns)

    cum_score = (
        data_frame["prev_match_score"].groupby(level=[TEAM_LEVEL, YEAR_LEVEL]).cumsum()
    )
    cum_oppo_score = (
        data_frame["prev_match_oppo_score"]
        .groupby(level=[TEAM_LEVEL, YEAR_LEVEL])
        .cumsum()
    )

    return data_frame.assign(cum_percent=cum_score / cum_oppo_score)


def add_ladder_position(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Append a column for teams' current ladder position."""
    REQUIRED_COLS = INDEX_COLS + ["cum_win_points", "cum_percent"]
    _validate_required_columns(REQUIRED_COLS, data_frame.columns)

    # Pivot to get round-by-round match points and cumulative percent
    ladder_pivot_table = data_frame[
        INDEX_COLS + ["cum_win_points", "cum_percent"]
    ].pivot_table(
        index=["year", "round_number"],
        values=["cum_win_points", "cum_percent"],
        columns="team",
        aggfunc={"cum_win_points": np.sum, "cum_percent": np.mean},
    )

    # To get round-by-round ladder ranks, we sort each round by win points & percent,
    # then save index numbers
    ladder_index = []
    ladder_values = []

    for year_round_idx, round_row in ladder_pivot_table.iterrows():
        sorted_row = round_row.unstack(level=TEAM_LEVEL).sort_values(
            ["cum_win_points", "cum_percent"], ascending=False
        )

        for ladder_idx, team_name in enumerate(sorted_row.index.to_numpy()):
            ladder_index.append(tuple([team_name, *year_round_idx]))
            ladder_values.append(ladder_idx + 1)

    ladder_multi_index = pd.MultiIndex.from_tuples(
        ladder_index, names=tuple(INDEX_COLS)
    )
    ladder_position_col = pd.Series(
        ladder_values, index=ladder_multi_index, name="ladder_position"
    )

    return data_frame.assign(ladder_position=ladder_position_col)


def add_elo_pred_win(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Append a column for whether teams are predicted to win per their elo ratings."""
    REQUIRED_COLS = {"elo_rating", "oppo_elo_rating"}
    _validate_required_columns(REQUIRED_COLS, data_frame.columns)

    is_favoured = (data_frame["elo_rating"] > data_frame["oppo_elo_rating"]).astype(int)
    are_even = (data_frame["elo_rating"] == data_frame["oppo_elo_rating"]).astype(int)

    # Give half point for predicted draws
    predicted_results = is_favoured + (are_even * 0.5)

    return data_frame.assign(elo_pred_win=predicted_results)


def _replace_col_names(at_home: bool):
    team_label = "home" if at_home else "away"
    oppo_label = "away" if at_home else "home"

    return (
        lambda col: col.replace("oppo_", f"{oppo_label}_", 1)
        if re.match(OPPO_REGEX, col)
        else f"{team_label}_{col}"
    )


def _match_data_frame(
    data_frame: pd.DataFrame, match_cols: List[str] = [], at_home: bool = True
) -> pd.DataFrame:
    home_index = "team" if at_home else "oppo_team"
    away_index = "oppo_team" if at_home else "team"
    # We drop oppo stats cols, because we end up with both teams' stats per match
    # when we join home and away teams. We keep 'oppo_team' and add the renamed column
    # to the index for convenience
    oppo_stats_cols = [
        col
        for col in data_frame.columns
        if re.match(OPPO_REGEX, col) and col != "oppo_team"
    ]

    return (
        data_frame.query(f"at_home == {int(at_home)}")
        # We index match rows by home_team, year, round_number
        .rename(columns={home_index: "home_team", away_index: "away_team"})
        .drop(["at_home"] + oppo_stats_cols, axis=1)
        # We add all match cols to the index, because they don't affect the upcoming
        # concat, and it's easier than creating a third data frame for match cols
        .set_index(["home_team", "away_team"] + MATCH_INDEX_COLS + match_cols)
        .rename(columns=_replace_col_names(at_home))
        .sort_index()
    )
