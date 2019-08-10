"""Module for functions that add features to data frames via FeatureBuilder.

All functions have the following signature:

Args:
    data_frame (pandas.DataFrame): Data frame to be transformed.

Returns:
    pandas.DataFrame
"""

from typing import List, Tuple, Sequence, Set, Union

from mypy_extensions import TypedDict
import pandas as pd
import numpy as np

from machine_learning.data_config import AVG_SEASON_LENGTH

EloDictionary = TypedDict(
    "EloDictionary",
    {
        "home_away_elo_ratings": List[Tuple[float, float]],
        "current_team_elo_ratings": np.ndarray,
        "year": int,
    },
)

TEAM_LEVEL = 0
YEAR_LEVEL = 1
ROUND_LEVEL = 2
REORDERED_TEAM_LEVEL = 2
REORDERED_YEAR_LEVEL = 0
REORDERED_ROUND_LEVEL = 1
WIN_POINTS = 4
EARTH_RADIUS = 6371

# Constants for ELO calculations
BASE_RATING = 1000
K = 35.6
X = 0.49
M = 130
HOME_GROUND_ADVANTAGE = 9
S = 250
SEASON_CARRYOVER = 0.575


def _validate_required_columns(
    required_columns: Union[Set[str], Sequence[str]],
    data_frame_columns: pd.Index,
    column_name: str,
):
    required_column_set = set(required_columns)
    data_frame_column_set = set(data_frame_columns)
    column_intersection = data_frame_column_set.intersection(required_column_set)

    if column_intersection != required_column_set:
        raise ValueError(
            f"To calculate {column_name}, all required columns must be in the "
            "data frame.\n"
            f"Required columns: {required_column_set}\n"
            f"Provided columns: {data_frame_column_set}"
        )


def add_last_year_brownlow_votes(data_frame: pd.DataFrame):
    """Add column for a player's total brownlow votes from the previous season"""

    required_cols = {"player_id", "year", "brownlow_votes"}
    _validate_required_columns(
        required_cols, data_frame.columns, "last_year_brownlow_votes"
    )

    brownlow_last_year = (
        data_frame[["player_id", "year", "brownlow_votes"]]
        .groupby(["player_id", "year"], group_keys=True)
        .sum()
        # Grouping by player to shift by year
        .groupby(level=0)
        .shift()
        .fillna(0)
        .rename(columns={"brownlow_votes": "last_year_brownlow_votes"})
    )
    return (
        data_frame.drop("brownlow_votes", axis=1)
        .merge(brownlow_last_year, on=["player_id", "year"], how="left")
        .set_index(data_frame.index)
    )


def add_rolling_player_stats(data_frame: pd.DataFrame):
    """Replace players' invidual match stats with rolling averages of those stats"""

    STATS_COLS = [
        "kicks",
        "marks",
        "handballs",
        "goals",
        "behinds",
        "hit_outs",
        "tackles",
        "rebounds",
        "inside_50s",
        "clearances",
        "clangers",
        "frees_for",
        "frees_against",
        "contested_possessions",
        "uncontested_possessions",
        "contested_marks",
        "marks_inside_50",
        "one_percenters",
        "bounces",
        "goal_assists",
        "time_on_ground",
    ]

    required_cols = STATS_COLS + ["player_id"]
    _validate_required_columns(
        required_cols, data_frame.columns, "rolling_player_stats"
    )

    player_data_frame = data_frame.sort_values(["player_id", "year", "round_number"])
    player_groups = (
        player_data_frame[STATS_COLS + ["player_id"]]
        .groupby("player_id", group_keys=False)
        .shift()
        .assign(player_id=player_data_frame["player_id"])
        .fillna(0)
        .groupby("player_id", group_keys=False)
    )

    rolling_stats = player_groups.rolling(window=AVG_SEASON_LENGTH).mean()
    expanding_stats = player_groups.expanding(1).mean()

    player_stats = rolling_stats.fillna(expanding_stats).sort_index()
    rolling_stats_cols = {
        stats_col: f"rolling_prev_match_{stats_col}" for stats_col in STATS_COLS
    }

    return (
        player_data_frame.assign(**player_stats.to_dict("series")).rename(
            columns=rolling_stats_cols
        )
        # Data types get inferred when assigning dictionary columns, which converts
        # 'player_id' to float
        .astype({"player_id": str})
    )


def add_cum_matches_played(data_frame: pd.DataFrame):
    """Add cumulative number of matches each player has played"""

    required_cols = {"player_id"}
    _validate_required_columns(required_cols, data_frame.columns, "cum_matches_played")

    return data_frame.assign(
        cum_matches_played=data_frame.groupby("player_id").cumcount()
    )


def _shift_features(columns: List[str], shift: bool, data_frame: pd.DataFrame):
    if shift:
        columns_to_shift = columns
    else:
        columns_to_shift = [col for col in data_frame.columns if col not in columns]

    _validate_required_columns(
        columns_to_shift, data_frame.columns, "time-shifted columns"
    )

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
