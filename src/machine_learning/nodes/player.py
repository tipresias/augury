"""Pipeline nodes for transforming player data"""

from typing import Callable, List, Dict, Union, Tuple
from functools import partial, update_wrapper

import pandas as pd

from machine_learning.settings import (
    TEAM_TRANSLATIONS,
    AVG_SEASON_LENGTH,
    INDEX_COLS,
)
from machine_learning.nodes import match
from .base import (
    _parse_dates,
    _filter_out_dodgy_data,
    _convert_id_to_string,
    _validate_required_columns,
)

PLAYER_COL_TRANSLATIONS = {
    "time_on_ground__": "time_on_ground",
    "id": "player_id",
    "round": "round_number",
    "season": "year",
}


UNUSED_PLAYER_COLS = [
    "attendance",
    "hq1g",
    "hq1b",
    "hq2g",
    "hq2b",
    "hq3g",
    "hq3b",
    "hq4g",
    "hq4b",
    "aq1g",
    "aq1b",
    "aq2g",
    "aq2b",
    "aq3g",
    "aq3b",
    "aq4g",
    "aq4b",
    "jumper_no_",
    "umpire_1",
    "umpire_2",
    "umpire_3",
    "umpire_4",
    "substitute",
    "group_id",
]

PLAYER_STATS_COLS = [
    "rolling_prev_match_kicks",
    "rolling_prev_match_marks",
    "rolling_prev_match_handballs",
    "rolling_prev_match_goals",
    "rolling_prev_match_behinds",
    "rolling_prev_match_hit_outs",
    "rolling_prev_match_tackles",
    "rolling_prev_match_rebounds",
    "rolling_prev_match_inside_50s",
    "rolling_prev_match_clearances",
    "rolling_prev_match_clangers",
    "rolling_prev_match_frees_for",
    "rolling_prev_match_frees_against",
    "rolling_prev_match_contested_possessions",
    "rolling_prev_match_uncontested_possessions",
    "rolling_prev_match_contested_marks",
    "rolling_prev_match_marks_inside_50",
    "rolling_prev_match_one_percenters",
    "rolling_prev_match_bounces",
    "rolling_prev_match_goal_assists",
    "rolling_prev_match_time_on_ground",
    "last_year_brownlow_votes",
]


def _translate_team_name(team_name: str) -> str:
    return TEAM_TRANSLATIONS[team_name] if team_name in TEAM_TRANSLATIONS else team_name


def _translate_team_column(col_name: str) -> Callable[[pd.DataFrame], str]:
    return lambda data_frame: data_frame[col_name].map(_translate_team_name)


def _player_id_col(data_frame: pd.DataFrame) -> pd.DataFrame:
    # Need to add year to ID, because there are some
    # player_id/match_id combos, decades apart, that by chance overlap
    return (
        data_frame["year"].astype(str)
        + "."
        + data_frame["match_id"].astype(str)
        + "."
        + data_frame["player_id"].astype(str)
    )


def clean_player_data(
    player_data: pd.DataFrame, match_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Basic cleaning of raw player data.

    Args:
        player_data (pandas.DataFrame): Raw player data.
        match_data (pandas.DataFrame): Raw match data (required for match_id &
            round_number columns).

    Returns:
        pandas.DataFrame: Clean player data
    """

    cleaned_match_data = (
        # Sometimes the time part of date differs between data sources,
        # so we merge player and match data on date without time.
        # This must happen before making datetimes timezone-aware
        match_data.assign(merge_date=lambda df: pd.to_datetime(df["date"]).dt.date)
        .pipe(match.clean_match_data)
        .loc[:, ["merge_date", "venue", "round_number", "match_id"]]
    )

    return (
        player_data.rename(columns=PLAYER_COL_TRANSLATIONS)
        .astype({"year": int})
        .assign(
            # Sometimes the time part of date differs between data sources,
            # so we merge player and match data on date without time.
            # This must happen before making datetimes timezone-aware
            merge_date=lambda df: pd.to_datetime(df["date"]).dt.date,
            # Some player data venues have trailing spaces
            venue=lambda x: x["venue"].str.strip(),
            player_name=lambda x: x["first_name"] + " " + x["surname"],
            player_id=_convert_id_to_string("player_id"),
            home_team=_translate_team_column("home_team"),
            away_team=_translate_team_column("away_team"),
            playing_for=_translate_team_column("playing_for"),
            date=partial(_parse_dates, time_col="local_start_time"),
        )
        .drop(
            UNUSED_PLAYER_COLS
            + ["first_name", "surname", "round_number", "local_start_time"],
            axis=1,
        )
        # Player data match IDs are wrong for recent years.
        # The easiest way to add correct ones is to graft on the IDs
        # from match_results. Also, match_results round_numbers are integers rather than
        # a mix of ints and strings.
        .merge(cleaned_match_data, on=["merge_date", "venue"], how="left")
        .pipe(
            _filter_out_dodgy_data(
                duplicate_subset=["year", "round_number", "player_id"]
            )
        )
        .drop(["venue", "merge_date"], axis=1)
        # brownlow_votes aren't known until the end of the season
        .fillna({"brownlow_votes": 0})
        # Joining on date/venue leaves two duplicates played at M.C.G.
        # on 29-4-1986 & 9-8-1986, but that's an acceptable loss of data
        # and easier than munging team names
        .dropna()
        .assign(id=_player_id_col)
        .set_index("id")
        .sort_index()
    )


def clean_roster_data(
    roster_data: pd.DataFrame, clean_player_data_frame: pd.DataFrame
) -> pd.DataFrame:
    if not roster_data.any().any():
        return roster_data.assign(player_id=[])

    roster_data_frame = (
        roster_data.assign(date=_parse_dates)
        .rename(columns={"season": "year"})
        .merge(
            clean_player_data_frame[["player_name", "player_id"]],
            on=["player_name"],
            how="left",
        )
        .sort_values("player_id", ascending=False)
        # There are some duplicate player names over the years, so we drop the oldest,
        # hoping that the contemporary player matches the one with the most-recent
        # entry into the AFL. If two players with the same name are playing in the
        # league at the same time, that will likely result in errors
        .drop_duplicates(subset=["player_name"], keep="first")
    )

    # If a player is new to the league, he won't have a player_id per AFL Tables data,
    # so we make one up just using his name
    roster_data_frame["player_id"].fillna(
        roster_data_frame["player_name"], inplace=True
    )

    return roster_data_frame.assign(id=_player_id_col).set_index("id")


def _sort_columns(data_frame: pd.DataFrame) -> pd.DataFrame:
    return data_frame[data_frame.columns.sort_values()]


def _replace_col_names(team_type: str) -> Callable[[str], str]:
    oppo_team_type = "away" if team_type == "home" else "home"

    return lambda col_name: (
        col_name.replace(f"{team_type}_", "").replace(f"{oppo_team_type}_", "oppo_")
    )


def _team_data_frame(data_frame: pd.DataFrame, team_type: str) -> pd.DataFrame:
    return (
        data_frame[data_frame["playing_for"] == data_frame[f"{team_type}_team"]]
        .rename(columns=_replace_col_names(team_type))
        .assign(at_home=1 if team_type == "home" else 0)
        .pipe(_sort_columns)
    )


def convert_player_match_rows_to_player_teammatch_rows(
    data_frame: pd.DataFrame
) -> pd.DataFrame:
    """Stack home & away player data, and add 'oppo_' team columns.

    Args:
        data_frame (pandas.DataFrame): Data frame to be transformed.

    Returns:
        pandas.DataFrame
    """

    REQUIRED_COLS = {
        "playing_for",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "match_id",
    }

    _validate_required_columns(REQUIRED_COLS, data_frame.columns)

    team_dfs = [
        _team_data_frame(data_frame, "home"),
        _team_data_frame(data_frame, "away"),
    ]

    return pd.concat(team_dfs, sort=True).drop(["match_id", "playing_for"], axis=1)


def add_last_year_brownlow_votes(data_frame: pd.DataFrame):
    """Add column for a player's total brownlow votes from the previous season"""

    REQUIRED_COLS = {"player_id", "year", "brownlow_votes"}
    _validate_required_columns(REQUIRED_COLS, data_frame.columns)

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

    REQUIRED_COLS = STATS_COLS + ["player_id"]
    _validate_required_columns(REQUIRED_COLS, data_frame.columns)

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

    REQUIRED_COLS = {"player_id"}
    _validate_required_columns(REQUIRED_COLS, data_frame.columns)

    return data_frame.assign(
        cum_matches_played=data_frame.groupby("player_id").cumcount()
    )


def _aggregations(
    match_stats_cols: List[str], aggregations: List[str] = []
) -> Dict[str, Union[str, List[str]]]:
    player_aggs = {col: aggregations for col in PLAYER_STATS_COLS}
    # Since match stats are the same across player rows, taking the mean
    # is the easiest way to aggregate them
    match_aggs = {col: "mean" for col in match_stats_cols}

    return {**player_aggs, **match_aggs}


def _agg_column_name(match_stats_cols: List[str], column_pair: Tuple[str, str]) -> str:
    column_label, _ = column_pair
    return column_label if column_label in match_stats_cols else "_".join(column_pair)


def _aggregate_player_stats_by_team_match_node(
    player_data_frame: pd.DataFrame, aggregations: List[str] = []
) -> pd.DataFrame:
    REQUIRED_COLS = (
        ["oppo_team", "player_id", "player_name", "date"]
        + PLAYER_STATS_COLS
        + INDEX_COLS
    )

    _validate_required_columns(REQUIRED_COLS, player_data_frame.columns)

    match_stats_cols = [
        col
        for col in player_data_frame.select_dtypes("number")
        # Excluding player stats columns & index columns, which are included in the
        # groupby index and readded to the dataframe later
        if col not in PLAYER_STATS_COLS + INDEX_COLS
    ]

    agg_data_frame = (
        player_data_frame.drop(["player_id", "player_name"], axis=1)
        .sort_values(INDEX_COLS)
        # Adding some non-index columns in the groupby, because it doesn't change
        # the grouping and makes it easier to keep for the final data frame.
        .groupby(INDEX_COLS + ["oppo_team", "date"])
        .aggregate(_aggregations(match_stats_cols, aggregations=aggregations))
    )

    agg_data_frame.columns = [
        _agg_column_name(match_stats_cols, column_pair)
        for column_pair in agg_data_frame.columns.values
    ]

    # Various finals matches have been draws and replayed,
    # and sometimes home/away is switched requiring us to drop duplicates
    # at the end.
    # This eliminates some matches from Round 15 in 1897, because they
    # played some sort of round-robin tournament for finals, but I'm
    # not too worried about the loss of that data.
    return (
        agg_data_frame.dropna()
        .reset_index()
        .sort_values("date")
        .drop_duplicates(subset=INDEX_COLS, keep="last")
        .astype({match_col: int for match_col in match_stats_cols})
        .set_index(INDEX_COLS, drop=False)
        .rename_axis([None] * len(INDEX_COLS))
        .sort_index()
    )


def aggregate_player_stats_by_team_match(
    aggregations: List[str]
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Perform aggregations to turn player-match data into team-match data."""

    return update_wrapper(
        partial(_aggregate_player_stats_by_team_match_node, aggregations=aggregations),
        _aggregate_player_stats_by_team_match_node,
    )
