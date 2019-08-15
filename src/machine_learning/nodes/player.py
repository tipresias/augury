"""Pipeline nodes for transforming player data"""

from typing import Callable

import pandas as pd

from machine_learning.data_config import TEAM_TRANSLATIONS
from machine_learning.nodes import match
from .base import _parse_dates, _filter_out_dodgy_data, _convert_id_to_string

PLAYER_COL_TRANSLATIONS = {
    "time_on_ground__": "time_on_ground",
    "id": "player_id",
    "round": "round_number",
    "season": "year",
}


UNUSED_PLAYER_COLS = [
    "local_start_time",
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

    cleaned_player_data = (
        player_data.rename(columns=PLAYER_COL_TRANSLATIONS)
        .astype({"year": int})
        .assign(
            # Some player data venues have trailing spaces
            venue=lambda x: x["venue"].str.strip(),
            player_name=lambda x: x["first_name"] + " " + x["surname"],
            player_id=_convert_id_to_string("player_id"),
            home_team=_translate_team_column("home_team"),
            away_team=_translate_team_column("away_team"),
            playing_for=_translate_team_column("playing_for"),
            date=_parse_dates,
        )
        .drop(UNUSED_PLAYER_COLS + ["first_name", "surname", "round_number"], axis=1)
        # Player data match IDs are wrong for recent years.
        # The easiest way to add correct ones is to graft on the IDs
        # from match_results. Also, match_results round_numbers are integers rather than
        # a mix of ints and strings.
        .merge(
            match_data.pipe(match.clean_match_data).loc[
                :, ["date", "venue", "round_number", "match_id"]
            ],
            on=["date", "venue"],
            how="left",
        )
        .pipe(
            _filter_out_dodgy_data(
                duplicate_subset=["year", "round_number", "player_id"]
            )
        )
        .drop("venue", axis=1)
        # brownlow_votes aren't known until the end of the season
        .fillna({"brownlow_votes": 0})
        # Joining on date/venue leaves two duplicates played at M.C.G.
        # on 29-4-1986 & 9-8-1986, but that's an acceptable loss of data
        # and easier than munging team names
        .dropna()
        # Need to add year to ID, because there are some
        # player_id/match_id combos, decades apart, that by chance overlap
        .assign(id=_player_id_col)
        .set_index("id")
        .sort_index()
    )

    return cleaned_player_data


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
