"""Functions for generating dummy data for use in tests."""

from datetime import timedelta

import numpy as np
import pandas as pd

from augury.settings import (
    TEAM_NAMES,
    TEAM_TRANSLATIONS,
)

MATCH_RESULTS_COLS = [
    "date",
    "tz",
    "updated",
    "round",
    "roundname",
    "year",
    "hbehinds",
    "hgoals",
    "hscore",
    "hteam",
    "hteamid",
    "abehinds",
    "agoals",
    "ascore",
    "ateam",
    "ateamid",
    "winner",
    "winnerteamid",
    "is_grand_final",
    "complete",
    "is_final",
    "id",
    "venue",
]


def fake_match_results_data(
    match_data: pd.DataFrame, round_number: int  # pylint: disable=unused-argument
) -> pd.DataFrame:
    """
    Generate dummy data that replicates match results data from the Squiggle API.

    Params
    ------
    row_count: Number of match rows to return

    Returns
    -------
    DataFrame of match results data
    """
    assert (
        len(match_data["season"].drop_duplicates()) == 1
    ), "Match results data is fetched one season at a time."

    return (
        match_data.query("round_number == @round_number")
        .assign(
            updated=lambda df: pd.to_datetime(df["date"]) + timedelta(hours=3),
            tz="+10:00",
            # AFLTables match_results already have a 'round' column,
            # so we have to replace rather than rename.
            round=lambda df: df["round_number"],
            roundname=lambda df: "Round " + df["round_number"].astype(str),
            hteam=lambda df: df["home_team"].map(
                lambda team: TEAM_TRANSLATIONS.get(team, team)
            ),
            ateam=lambda df: df["away_team"].map(
                lambda team: TEAM_TRANSLATIONS.get(team, team)
            ),
            hteamid=lambda df: df["hteam"].map(TEAM_NAMES.index),
            ateamid=lambda df: df["ateam"].map(TEAM_NAMES.index),
            winner=lambda df: np.where(df["margin"] >= 0, df["hteam"], df["ateam"]),
            winnerteamid=lambda df: df["winner"].map(TEAM_NAMES.index),
            is_grand_final=0,
            complete=100,
            is_final=0,
        )
        .astype({"updated": str})
        .reset_index(drop=False)
        .rename(
            columns={
                "index": "id",
                "season": "year",
                "home_behinds": "hbehinds",
                "home_goals": "hgoals",
                "away_behinds": "abehinds",
                "away_goals": "agoals",
                "home_points": "hscore",
                "away_points": "ascore",
            }
        )
    ).loc[:, MATCH_RESULTS_COLS]
