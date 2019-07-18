from typing import Callable, List

import pandas as pd
import numpy as np

from machine_learning.data_config import TEAM_TRANSLATIONS, INDEX_COLS


def _validate_required_columns(columns: pd.Index, required_columns: List[str]) -> None:
    req_col_set = {*required_columns}
    column_set = {*columns}

    if req_col_set == req_col_set & column_set:
        return None

    missing_columns = req_col_set - column_set

    raise ValueError(
        f"Required columns {missing_columns} are missing from the data frame."
    )


def _translate_team_name(team_name: str) -> str:
    return TEAM_TRANSLATIONS[team_name] if team_name in TEAM_TRANSLATIONS else team_name


def _translate_team_column(col_name: str) -> Callable[[pd.DataFrame], str]:
    return lambda data_frame: data_frame[col_name].map(_translate_team_name)


def clean_data(betting_data: pd.DataFrame) -> pd.DataFrame:
    """
    Basic data cleaning, translation, and dropping in preparation for ML-specific
    transformations

    Args:
        betting_data (pandas.DataFrame): Raw betting data

    Returns:
        pandas.DataFrame
    """
    return (
        betting_data.rename(columns={"season": "year"})
        .drop(
            [
                "home_win_paid",
                "home_line_paid",
                "away_win_paid",
                "away_line_paid",
                "venue",
                "home_margin",
                "away_margin",
            ],
            axis=1,
        )
        .assign(
            home_team=_translate_team_column("home_team"),
            away_team=_translate_team_column("away_team"),
        )
        .drop("round", axis=1)
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

    _validate_required_columns(match_row_data_frame.columns, REQUIRED_COLS)

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
