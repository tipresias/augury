from typing import Callable, Union, Set, Sequence

import pandas as pd

from machine_learning.settings import MELBOURNE_TIMEZONE
from machine_learning.data_config import TEAM_TRANSLATIONS


def _parse_dates(data_frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(data_frame["date"]).dt.tz_localize(MELBOURNE_TIMEZONE)


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
        "but the provided columns are:\n"
        f"{data_frame_column_set}"
    )
