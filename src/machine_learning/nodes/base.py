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
        "the missing columns are:\n"
        f"{required_column_set - column_intersection}"
    )


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


# ID values are converted to floats automatically, making for awkward strings later.
# We want them as strings, because sometimes we have to use player names as replacement
# IDs, and we concatenate multiple ID values to create a unique index.
def _convert_id_to_string(id_label: str) -> Callable:
    return lambda df: df[id_label].astype(int).astype(str)
