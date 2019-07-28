"""Pipeline nodes for transforming data"""

from typing import Sequence, List, Dict, Any, cast
from functools import reduce

import pandas as pd


def convert_to_data_frame(
    *data: Sequence[List[Dict[str, Any]]]
) -> Sequence[pd.DataFrame]:
    """
    Converts JSON data in the form of a list of dictionaries into a data frame

    Args:
        data (sequence of list of dictionaries): Data received from a JSON data set.

    Returns:
        Sequence of pandas.DataFrame
    """

    if len(data) == 1:
        return pd.DataFrame(data[0])

    return [pd.DataFrame(datum) for datum in data]


def _combine_data_vertically(
    acc_data_frame: pd.DataFrame, curr_data_frame: pd.DataFrame
) -> pd.DataFrame:
    """
    Assumes the following:
        - All data frames have a date column
        - Data frames are sorted by date in ascending order
        - Data frames have all data for a given date (i.e. all matches played
            on a date, not 1 of 3, which would result in missing data)
    """

    max_accumulated_date = acc_data_frame[  # pylint: disable=unused-variable
        "date"
    ].max()
    sliced_current_data_frame = curr_data_frame.query("date > @max_accumulated_date")

    return acc_data_frame.append(sliced_current_data_frame, sort=False)


def combine_data(*data_frames: Sequence[pd.DataFrame], axis=0) -> pd.DataFrame:
    """
    Concatenate data frames from multiple sources into one data frame

    Args:
        data_frames (list of pandas.DataFrame): Data frames to be concatenated.
        axis (0 or 1, defaults to 0): Whether to concatenate by rows (0) or columns (1).

    Returns:
        Concatenated data frame.
    """

    if len(data_frames) == 1:
        return data_frames[0]

    if axis == 0:
        sorted_data_frames = sorted(
            cast(Sequence[pd.DataFrame], data_frames), key=lambda df: df["date"].min()
        )
        return reduce(_combine_data_vertically, sorted_data_frames).fillna(0)

    if axis == 1:
        return pd.concat(data_frames, axis=axis, sort=False).fillna(0)

    raise ValueError(f"Expected axis to be 0 or 1, but recieved {axis}")
