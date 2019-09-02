"""Module for data cleaning functions"""

from typing import List

import pandas as pd


def clean_joined_data(data_frames: List[pd.DataFrame]):
    # We need to sort by length (going from longest to shortest), then keeping first
    # duplicated column to make sure we don't lose earlier values of shared columns
    # (e.g. dropping match data's 'date' column in favor of the betting data 'date'
    # column results in lots of NaT values, because betting data only goes back to 2010)
    sorted_data_frames = sorted(data_frames, key=len, reverse=True)
    joined_data_frame = pd.concat(sorted_data_frames, axis=1)
    duplicate_columns = joined_data_frame.columns.duplicated(keep="first")

    return joined_data_frame.loc[:, ~duplicate_columns]
