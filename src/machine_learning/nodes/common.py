"""Pipeline nodes for transforming data"""

from typing import Sequence, List, Dict, Any

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

    return [pd.DataFrame(datum) for datum in data]
