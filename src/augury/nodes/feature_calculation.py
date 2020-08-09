"""Collection of functions for performing cross-feature mathematical calculations."""

from typing import List, Sequence, Dict
from functools import partial, reduce, update_wrapper
import itertools
import pandas as pd
import numpy as np

from augury.types import (
    DataFrameTransformer,
    CalculatorPair,
    Calculator,
    DataFrameCalculator,
)
from augury.settings import AVG_SEASON_LENGTH

TEAM_LEVEL = 0
# Varies by season and number of teams, but teams play each other about 1.5 times per season,
# and I found a rolling window of 3 for such aggregations to one of the most predictive
# of match results
ROLLING_OPPO_TEAM_WINDOW = 3
# Kind of an awkward aggregation given the imbalance of number of matches a team plays
# at each venue, but after some experimenting a window of 8 works well. It represents
# a little less than a typical season for home venues, about 2-3 seasons for away Melbourne
# venues, and about 5-6 seasons for most other away venues
ROLLING_VENUE_WINDOW = 8

ROLLING_WINDOWS = {"oppo_team": ROLLING_OPPO_TEAM_WINDOW, "venue": ROLLING_VENUE_WINDOW}


def _calculate_feature_col(
    data_calculator: Calculator, column_sets: List[Sequence[str]]
) -> List[DataFrameCalculator]:
    if len(column_sets) != len(set(column_sets)):
        raise ValueError(
            "Some column sets are duplicated, which will result in duplicate data frame "
            "columns. Make sure each column is calculated once."
        )

    return [data_calculator(column_set) for column_set in column_sets]


def _calculate_features(calculators: List[CalculatorPair], data_frame: pd.DataFrame):
    calculator_func_lists = [
        _calculate_feature_col(calculator, column_sets)
        for calculator, column_sets in calculators
    ]
    calculator_funcs = list(itertools.chain.from_iterable(calculator_func_lists))
    calculated_cols = [calc_func(data_frame) for calc_func in calculator_funcs]

    return pd.concat([data_frame, *calculated_cols], axis=1)


def feature_calculator(calculators: List[CalculatorPair]) -> DataFrameTransformer:
    """Call individual feature-calculation functions."""
    return update_wrapper(
        partial(_calculate_features, calculators), _calculate_features
    )


def rolling_rate_filled_by_expanding_rate(
    groups: pd.DataFrame, rolling_window: int
) -> pd.DataFrame:
    """
    Fill blank values from rolling mean with expanding mean values.

    Params:
    -------
    groups: A data frame produced by pandas' groupby method
    rolling_window: How large a window to use for rolling mean calculations

    Returns:
    --------
    A data frame with rolling mean values, using expanding values
        for the initial window
    """
    expanding_rate = groups.expanding(1).mean()
    rolling_rate = groups.rolling(window=rolling_window).mean()

    # When the original data frame has a multi-index, expanding and rolling produce
    # different index shapes:
    # Group.expanding appends the data frame's original index at the end
    # of the group-key index.
    # Group.rolling converts a data frame's multi-index into a single level
    # with tuple values and appends that at the end of the group-key index.
    # It seems that the easiest way to make the indices compatible is to expand
    # the tuples from rolling into separate index levels to match expanding.
    if expanding_rate.index.names != rolling_rate.index.names:
        rolling_rate.index = pd.MultiIndex.from_tuples(
            [tuple([*value[:-1], *value[-1]]) for value in rolling_rate.index.values],
            names=expanding_rate.index.names,
        )

    return rolling_rate.fillna(expanding_rate)


def _rolling_rate(column: str, data_frame: pd.DataFrame) -> pd.Series:
    if column not in data_frame.columns:
        raise ValueError(
            f"To calculate rolling rate, '{column}' "
            "must be in data frame, but the columns given were "
            f"{data_frame.columns}"
        )

    groups = data_frame[column].groupby(level=TEAM_LEVEL, group_keys=True)

    return (
        rolling_rate_filled_by_expanding_rate(groups, AVG_SEASON_LENGTH)
        .dropna()
        .sort_index()
        .rename(f"rolling_{column}_rate")
    )


def calculate_rolling_rate(column: Sequence[str]) -> DataFrameCalculator:
    """Calculate the rolling mean of a column."""
    if len(column) != 1:
        raise ValueError(
            "Can only calculate one rolling average at a time, but received "
            f"{column}"
        )
    return update_wrapper(partial(_rolling_rate, column[0]), _rolling_rate)


def _rolling_mean_by_dimension(
    column_pair: Sequence[str],
    rolling_windows: Dict[str, int],
    data_frame: pd.DataFrame,
) -> pd.Series:
    dimension_column, metric_column = column_pair
    required_columns = ["team", *column_pair]
    rolling_window = (
        rolling_windows[dimension_column]
        if dimension_column in rolling_windows.keys()
        else AVG_SEASON_LENGTH
    )

    if any([col not in data_frame.columns for col in required_columns]):
        raise ValueError(
            f"To calculate rolling rate, 'team', {dimension_column}, and '{metric_column}' "
            "must be in data frame, but the columns given were "
            f"{data_frame.columns}"
        )

    prev_match_values = (
        data_frame.groupby(["team", dimension_column])[metric_column].shift().fillna(0)
    )
    prev_match_values_label = f"prev_{metric_column}_by_{dimension_column}"

    groups = data_frame.assign(**{prev_match_values_label: prev_match_values}).groupby(
        ["team", dimension_column], group_keys=True
    )[prev_match_values_label]

    return (
        rolling_rate_filled_by_expanding_rate(groups, rolling_window)
        .reset_index(level=[0, 1], drop=True)
        .dropna()
        .sort_index()
        .rename(f"rolling_mean_{metric_column}_by_{dimension_column}")
    )


def calculate_rolling_mean_by_dimension(
    column_pair: Sequence[str], rolling_windows: Dict[str, int] = ROLLING_WINDOWS
) -> DataFrameCalculator:
    """Calculate the rolling mean of a numeric column grouped by a categorical column.

    Note: Be sure not to use 'last_week'/'prev_match' metric columns, because that data
    refers to the previous match's dimension, not the current one, so grouping the metric
    values will result in incorrect aggregations.
    """
    if len(column_pair) != 2:
        raise ValueError(
            "Can only calculate one rolling average at a time, grouped by one dimension "
            f"at a time, but received {column_pair}"
        )

    return update_wrapper(
        partial(_rolling_mean_by_dimension, column_pair, rolling_windows),
        _rolling_mean_by_dimension,
    )


def _division(column_pair: Sequence[str], data_frame: pd.DataFrame) -> pd.Series:
    divisor, dividend = column_pair

    if divisor not in data_frame.columns or dividend not in data_frame.columns:
        raise ValueError(
            f"To calculate division of '{divisor}' by '{dividend}', both "
            "must be in data frame, but the columns given were "
            f"{data_frame.columns}"
        )

    return (
        (data_frame[divisor] / data_frame[dividend])
        # Dividing by 0 results in inf, and I'd rather have it just be 0
        .map(lambda val: 0 if val == np.inf else val).rename(
            f"{divisor}_divided_by_{dividend}"
        )
    )


def calculate_division(column_pair: Sequence[str]) -> DataFrameCalculator:
    """Calculate the first column's values divided by the second's."""
    if len(column_pair) != 2:
        raise ValueError(
            "Can only calculate one column divided by another, but received "
            f"{column_pair}"
        )

    return update_wrapper(partial(_division, column_pair), _division)


def _multiplication(column_pair: Sequence[str], data_frame: pd.DataFrame) -> pd.Series:
    first_col, second_col = column_pair
    if first_col not in data_frame.columns or second_col not in data_frame.columns:
        raise ValueError(
            f"To calculate multiplication of '{first_col}' by '{second_col}', "
            "both must be in data frame, but the columns given were "
            f"{data_frame.columns}"
        )

    return (data_frame[first_col] * data_frame[second_col]).rename(
        f"{first_col}_multiplied_by_{second_col}"
    )


def calculate_multiplication(column_pair: Sequence[str]) -> DataFrameCalculator:
    """Multiply the values of two columns."""
    if len(column_pair) != 2:
        raise ValueError(
            "Can only calculate one column multiplied by another, but received "
            f"{column_pair}"
        )

    return update_wrapper(partial(_multiplication, column_pair), _multiplication)


def _add_columns(
    data_frame: pd.DataFrame, addition_column: pd.Series, column_label: str
):
    if addition_column is None:
        return data_frame.loc[:, column_label]

    return addition_column + data_frame[column_label]


def _addition(columns: Sequence[str], data_frame: pd.DataFrame) -> pd.Series:
    if any([col not in data_frame.columns for col in columns]):
        raise ValueError(
            f"To calculate addition of all columns: {columns}, "
            "all must be in data frame, but the columns given were "
            f"{data_frame.columns}"
        )

    addition_column = reduce(
        update_wrapper(partial(_add_columns, data_frame), _add_columns), columns, None
    )
    column_label = "_plus_".join(columns)

    return addition_column.rename(column_label)


def calculate_addition(columns: Sequence[str]) -> DataFrameCalculator:
    """Add the values of multiple columns."""
    if len(columns) < 2:
        raise ValueError(
            "Must have at least two columns to add together, but received " f"{columns}"
        )

    return update_wrapper(partial(_addition, columns), _addition)
