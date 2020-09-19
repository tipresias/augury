"""Scikit-learn-style transformer classes to put in Scikit-learn pipelines."""

from typing import List, Optional, Type, Union
import re

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

from augury.nodes.base import _validate_required_columns
from augury.types import T


MATCH_COLS = ["date", "venue", "round_type"]
MATCH_INDEX_COLS = ["year", "round_number"]
OPPO_REGEX = re.compile("^oppo_")


class CorrelationSelector(BaseEstimator, TransformerMixin):
    """Transformer for filtering out features that are less correlated with labels."""

    def __init__(
        self,
        cols_to_keep: List[str] = [],
        threshold: Optional[float] = None,
        labels=pd.Series(dtype="object"),
    ) -> None:
        """Instantiate a CorrelationSelector transformer.

        Params
        ------
        cols_to_keep: List of feature names to always keep in the data set.
        threshold: Minimum correlation value (exclusive) for keeping a feature.
        labels: Label values from the training data set for calculating
            correlations.
        """
        self.threshold = threshold
        self.labels = labels
        self._cols_to_keep = cols_to_keep
        self._above_threshold_columns = cols_to_keep

    def transform(self, X: pd.DataFrame, _y=None) -> pd.DataFrame:
        """Filter out features with weak correlation with the labels."""
        return X[self._above_threshold_columns]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Type[T]:
        """Calculate feature/label correlations and save high-correlation features."""
        if not any(self.labels) and y is not None:
            self.labels = y

        assert any(
            self.labels
        ), "Need labels argument for calculating feature correlations."

        data_frame = pd.concat([X, self.labels], axis=1).drop(self.cols_to_keep, axis=1)
        label_correlations = data_frame.corr().fillna(0)[self.labels.name].abs()

        if self.threshold is None:
            correlated_columns = data_frame.columns
        else:
            correlated_columns = data_frame.columns[label_correlations > self.threshold]

        self._above_threshold_columns = self.cols_to_keep + [
            col for col in correlated_columns if col in X.columns
        ]

        return self

    @property
    def cols_to_keep(self) -> List[str]:
        """List columns that never filtered out."""
        return self._cols_to_keep

    @cols_to_keep.setter
    def cols_to_keep(self, cols_to_keep: List[str]) -> None:
        """Set the list of columns to always keep.

        Also resets the overall list of columns to keep to the given list.
        """
        self._cols_to_keep = cols_to_keep
        self._above_threshold_columns = self._cols_to_keep


class TeammatchToMatchConverter(BaseEstimator, TransformerMixin):
    """Transformer for converting data frames to be organised by match."""

    def __init__(self, match_cols=MATCH_COLS):
        """
        Instantiate a TeammatchToMatchConverter transformer.

        Params
        ------
        match_cols (list of strings,
            default=["date", "venue", "round_type"]):
            List of match columns that are team neutral (e.g. round_number, venue).
            These won't be renamed with 'home_' or 'away_' prefixes.
        """
        self.match_cols = match_cols
        self._match_cols = list(set(match_cols + MATCH_INDEX_COLS))

    def fit(self, _X, _y=None):
        """Include for consistency with the Scikit-learn interface."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data from being organised by team-match to match.

        This means that input has two rows per match (one row each
        for home and away teams), and output has one row per match (with separate
        columns for home and away team data).
        """
        self._validate_required_columns(X)

        return (
            pd.concat(
                [
                    self._match_data_frame(X, at_home=True),
                    self._match_data_frame(X, at_home=False),
                ],
                axis=1,
            )
            .reset_index()
            .set_index(["home_team"] + MATCH_INDEX_COLS, drop=False)
            .rename_axis([None] * (len(MATCH_INDEX_COLS) + 1))
            .sort_index()
        )

    def _validate_required_columns(self, data_frame: pd.DataFrame):
        required_cols: List[str] = ["team", "oppo_team", "at_home"] + self._match_cols

        _validate_required_columns(required_cols, data_frame.columns)

    def _match_data_frame(
        self, data_frame: pd.DataFrame, at_home: bool = True
    ) -> pd.DataFrame:
        home_index = "team" if at_home else "oppo_team"
        away_index = "oppo_team" if at_home else "team"
        # We drop oppo stats cols, because we end up with both teams' stats per match
        # when we join home and away teams. We keep 'oppo_team' and add the renamed column
        # to the index for convenience
        oppo_stats_cols = [
            col
            for col in data_frame.columns
            if re.match(OPPO_REGEX, col) and col != "oppo_team"
        ]

        return (
            data_frame.query(f"at_home == {int(at_home)}")
            # We index match rows by home_team, year, round_number
            .rename(columns={home_index: "home_team", away_index: "away_team"})
            .drop(["at_home"] + oppo_stats_cols, axis=1)
            # We add all match cols to the index, because they don't affect the upcoming
            # concat, and it's easier than creating a third data frame for match cols
            .set_index(["home_team", "away_team"] + self._match_cols)
            .rename(columns=self._replace_col_names(at_home))
            .sort_index()
        )

    @staticmethod
    def _replace_col_names(at_home: bool):
        team_label = "home" if at_home else "away"
        oppo_label = "away" if at_home else "home"

        return (
            lambda col: col.replace("oppo_", f"{oppo_label}_", 1)
            if re.match(OPPO_REGEX, col)
            else f"{team_label}_{col}"
        )


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Transformer that drops named columns from data frames."""

    def __init__(self, cols_to_drop: List[str] = []):
        """Instantiate a ColumbnDropper transformer.

        Params
        ------
        cols_to_drop: List of column names to drop.
        """
        self.cols_to_drop = cols_to_drop

    def fit(self, _X, y=None):  # pylint: disable=unused-argument
        """Include for consistency with Scikit-learn interface."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop the given columns from the data."""
        return X.drop(self.cols_to_drop, axis=1, errors="ignore")


class DataFrameConverter(BaseEstimator, TransformerMixin):
    """Transformer that converts numpy arrays into DataFrames with named axes.

    Resulting data frame is assigned named columns and indices per the initial data sets
    passed to fit/predict. This is mostly for cases when classes from packages
    convert DataFrames to numpy arrays without asking, and later transformers depend
    on named indices/columns to work.
    """

    def __init__(
        self,
        columns: Optional[Union[List[str], pd.Index]] = None,
        index: Optional[Union[List[str], pd.Index]] = None,
    ):
        """Instantiate a DataFrameConverter transformer.

        Params
        ------
        columns: List of column names or a pd.Index to assign as columns.
        index: List of row names or a pd.Index to assign as the index.
        """
        self.columns = columns
        self.index = index

    def fit(self, X, y=None):  # pylint: disable=unused-argument
        """Include for consistency with Scikit-learn interface."""
        return self

    def transform(self, X: Union[pd.DataFrame, np.array]):
        """Convert data into a pandas DataFrame with the given columns and index."""
        if self.columns is not None:
            assert X.shape[1] == len(self.columns), (
                f"X must have the same number of columns {X.shape[1]} "
                f"as self.columns {len(self.columns)}."
            )

        if self.index is not None:
            assert X.shape[0] == len(self.index), (
                f"X must have the same number of rows {X.shape[0]} "
                f"as indicated by self.index {len(self.index)}."
            )

        return pd.DataFrame(X, columns=self.columns, index=self.index)
