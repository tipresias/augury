"""Module for holding model data and returning it in a form useful for ML pipelines."""

from typing import Tuple, Optional, List
from datetime import date

import pandas as pd
from kedro.framework.context import KedroContext
from kedro.framework.session import get_current_session

from augury.types import YearRange
from augury.settings import (
    INDEX_COLS,
    TRAIN_YEAR_RANGE,
    VALIDATION_YEAR_RANGE,
)


END_OF_YEAR = f"{date.today().year}-12-31"


class MLData:
    """Holds model data and returns it in a form useful for ML pipelines."""

    def __init__(
        self,
        context: Optional[KedroContext] = None,
        data_set: str = "full_data",
        train_year_range: YearRange = TRAIN_YEAR_RANGE,
        test_year_range: YearRange = VALIDATION_YEAR_RANGE,
        index_cols: List[str] = INDEX_COLS,
        label_col: str = "margin",
    ) -> None:
        """
        Instantiate an MLData object.

        Params
        ------
        context: Relevant context for loading data sets.
        data_set: Name of the data set to load.
        train_year_range: Year range (inclusive, exclusive per `range` function)
            for data to include in training sets.
        test_year_range: Year range (inclusive, exclusive per `range` function)
            for data to include in testing sets.
        index_cols: Column names to use for the DataFrame's index.
        label_col: Name of the column to use for data labels (i.e. y data set).
        """
        self.context = context
        self._data_set = data_set
        self._train_year_range = train_year_range
        self._test_year_range = test_year_range
        self.index_cols = index_cols
        self.label_col = label_col
        self._data = pd.DataFrame()
        self._X_data = None
        self._y_data = None

        if self.context is None:
            session = get_current_session()
            assert session is not None
            self.context = session.load_context()

    @property
    def data(self) -> pd.DataFrame:
        """Full data set stored in the given class instance."""
        if self._data.empty:
            self._data = self._load_data()

        return self._data

    @property
    def train_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filter data by year to produce training data."""
        if len(self.data.index.names) != 3:
            raise ValueError(
                "The index of the data frame must have 3 levels. The expected indexes "
                "are ['team', 'year', 'round_number'], but the index names are: "
                f"{self.data.index.names}"
            )

        train_year_range = range(*self.train_year_range)
        X_train = self._X.loc[(slice(None), train_year_range, slice(None)), :]
        y_train = self._y.loc[
            (
                slice(None),
                # Series can't slice by range, so we have to convert to a valid
                # slice (i.e. end inclusive, so subtract 1)
                slice(min(train_year_range), max(train_year_range)),
                slice(None),
            )
        ]

        return X_train, y_train

    @property
    def test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filter data by year to produce test data."""
        if len(self.data.index.names) != 3:
            raise ValueError(
                "The index of the data frame must have 3 levels. The expected indexes "
                "are ['team', 'year', 'round_number'], but the index names are: "
                f"{self.data.index.names}"
            )

        test_year_range = range(*self.test_year_range)
        X_test = self._X.loc[(slice(None), test_year_range, slice(None)), :]
        y_test = self._y.loc[
            (
                slice(None),
                # Series can't slice by range, so we have to convert to a valid
                # slice (i.e. end inclusive)
                slice(min(test_year_range), max(test_year_range)),
                slice(None),
            )
        ]

        return X_test, y_test

    @property
    def train_year_range(self) -> YearRange:
        """Range of years for slicing training data."""
        return self._train_year_range

    @train_year_range.setter
    def train_year_range(self, years: YearRange) -> None:
        self._train_year_range = years

    @property
    def test_year_range(self) -> YearRange:
        """Range of years for slicing test data."""
        return self._test_year_range

    @test_year_range.setter
    def test_year_range(self, years: YearRange) -> None:
        self._test_year_range = years

    @property
    def data_set(self) -> str:
        """Name of the associated kedro data set."""
        return self._data_set

    @data_set.setter
    def data_set(self, name: str) -> None:
        if self._data_set != name:
            self._data = None
            self._X_data = None
            self._y_data = None

        self._data_set = name

    def _load_data(self):
        data_frame = pd.DataFrame(self.context.catalog.load(self.data_set))

        # When loading date columns directly from JSON, we need to convert them
        # from string to datetime
        if "date" in list(data_frame.columns) and data_frame["date"].dtype == "object":
            data_frame.loc[:, "date"] = pd.to_datetime(data_frame["date"])

        return (
            data_frame.set_index(self.index_cols, drop=False)
            .rename_axis([None] * len(self.index_cols))
            .sort_index()
        )

    @property
    def _X(self) -> pd.DataFrame:
        if self._X_data is None:
            self._X_data = self._load_X()

        return self._X_data

    def _load_X(self) -> pd.DataFrame:
        labels = [
            "(?:oppo_)?score",
            "(?:oppo_)?(?:team_)?behinds",
            "(?:oppo_)?(?:team_)?goals",
            "(?:oppo_)?margin",
            "(?:oppo_)?result",
        ]
        label_cols = self.data.filter(regex=f"^{'$|^'.join(labels)}$").columns
        features = self.data.drop(label_cols, axis=1)

        numeric_features = features.select_dtypes("number").astype(float)
        categorical_features = features.select_dtypes(exclude=["number"])

        # Sorting columns with categorical features first to allow for positional indexing
        # for some data transformations further down the pipeline.
        # We sort each column group alphabetically to guarantee the same column order
        # as long as the columns are the same.
        return pd.concat(
            [
                categorical_features[sorted(categorical_features.columns)],
                numeric_features[sorted(numeric_features.columns)],
            ],
            axis=1,
        )

    @property
    def _y(self) -> pd.Series:
        if self._y_data is None:
            self._y_data = self._load_y()

        return self._y_data

    def _load_y(self) -> pd.Series:
        return self.data[self.label_col]
