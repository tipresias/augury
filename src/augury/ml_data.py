"""Module for holding model data and returning it in a form useful for ML pipelines."""

from typing import Tuple, Optional, List
from datetime import date

import pandas as pd
from kedro.context import load_context, KedroContext

from augury.types import YearRange
from augury.settings import (
    BASE_DIR,
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
        pipeline: str = "full",
        data_set: str = "model_data",
        train_year_range: YearRange = TRAIN_YEAR_RANGE,
        test_year_range: YearRange = VALIDATION_YEAR_RANGE,
        update_data: bool = False,
        index_cols: List[str] = INDEX_COLS,
        **pipeline_kwargs,
    ) -> None:
        """
        Instantiate an MLData object.

        Params
        ------
        context: Relevant context for loading data sets.
        pipeline: Name of the pipeline to run if the desired data set
            is not available.
        data_set: Name of the data set to load.
        train_year_range: Year range (inclusive, exclusive per `range` function)
            for data to include in training sets.
        test_year_range: Year range (inclusive, exclusive per `range` function)
            for data to include in testing sets.
        update_data: Whether to run the pipeline regardless of the status
            of the data set.
        index_cols: Column names to use for the DataFrame's index.
        pipeline_kwargs: Keyword arguments to pass to context.run when running
            the pipeline.
        """
        self.context = context or load_context(BASE_DIR)
        self.pipeline = pipeline
        self._data_set = data_set
        self._train_year_range = train_year_range
        self._test_year_range = test_year_range
        self.update_data = update_data
        self.index_cols = index_cols
        self._data = None
        self.pipeline_kwargs = pipeline_kwargs

    @property
    def data(self) -> pd.DataFrame:
        """Full data set stored in the given class instance."""
        if self._data is None:
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

        data_train = self.data.loc[
            (slice(None), range(*self.train_year_range), slice(None)), :
        ]

        X_train = self.__X(data_train)
        y_train = self.__y(data_train)

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

        data_test = self.data.loc[
            (slice(None), range(*self.test_year_range), slice(None)), :
        ]
        X_test = self.__X(data_test)
        y_test = self.__y(data_test)

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

        self._data_set = name

    def _load_data(self):
        if self.update_data or not self.context.catalog.exists(self.data_set):
            self.context.run(pipeline_name=self.pipeline, **self.pipeline_kwargs)

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

    @staticmethod
    def __X(data_frame: pd.DataFrame) -> pd.DataFrame:
        labels = [
            "(?:oppo_)?score",
            "(?:oppo_)?(?:team_)?behinds",
            "(?:oppo_)?(?:team_)?goals",
            "(?:oppo_)?margin",
            "(?:oppo_)?result",
        ]
        label_cols = data_frame.filter(regex=f"^{'$|^'.join(labels)}$").columns
        features = data_frame.drop(label_cols, axis=1)

        numeric_features = features.select_dtypes("number").astype(float)
        categorical_features = features.select_dtypes(exclude=["number"])

        # Sorting columns with categorical features first to allow for positional indexing
        # for some data transformations further down the pipeline
        return pd.concat([categorical_features, numeric_features], axis=1)

    @staticmethod
    def __y(data_frame: pd.DataFrame) -> pd.Series:
        TEAM_SCORE_SET = set(["score", "oppo_score"])
        if TEAM_SCORE_SET & set(data_frame.columns) == TEAM_SCORE_SET:
            return data_frame.eval("score - oppo_score").rename("margin")

        HOME_AWAY_SCORE_SET = set(["home_score", "away_score"])
        if HOME_AWAY_SCORE_SET & set(data_frame.columns) == HOME_AWAY_SCORE_SET:
            return data_frame.eval("home_score - away_score").rename("home_margin")

        raise ValueError(
            "Didn't find a valid pair of score columns to calculate margins"
        )
