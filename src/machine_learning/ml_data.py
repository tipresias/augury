"""Module for holding model data and returning it in a form useful for ML pipelines"""

from typing import Tuple, Optional, Callable
from datetime import date

import pandas as pd
from kedro.pipeline import Pipeline

from machine_learning.run import run_pipeline
from machine_learning.pipeline import pipeline as data_pipeline
from machine_learning.types import YearPair


END_OF_YEAR = f"{date.today().year}-12-31"


class MLData:
    """
    Class for holding model data and returning it in a form useful for ML pipelines
    """

    @classmethod
    def class_path(cls):
        return f"{cls.__module__}.{cls.__name__}"

    def __init__(
        self,
        pipeline: Callable[[str, str], Pipeline] = data_pipeline,
        train_years: YearPair = (None, 2015),
        test_years: YearPair = (2016, 2016),
        start_date: str = "1897-01-01",
        end_date: str = str(date.today()),
        round_number: Optional[int] = None,
    ) -> None:
        self._pipeline = pipeline
        self._train_years = train_years
        self._test_years = test_years
        self.start_date = start_date
        self.end_date = end_date
        self.round_number = round_number
        self._data = None

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self._data = run_pipeline(
                self.start_date,
                self.end_date,
                pipeline=self._pipeline,
                round_number=self.round_number,
            ).get("data")

        return self._data

    def train_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filter data by year to produce training data"""

        if len(self.data.index.names) != 3:
            raise ValueError(
                "The index of the data frame must have 3 levels. The expected indexes "
                "are ['team', 'year', 'round_number'], but the index names are: "
                f"{self.data.index.names}"
            )

        data_train = self.data.loc[
            (slice(None), slice(*self.train_years), slice(None)), :
        ]

        X_train = self.__X(data_train)
        y_train = self.__y(data_train)

        return X_train, y_train

    def test_data(self, test_round=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filter data by year to produce test data"""

        if len(self.data.index.names) != 3:
            raise ValueError(
                "The index of the data frame must have 3 levels. The expected indexes "
                "are ['team', 'year', 'round_number'], but the index names are: "
                f"{self.data.index.names}"
            )

        data_test = self.data.loc[
            (slice(None), slice(*self.test_years), slice(test_round, test_round)), :
        ]
        X_test = self.__X(data_test)
        y_test = self.__y(data_test)

        return X_test, y_test

    @property
    def train_years(self) -> YearPair:
        """Range of years for slicing training data"""

        return self._train_years

    @train_years.setter
    def train_years(self, years: YearPair) -> None:
        self._train_years = years

    @property
    def test_years(self) -> YearPair:
        """Range of years for slicing test data"""

        return self._test_years

    @test_years.setter
    def test_years(self, years: YearPair) -> None:
        self._test_years = years

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
        categorical_features = features.select_dtypes(
            # Excluding datetime for now, as it's useful for data processing, but isn't
            # a feature in the models
            exclude=["number", "datetimetz", "datetime"]
        )

        # Sorting columns with categorical features first to allow for positional indexing
        # for some data transformations further down the pipeline
        return pd.concat([categorical_features, numeric_features], axis=1)

    @staticmethod
    def __y(data_frame: pd.DataFrame) -> pd.Series:
        return (data_frame["score"] - data_frame["oppo_score"]).rename("margin")
