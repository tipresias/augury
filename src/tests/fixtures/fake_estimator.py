import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from machine_learning.ml_estimators import BaseMLEstimator
from machine_learning.ml_data import MLData
from machine_learning.settings import (
    TEAM_NAMES,
    VENUES,
    ROUND_TYPES,
)


# We just using this to set fake data to correct year, so no need to get into
# leap days and such
YEAR_IN_DAYS = 365


CATEGORY_COLS = ["team", "oppo_team", "venue", "round_type"]
PIPELINE = make_pipeline(
    ColumnTransformer(
        [
            (
                "onehotencoder",
                OneHotEncoder(
                    categories=[TEAM_NAMES, TEAM_NAMES, VENUES, ROUND_TYPES],
                    sparse=False,
                    handle_unknown="ignore",
                ),
                CATEGORY_COLS,
            )
        ],
        remainder="passthrough",
    ),
    Lasso(),
)


class FakeEstimator(BaseMLEstimator):
    """Create test MLModel for use in integration tests"""

    def __init__(self, pipeline=PIPELINE, name="fake_estimator"):
        super().__init__(pipeline=pipeline, name=name)


class FakeEstimatorData(MLData):
    """Process data for FakeEstimator"""

    def __init__(self, pipeline="fake", data_set="fake_data", max_year=2019, **kwargs):
        super().__init__(
            pipeline=pipeline,
            data_set=data_set,
            train_years=(None, max_year - 1),
            test_years=(max_year, max_year),
            **kwargs,
        )

        self.max_year = max_year

    @property
    def data(self):
        if self._data is None:
            self._data = super().data

            max_data_year = self._data["year"].max()

            # If the date range of the data doesn't line up with the year filters
            # for train/test data, we risk getting empty data sets
            if self.max_year != max_data_year:
                max_year_diff = pd.to_timedelta(
                    [(YEAR_IN_DAYS * (self.max_year - max_data_year))]
                    * len(self._data),
                    unit="days",
                )

                self._data.loc[:, "date"] = self._data["date"] + max_year_diff
                self._data.loc[:, "year"] = self._data["date"].dt.year
                self._data.set_index(
                    ["team", "year", "round_number"], drop=False, inplace=True
                )

        return self._data


def pickle_fake_estimator():
    estimator = FakeEstimator()
    data = FakeEstimatorData()

    estimator.fit(*data.train_data())
    estimator.dump()
