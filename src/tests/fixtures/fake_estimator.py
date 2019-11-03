from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from kedro.context import load_context

from machine_learning.ml_estimators import BaseMLEstimator
from machine_learning.ml_data import MLData
from machine_learning.settings import (
    TEAM_NAMES,
    VENUES,
    ROUND_TYPES,
    BASE_DIR,
    INDEX_COLS,
)


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

    def __init__(self, pipeline="fake", max_year=2019):
        super().__init__(pipeline=pipeline)

        self.max_year = max_year

    @property
    def data(self):
        if self._data is None:
            self._data = (
                load_context(
                    BASE_DIR, start_date=self.start_date, end_date=self.end_date
                )
                .run(pipeline_name=self.pipeline)
                .get("data")
                .set_index(INDEX_COLS, drop=False)
                .query("year > 2000")
            )

            max_data_year = self._data["year"].max()

            if self.max_year != max_data_year:
                max_year_diff = self.max_year - max_data_year

                self._data.loc[:, "date"] = self._data["date"].map(
                    lambda dt: dt.replace(year=(dt.year + max_year_diff))
                )
                self._data.loc[:, "year"] = self._data["year"].map(
                    lambda yr: yr + max_year_diff
                )
                self._data.set_index(
                    ["team", "year", "round_number"], drop=False, inplace=True
                )

        return self._data


def pickle_fake_estimator():
    estimator = FakeEstimator()
    data = FakeEstimatorData()

    estimator.fit(*data.train_data())
    estimator.dump()
