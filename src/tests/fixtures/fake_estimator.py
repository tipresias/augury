"""Fixture for dummy estimator and associated data class for use in tests."""

import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from kedro.pipeline import Pipeline, node

from augury.pipelines.nodes import common
from augury.pipelines.match import nodes as match
from augury.ml_estimators.base_ml_estimator import BaseMLEstimator
from augury.sklearn.preprocessing import ColumnDropper
from augury.ml_data import MLData
from augury.settings import (
    TEAM_NAMES,
    VENUES,
    ROUND_TYPES,
    VALIDATION_YEAR_RANGE,
)


# We just using this to set fake data to correct year, so no need to get into
# leap days and such
YEAR_IN_DAYS = 365


CATEGORY_COLS = ["team", "oppo_team", "venue", "round_type"]
PIPELINE = make_pipeline(
    ColumnDropper(cols_to_drop=["date"]),
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
    """Test MLEstimator for use in integration tests."""

    def __init__(self, pipeline=PIPELINE, name="fake_estimator"):
        """Instantiate a FakeEstimator object.

        Params
            pipeline: Scikit-learn pipeline.
            name: Name of the estimator for finding its pickle file and loading it.
        """
        super().__init__(pipeline=pipeline, name=name)


class FakeEstimatorData(MLData):
    """Process data for FakeEstimator."""

    def __init__(
        self, data_set="fake_data", max_year=(VALIDATION_YEAR_RANGE[0] - 1), **kwargs
    ):
        """Instantiate a FakeEstimatorData object.

        Params
            data_set: Name of the data set to load.
            max_year: Maximum year (inclusive) for the data set. This lets us load
                the same data fixture without having worry about whether it has
                the correct range of years for a given test.
        """
        data_kwargs = {
            **{
                "data_set": data_set,
                "train_year_range": (max_year,),
                "test_year_range": (max_year, max_year + 1),
            },
            **kwargs,
        }
        super().__init__(
            **data_kwargs,
        )

        self.max_year = max_year

    @property
    def data(self):
        """Return full data set."""
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


def create_fake_pipeline(*_args, **_kwargs):
    """Create a pipeline for loading and transforming match data for test estimator."""
    return Pipeline(
        [
            node(match.clean_match_data, "fake_match_data", "clean_match_data"),
            node(
                common.convert_match_rows_to_teammatch_rows,
                "clean_match_data",
                "match_data_b",
            ),
            node(match.add_out_of_state, "match_data_b", "match_data_c"),
            node(match.add_travel_distance, "match_data_c", "match_data_d"),
            node(match.add_result, "match_data_d", "match_data_e"),
            node(match.add_margin, "match_data_e", "match_data_f"),
            node(
                match.add_shifted_team_features(
                    shift_columns=[
                        "score",
                        "oppo_score",
                        "result",
                        "margin",
                        "team_goals",
                        "team_behinds",
                    ]
                ),
                "match_data_f",
                "match_data_g",
            ),
            node(match.add_cum_win_points, "match_data_g", "match_data_h"),
            node(match.add_win_streak, "match_data_h", "match_data_i"),
        ]
    )


def pickle_fake_estimator():
    """Save FakeEstimator as a pickle file."""
    estimator = FakeEstimator()
    data = FakeEstimatorData()

    estimator.fit(*data.train_data)
    estimator.dump(filepath="src/tests/fixtures/fake_estimator.pkl")
