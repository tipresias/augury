import os
import warnings
from unittest import TestCase

import pandas as pd
from faker import Faker

from machine_learning.ml_data import MLData


RAW_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../fixtures")
)
FAKE = Faker()
MATCH_COUNT_PER_YEAR = 10
YEAR_RANGE = (2016, 2017)
# Need to multiply by two, because we add team & oppo_team row per match
ROW_COUNT = MATCH_COUNT_PER_YEAR * len(range(*YEAR_RANGE)) * 2

# MLData does a .loc call with all the column names, resulting in a
# warning about passing missing column names to .loc when we run tests, so
# we're ignoring the warnings rather than adding all the columns
warnings.simplefilter("ignore", FutureWarning)


class TestMLData(TestCase):
    """Tests for MLData class"""

    def setUp(self):
        self.data = MLData(pipeline="fake", train_years=(None, 2016))

    def test_train_data(self):
        X_train, y_train = self.data.train_data()

        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(y_train, pd.Series)
        self.assertNotIn("score", X_train.columns)
        self.assertNotIn("oppo_score", X_train.columns)
        self.assertNotIn("goals", X_train.columns)
        self.assertNotIn("team_goals", X_train.columns)
        self.assertNotIn("oppo_team_goals", X_train.columns)
        self.assertNotIn("behinds", X_train.columns)
        self.assertNotIn("team_behinds", X_train.columns)
        self.assertNotIn("oppo_team_behinds", X_train.columns)
        self.assertNotIn("margin", X_train.columns)
        self.assertNotIn("result", X_train.columns)

        # Applying StandardScaler to integer columns raises a warning
        self.assertFalse(
            any([X_train[column].dtype == int for column in X_train.columns])
        )

    def test_test_data(self):
        X_test, y_test = self.data.test_data()

        self.assertIsInstance(X_test, pd.DataFrame)
        self.assertIsInstance(y_test, pd.Series)
        self.assertNotIn("score", X_test.columns)
        self.assertNotIn("oppo_score", X_test.columns)
        self.assertNotIn("goals", X_test.columns)
        self.assertNotIn("team_goals", X_test.columns)
        self.assertNotIn("oppo_team_goals", X_test.columns)
        self.assertNotIn("behinds", X_test.columns)
        self.assertNotIn("team_behinds", X_test.columns)
        self.assertNotIn("oppo_team_behinds", X_test.columns)
        self.assertNotIn("margin", X_test.columns)
        self.assertNotIn("result", X_test.columns)

        # Applying StandardScaler to integer columns raises a warning
        self.assertFalse(
            any([X_test[column].dtype == int for column in X_test.columns])
        )

    def test_train_test_data_compatibility(self):
        self.maxDiff = None

        X_train, _ = self.data.train_data()
        X_test, _ = self.data.test_data()

        self.assertCountEqual(list(X_train.columns), list(X_test.columns))

    @staticmethod
    def __set_valid_index(data_frame):
        return data_frame.set_index(["team", "year", "round_number"])
