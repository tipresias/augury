# pylint: disable=missing-module-docstring, missing-function-docstring
# pylint: disable=missing-class-docstring

import os
from unittest import TestCase
from unittest.mock import MagicMock

import pandas as pd
from faker import Faker

from augury.ml_data import MLData


RAW_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../fixtures")
)
FAKE = Faker()
MATCH_COUNT_PER_YEAR = 10
YEAR_RANGE = (2016, 2017)
# Need to multiply by two, because we add team & oppo_team row per match
ROW_COUNT = MATCH_COUNT_PER_YEAR * len(range(*YEAR_RANGE)) * 2


class TestMLData(TestCase):
    """Tests for MLData class"""

    def setUp(self):
        self.data = MLData(data_set="fake_data", train_year_range=(2017,))

    def test_train_data(self):
        X_train, y_train = self.data.train_data

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
        X_test, y_test = self.data.test_data

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

        X_train, _ = self.data.train_data
        X_test, _ = self.data.test_data

        self.assertCountEqual(list(X_train.columns), list(X_test.columns))

    def test_data(self):
        self.data._load_data = MagicMock(  # pylint: disable=protected-access
            return_value="dataz"
        )

        self.assertTrue(self.data._data.empty)  # pylint: disable=protected-access
        self.assertEqual(self.data.data, "dataz")

    def test_data_set(self):
        data_set_name = "even_faker_data"
        self.data.data_set = data_set_name

        self.assertIsNone(self.data._data)  # pylint: disable=protected-access
        self.assertEqual(self.data.data_set, data_set_name)
