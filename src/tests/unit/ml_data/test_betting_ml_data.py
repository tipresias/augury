from unittest import TestCase
import os

import pandas as pd
from faker import Faker

from machine_learning.ml_data import BettingMLData
from machine_learning.data_import import FootywireDataImporter
from machine_learning.settings import BASE_DIR


TEST_DATA_DIR = os.path.join(BASE_DIR, "src/tests/fixtures")
FAKE = Faker()


class TestBettingMLData(TestCase):
    def setUp(self):
        self.data = BettingMLData(
            train_years=(2015, 2015),
            test_years=(2016, 2016),
            data_readers={
                "betting": (
                    FootywireDataImporter(json_dir=TEST_DATA_DIR).get_betting_odds,
                    {},
                )
            },
        )

    def test_train_data(self):
        X_train, y_train = self.data.train_data()

        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(y_train, pd.Series)

        # Applying StandardScaler to integer columns raises a warning
        self.assertFalse(
            any([X_train[column].dtype == int for column in X_train.columns])
        )

    def test_test_data(self):
        X_test, y_test = self.data.test_data()

        self.assertIsInstance(X_test, pd.DataFrame)
        self.assertIsInstance(y_test, pd.Series)

        # Applying StandardScaler to integer columns raises a warning
        self.assertFalse(
            any([X_test[column].dtype == int for column in X_test.columns])
        )

    def test_train_test_data_compatibility(self):
        X_train, _ = self.data.train_data()
        X_test, _ = self.data.test_data()

        self.assertCountEqual(list(X_train.columns), list(X_test.columns))
