from unittest import TestCase
from unittest.mock import Mock
import os

import pandas as pd
import numpy as np
from faker import Faker
from kedro.context import load_context
from xgboost import XGBRegressor

from machine_learning.settings import BASE_DIR
from machine_learning.ml_estimators import StackingEstimator
from machine_learning.ml_estimators.sklearn import EloRegressor
from tests.fixtures.data_factories import fake_cleaned_match_data


FAKE = Faker()
N_MATCHES_PER_YEAR = 10
INTERNAL_REGRESSORS = [
    ("meta", XGBRegressor),
    ("xgbregressor", XGBRegressor),
    ("eloregressor", EloRegressor),
]


class TestStackingEstimator(TestCase):
    def setUp(self):
        context = load_context(BASE_DIR)
        self.loaded_model = context.catalog.load(StackingEstimator().name)
        # Starting in 2017 to avoid having to refit the Elo model for the sake
        # of continuous rounds
        self.data = fake_cleaned_match_data(N_MATCHES_PER_YEAR, (2017, 2019)).assign(
            prev_match_oppo_team=lambda df: df["oppo_team"].sample(
                frac=1, replace=False
            ),
            oppo_prev_match_oppo_team=lambda df: df["team"].sample(
                frac=1, replace=False
            ),
            prev_match_at_home=lambda df: np.random.rand(len(df)).round(),
            oppo_prev_match_at_home=lambda df: np.random.rand(len(df)).round(),
        )

    def test_pickle_file_compatibility(self):
        self.assertIsInstance(self.loaded_model, StackingEstimator)

    def test_get_internal_regressor(self):
        for reg_label, reg_class in INTERNAL_REGRESSORS:
            with self.subTest(regressor_name=reg_label):
                regressor = self.loaded_model.get_internal_regressor(
                    regressor_name=reg_label
                )

                self.assertIsInstance(regressor, reg_class)
