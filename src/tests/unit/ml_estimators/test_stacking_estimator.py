from unittest import TestCase
from unittest.mock import Mock
import os

import pandas as pd
from faker import Faker
from kedro.context import load_context

from machine_learning.settings import BASE_DIR
from machine_learning.ml_estimators import StackingEstimator


FAKE = Faker()
N_ROWS = 10

get_afltables_stats_df = pd.read_csv(
    os.path.join(BASE_DIR, "src/tests/fixtures/fitzroy_get_afltables_stats.csv")
)
match_results_df = pd.read_csv(
    os.path.join(BASE_DIR, "src/tests/fixtures/fitzroy_match_results.csv")
)
get_afltables_stats_mock = Mock(return_value=get_afltables_stats_df)
match_results_mock = Mock(return_value=match_results_df)


class TestBenchmarkEstimator(TestCase):
    def setUp(self):
        self.model = StackingEstimator()

    def test_pickle_file_compatibility(self):
        context = load_context(BASE_DIR)
        loaded_model = context.catalog.load(self.model.name)
        self.assertIsInstance(loaded_model, StackingEstimator)
