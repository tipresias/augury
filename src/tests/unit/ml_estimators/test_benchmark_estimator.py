from unittest import TestCase
import os

from kedro.context import load_context
from faker import Faker

from augury.settings import BASE_DIR
from augury.ml_estimators import BenchmarkEstimator


FAKE = Faker()
N_ROWS = 10


class TestBenchmarkEstimator(TestCase):
    def setUp(self):
        # Need to use production environment for loading model if in CI, because we
        # don't check model files into source control
        kedro_env = (
            "production"
            if os.environ.get("CI") == "true"
            else os.environ.get("PYTHON_ENV")
        )
        context = load_context(BASE_DIR, env=kedro_env)
        self.loaded_model = context.catalog.load(BenchmarkEstimator().name)

    def test_pickle_file_compatibility(self):
        self.assertIsInstance(self.loaded_model, BenchmarkEstimator)
