from unittest import TestCase
import os

from kedro.context import load_context

from augury.ml_estimators.base_ml_estimator import BaseMLEstimator
from augury.settings import ML_MODELS, BASE_DIR

PICKLE_FILEPATHS = [
    "src/augury/ml_estimators/bagging_estimator/tipresias_2019.pkl",
    "src/augury/ml_estimators/benchmark_estimator/benchmark_estimator.pkl",
]


class TestMLEstimators(TestCase):
    """Basic spot check for being able to load saved ML estimators"""

    def setUp(self):
        # Need to use production environment for loading model if in CI, because we
        # don't check model files into source control
        kedro_env = (
            "production"
            if os.environ.get("CI") == "true"
            else os.environ.get("PYTHON_ENV")
        )
        self.context = load_context(
            BASE_DIR, start_date="2000-01-01", end_date="2010-12-31", env=kedro_env
        )

    def test_estimator_validity(self):
        for model in ML_MODELS:
            print(model)
            estimator = self.context.catalog.load(model["name"])
            self.assertIsInstance(estimator, BaseMLEstimator)
