from unittest import TestCase

from kedro.context import load_context

from machine_learning.ml_estimators.base_ml_estimator import BaseMLEstimator
from machine_learning.settings import ML_MODELS, BASE_DIR

PICKLE_FILEPATHS = [
    "src/machine_learning/ml_estimators/bagging_estimator/tipresias_2019.pkl",
    "src/machine_learning/ml_estimators/benchmark_estimator/benchmark_estimator.pkl",
]


class TestMLEstimators(TestCase):
    """Basic spot check for being able to load saved ML estimators"""

    def setUp(self):
        self.context = load_context(
            BASE_DIR, start_date="2000-01-01", end_date="2010-12-31"
        )

    def test_estimator_validity(self):
        for model in ML_MODELS:
            print(model)
            estimator = self.context.catalog.load(model["name"])
            self.assertIsInstance(estimator, BaseMLEstimator)
