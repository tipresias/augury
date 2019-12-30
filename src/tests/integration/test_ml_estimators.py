from unittest import TestCase

from augury.ml_estimators.base_ml_estimator import BaseMLEstimator
from augury.settings import ML_MODELS
from tests.helpers import KedroContextMixin


class TestMLEstimators(TestCase, KedroContextMixin):
    """Basic spot check for being able to load saved ML estimators"""

    def setUp(self):
        self.context = self._load_context(
            start_date="2000-01-01", end_date="2010-12-31"
        )

    def test_estimator_validity(self):
        for model in ML_MODELS:
            print(model)
            estimator = self.context.catalog.load(model["name"])
            self.assertIsInstance(estimator, BaseMLEstimator)
