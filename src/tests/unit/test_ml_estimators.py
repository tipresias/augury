from unittest import TestCase

from tests.helpers import KedroContextMixin
from augury.ml_estimators.base_ml_estimator import BaseMLEstimator
from augury.settings import ML_MODELS


class TestMLEstimators(TestCase, KedroContextMixin):
    """Basic spot check for being able to load saved ML estimators"""

    def setUp(self):
        self.context = self.load_context(start_date="2000-01-01", end_date="2010-12-31")

    def test_pickle_file_compatibility(self):
        for model in ML_MODELS:
            with self.subTest(model_name=model["name"]):
                estimator = self.context.catalog.load(model["name"])
                self.assertIsInstance(estimator, BaseMLEstimator)
