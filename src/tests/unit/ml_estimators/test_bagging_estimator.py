from unittest import TestCase

from augury.ml_estimators import BaggingEstimator
from tests.helpers import KedroContextMixin


class TestBaggingEstimator(TestCase, KedroContextMixin):
    def setUp(self):
        context = self.load_context()
        self.loaded_model = context.catalog.load(BaggingEstimator().name)

    def test_pickle_file_compatibility(self):
        self.assertIsInstance(self.loaded_model, BaggingEstimator)
