from unittest import TestCase

from augury.ml_estimators import StackingEstimator
from tests.helpers import KedroContextMixin


class TestStackingEstimator(TestCase, KedroContextMixin):
    def setUp(self):
        context = self.load_context()
        self.loaded_model = context.catalog.load(StackingEstimator().name)

    def test_pickle_file_compatibility(self):
        self.assertIsInstance(self.loaded_model, StackingEstimator)
