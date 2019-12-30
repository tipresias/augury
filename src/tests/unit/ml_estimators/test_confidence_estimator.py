from unittest import TestCase


from augury.ml_estimators import ConfidenceEstimator
from tests.helpers import KedroContextMixin


class TestConfidenceEstimator(TestCase, KedroContextMixin):
    def setUp(self):
        context = self.load_context()
        self.loaded_model = context.catalog.load(ConfidenceEstimator().name)

    def test_pickle_file_compatibility(self):
        self.assertIsInstance(self.loaded_model, ConfidenceEstimator)
