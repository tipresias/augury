from unittest import TestCase

from augury.ml_estimators import BenchmarkEstimator


class TestBenchmarkEstimator(TestCase):
    def setUp(self):
        context = self.load_context()
        self.loaded_model = context.catalog.load(BenchmarkEstimator().name)

    def test_pickle_file_compatibility(self):
        self.assertIsInstance(self.loaded_model, BenchmarkEstimator)
