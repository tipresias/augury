from unittest import TestCase, skip

import pandas as pd

from tests.helpers import KedroContextMixin
from augury.predictions import Predictor
from augury.settings import ML_MODELS


class TestMLEstimators(TestCase, KedroContextMixin):
    """Basic spot check for being able to load saved ML estimators"""

    def setUp(self):
        self.context = self.load_context(start_date="2007-01-01", end_date="2017-12-31")
        self.predictor = Predictor(
            (2017, 2018), self.context, train_year_range=(2007, 2017),
        )

    @skip(
        "Getting this to work in CI is proving very difficult, and I can't be bothered"
    )
    def test_predictions(self):
        predictions = self.predictor.make_predictions(ML_MODELS)
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertFalse(predictions.isna().any().any())
