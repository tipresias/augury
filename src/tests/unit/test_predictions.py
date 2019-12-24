from unittest import TestCase
from unittest.mock import patch

from freezegun import freeze_time

from tests.fixtures.fake_estimator import FakeEstimatorData
from augury.predictions import Predictor

YEAR_RANGE = (2018, 2019)
PREDICTION_ROUND = 1
FAKE_ML_MODELS = [
    {"name": "fake_estimator", "data_set": "fake_data", "prediction_type": "margin"}
]


class TestPredictor(TestCase):
    def setUp(self):
        self.predictor = Predictor(YEAR_RANGE, PREDICTION_ROUND)
        self.max_year = YEAR_RANGE[1] - 1

        fake_data = FakeEstimatorData(max_year=self.max_year)
        self.prediction_matches = fake_data.data.query(
            "year == @self.max_year & round_number == @PREDICTION_ROUND"
        )

        self.predictor._data = fake_data  # pylint: disable=protected-access

    @patch("augury.predictions.ML_MODELS", FAKE_ML_MODELS)
    def test_make_predictions(self):
        with freeze_time(f"{self.max_year}-06-15"):
            model_predictions = self.predictor.make_predictions(
                ml_model_names=["fake_estimator"]
            )

        self.assertEqual(len(model_predictions), len(self.prediction_matches))

        self.assertEqual(
            set(model_predictions.columns),
            set(
                [
                    "team",
                    "year",
                    "round_number",
                    "at_home",
                    "oppo_team",
                    "ml_model",
                    "predicted_margin",
                    "prediction_type",
                ]
            ),
        )

        prediction_years = model_predictions["year"].drop_duplicates()
        self.assertEqual(len(prediction_years), 1)
        prediction_year = prediction_years.iloc[0]
        self.assertEqual(prediction_year, [YEAR_RANGE[0]])
