# pylint: disable=missing-module-docstring, missing-function-docstring
# pylint: disable=missing-class-docstring

from unittest import TestCase
from unittest.mock import MagicMock, patch
import os

import joblib
from freezegun import freeze_time

from tests.helpers import KedroContextMixin
from tests.fixtures.fake_estimator import FakeEstimatorData, create_fake_pipeline
from augury.predictions import Predictor
from augury.settings import BASE_DIR

YEAR_RANGE = (2018, 2019)
PREDICTION_ROUND = 1
FAKE_ML_MODELS = [
    {
        "name": "fake_estimator",
        "data_set": "fake_data",
        "prediction_type": "margin",
        "label_col": "margin",
    },
    {
        "name": "fake_estimator",
        "data_set": "fake_data",
        "prediction_type": "win_probability",
        "label_col": "result",
    },
]


class TestPredictor(TestCase, KedroContextMixin):
    def setUp(self):
        self.context = self.load_context()
        self.context.catalog.load = MagicMock(
            return_value=joblib.load(
                os.path.join(BASE_DIR, "src/tests/fixtures/fake_estimator.pkl")
            )
        )
        self.predictor = Predictor(YEAR_RANGE, self.context, PREDICTION_ROUND)
        self.max_year = YEAR_RANGE[1] - 1

        fake_data = FakeEstimatorData(max_year=self.max_year)
        self.prediction_matches = fake_data.data.query(
            "year == @self.max_year & round_number == @PREDICTION_ROUND"
        )

        self.predictor._data = fake_data  # pylint: disable=protected-access

    @patch(
        "augury.hooks.ProjectHooks.register_pipelines", {"fake": create_fake_pipeline()}
    )
    def test_make_predictions(self):
        with freeze_time(f"{self.max_year}-06-15"):
            model_predictions = self.predictor.make_predictions(FAKE_ML_MODELS)

            self.assertEqual(len(model_predictions), len(self.prediction_matches) * 2)
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
                        "predicted_win_probability",
                    ]
                ),
            )

            prediction_years = model_predictions["year"].drop_duplicates()
            self.assertEqual(len(prediction_years), 1)
            prediction_year = prediction_years.iloc[0]
            self.assertEqual(prediction_year, [YEAR_RANGE[0]])

            with self.subTest("when only one ml_model is given"):
                model_predictions = self.predictor.make_predictions(FAKE_ML_MODELS[1:])

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
                            "predicted_win_probability",
                        ]
                    ),
                )
