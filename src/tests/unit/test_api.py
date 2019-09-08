from unittest import TestCase
from unittest.mock import Mock, patch
from datetime import date

from freezegun import freeze_time

from tests.fixtures.data_factories import fake_fixture_data, fake_raw_match_results_data
from tests.fixtures.fake_estimator import FakeEstimatorData
from machine_learning.data_import import match_data
from machine_learning import api
from machine_learning import settings


THIS_YEAR = date.today().year
YEAR_RANGE = (THIS_YEAR, THIS_YEAR + 1)
PREDICTION_ROUND = 1
N_MATCHES = 5
FAKE_ML_MODELS = [
    {"name": "fake_estimator", "filepath": "src/tests/fixtures/fake_estimator.pkl"}
]


@freeze_time(f"{THIS_YEAR}-06-15")
class TestApi(TestCase):
    @patch("machine_learning.api.ML_MODELS", FAKE_ML_MODELS)
    def test_make_predictions(self):
        max_year = YEAR_RANGE[1] - 1
        fake_data = FakeEstimatorData(max_year=max_year)
        prediction_matches = fake_data.data.query(
            "year == @max_year & round_number == @PREDICTION_ROUND"
        )

        response = api.make_predictions(
            YEAR_RANGE,
            PREDICTION_ROUND,
            data=fake_data,
            ml_model_names="fake_estimator",
            verbose=0,
        )

        predictions = response["data"]

        self.assertEqual(len(predictions), len(prediction_matches))

        first_prediction = predictions[0]

        self.assertEqual(
            set(first_prediction.keys()),
            set(
                [
                    "team",
                    "year",
                    "round_number",
                    "at_home",
                    "oppo_team",
                    "ml_model",
                    "predicted_margin",
                ]
            ),
        )

        prediction_years = list({pred["year"] for pred in predictions})
        self.assertEqual(prediction_years, [YEAR_RANGE[0]])

    def test_fetch_fixture_data(self):
        PROCESSED_FIXTURE_FIELDS = [
            "date",
            "home_team",
            "year",
            "round_number",
            "away_team",
            "round_type",
            "venue",
            "match_id",
        ]

        data_importer = match_data
        data_importer.fetch_fixture_data = Mock(
            return_value=fake_fixture_data(N_MATCHES, YEAR_RANGE, clean=False)
        )

        response = api.fetch_fixture_data(
            f"{THIS_YEAR}-01-01",
            f"{THIS_YEAR}-12-31",
            data_import=data_importer,
            verbose=0,
        )

        matches = response["data"]

        older_matches = [
            match for match in matches if match["date"] < str(date.today())
        ]
        self.assertFalse(any(older_matches))

        first_match = matches[0]

        self.assertEqual(set(first_match.keys()), set(PROCESSED_FIXTURE_FIELDS))

        fixture_years = list({match["year"] for match in matches})
        self.assertEqual(fixture_years, [YEAR_RANGE[0]])

    def test_fetch_match_results_data(self):
        data_importer = match_data
        data_importer.fetch_match_data = Mock(
            return_value=fake_raw_match_results_data(N_MATCHES, YEAR_RANGE)
        )

        response = api.fetch_match_results_data(
            "2019-01-01", "2019-12-31", data_import=data_importer, verbose=0
        )

        matches = response["data"]

        self.assertEqual(len(matches), N_MATCHES)

        first_match = matches[0]

        self.assertEqual(
            set(first_match.keys()),
            set(
                [
                    "date",
                    "year",
                    "round_number",
                    "home_team",
                    "away_team",
                    "venue",
                    "home_score",
                    "away_score",
                    "match_id",
                    "crowd",
                ]
            ),
        )

        match_years = list({match["year"] for match in matches})
        self.assertEqual(match_years, [YEAR_RANGE[0]])

    def test_fetch_ml_model_info(self):
        response = api.fetch_ml_model_info()

        models = response["data"]

        self.assertEqual(models, settings.ML_MODELS)
