from unittest import TestCase
from unittest.mock import Mock, patch
from datetime import date

from tests.fixtures.data_factories import fake_fixture_data, fake_raw_match_results_data
from augury.data_import import match_data
from augury import api
from augury import settings


THIS_YEAR = date.today().year
YEAR_RANGE = (2018, 2019)
N_MATCHES = 5


class TestApi(TestCase):
    # It doesn't matter what data Predictor returns since this method doesn't check
    @patch("augury.api.Predictor.make_predictions")
    def test_make_predictions(self, mock_make_predictions):
        mock_make_predictions.return_value = fake_fixture_data(N_MATCHES, YEAR_RANGE)
        response = api.make_predictions(YEAR_RANGE, ml_model_names=["fake_model"])

        data = response["data"]

        self.assertIsInstance(data, list)
        self.assertIsInstance(data[0], dict)
        self.assertGreater(len(data[0].keys()), 0)
        mock_make_predictions.assert_called_with(
            ml_model_names=["fake_model"], train=False
        )

        with self.subTest(ml_model_names=None):
            mock_make_predictions.reset_mock()
            api.make_predictions(YEAR_RANGE, ml_model_names=None)
            mock_make_predictions.assert_called_with(train=False)

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
            f"{YEAR_RANGE[0]}-01-01",
            f"{YEAR_RANGE[0]}-12-31",
            data_import=data_importer,
            verbose=0,
        )

        matches = response["data"]
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
