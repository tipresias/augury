# pylint: disable=missing-module-docstring, missing-function-docstring
# pylint: disable=missing-class-docstring

from unittest import TestCase
from unittest.mock import Mock, patch, MagicMock
from datetime import date
from typing import List
import json
from faker import Faker
from candystore import CandyStore

from tests.fixtures.fake_estimator import create_fake_pipeline
from tests.fixtures import data_factories
from augury.data_import import match_data
from augury import api
from augury import settings
from augury.types import MLModelDict


FAKE = Faker()
THIS_YEAR = date.today().year
YEAR_RANGE = (2018, 2019)
FAKE_ML_MODELS: List[MLModelDict] = [
    {
        "name": "fake_estimator",
        "data_set": "fake_data",
        "prediction_type": "margin",
        "trained_to": 2018,
    }
]
REQUIRED_MATCH_COLUMNS = {
    "date",
    "year",
    "round_number",
    "home_team",
    "away_team",
    "venue",
    "home_score",
    "away_score",
    "match_id",
}


class TestApi(TestCase):
    # It doesn't matter what data Predictor returns since this method doesn't check
    @patch("augury.api.Predictor.make_predictions")
    @patch("augury.api.settings.ML_MODELS", FAKE_ML_MODELS)
    @patch("augury.api.PIPELINE_NAMES", {"fake_data": "fake"})
    @patch(
        "augury.settings.ProjectContext._get_pipelines",
        MagicMock(return_value={"fake": create_fake_pipeline()}),
    )
    def test_make_predictions(self, mock_make_predictions):
        mock_make_predictions.return_value = CandyStore(seasons=YEAR_RANGE).fixtures()
        response = api.make_predictions(YEAR_RANGE, ml_model_names=["fake_estimator"])

        # Check that it serializes to valid JSON due to potential issues
        # with pd.Timestamp and np.nan values
        self.assertEqual(response, json.loads(json.dumps(response)))

        data = response["data"]

        self.assertIsInstance(data, list)
        self.assertIsInstance(data[0], dict)
        self.assertGreater(len(data[0].keys()), 0)
        mock_make_predictions.assert_called_with(FAKE_ML_MODELS)

        with self.subTest(ml_model_names=None):
            mock_make_predictions.reset_mock()
            api.make_predictions(YEAR_RANGE, ml_model_names=None)
            mock_make_predictions.assert_called_with(FAKE_ML_MODELS)

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
            return_value=CandyStore(seasons=YEAR_RANGE).fixtures()
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

    def test_fetch_match_data(self):
        fake_match_results = CandyStore(seasons=YEAR_RANGE).match_results()
        data_importer = match_data
        data_importer.fetch_match_data = Mock(return_value=fake_match_results)

        response = api.fetch_match_data(
            f"{YEAR_RANGE[0]}-01-01",
            f"{YEAR_RANGE[1]}-12-31",
            data_import=data_importer,
            verbose=0,
        )

        matches = response["data"]

        self.assertEqual(len(matches), len(fake_match_results))

        first_match = matches[0]

        self.assertEqual(
            set(first_match.keys()) & REQUIRED_MATCH_COLUMNS,
            REQUIRED_MATCH_COLUMNS,
        )

        match_years = list({match["year"] for match in matches})
        self.assertEqual(match_years, [YEAR_RANGE[0]])

    def test_fetch_match_results_data(self):
        full_fake_match_results = CandyStore(seasons=1).match_results()
        round_number = FAKE.pyint(1, full_fake_match_results["round_number"].max())
        fake_match_results = data_factories.fake_match_results_data(
            full_fake_match_results, round_number
        )

        data_importer = match_data
        data_importer.fetch_match_results_data = Mock(return_value=fake_match_results)

        response = api.fetch_match_results_data(
            round_number, data_import=data_importer, verbose=0
        )

        match_results = response["data"]

        # It returns all available match results for the round
        self.assertEqual(
            len(match_results),
            len(fake_match_results.query("round == @round_number")),
        )

        required_fields = set(
            [
                "date",
                "year",
                "round_number",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
            ]
        )
        first_match = match_results[0]

        self.assertEqual(required_fields, set(first_match.keys()) & required_fields)

        match_rounds = {result["round_number"] for result in match_results}
        self.assertEqual(match_rounds, set([round_number]))

    def test_fetch_ml_model_info(self):
        response = api.fetch_ml_model_info()

        models = response["data"]

        self.assertEqual(models, settings.ML_MODELS)
