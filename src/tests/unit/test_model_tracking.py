from unittest import TestCase
from unittest.mock import patch, MagicMock
import re

from kedro.context import load_context

from tests.fixtures.fake_estimator import FakeEstimatorData
from machine_learning.model_tracking import (
    present_model_params,
    IRRELEVANT_PARAM_REGEX,
    BASE_PARAM_VALUE_TYPES,
    start_run,
)
from machine_learning.settings import BASE_DIR, CV_YEAR_RANGE, VALIDATION_YEAR_RANGE


FAKE_ML_MODELS = [{"name": "fake_estimator_model", "data_set": "fake_data"}]


class TestModelTracking(TestCase):
    def setUp(self):
        self.model_name = "fake_estimator_model"

    def test_present_model_params(self):
        estimator = load_context(BASE_DIR).catalog.load(self.model_name)
        trackable_params = present_model_params(estimator)

        self.assertIsInstance(trackable_params, dict)

        param_keys = trackable_params.keys()
        self.assertTrue(all(["pipeline" in key for key in param_keys]))
        self.assertFalse(
            any([re.search(IRRELEVANT_PARAM_REGEX, key) for key in param_keys])
        )

        param_values = trackable_params.values()
        self.assertTrue(
            all([isinstance(value, BASE_PARAM_VALUE_TYPES) for value in param_values])
        )

    @patch("machine_learning.model_tracking.ML_MODELS", FAKE_ML_MODELS)
    @patch("machine_learning.model_tracking.mlflow")
    def test_start_run(self, mock_mlflow):
        mock_mlflow.set_experiment = MagicMock()
        mock_mlflow.start_run = MagicMock()
        mock_mlflow.log_params = MagicMock()
        mock_mlflow.log_metric = MagicMock()
        mock_mlflow.set_tags = MagicMock()

        max_of_year_range = VALIDATION_YEAR_RANGE[1]

        start_run(
            "fake_experiment",
            [self.model_name],
            ml_data=FakeEstimatorData,
            cv_year_range=(max_of_year_range - 1, max_of_year_range),
            train_year_range=(max_of_year_range - 2, max_of_year_range),
            max_year=(VALIDATION_YEAR_RANGE[1] - 1),
        )

        mock_mlflow.set_experiment.assert_called_once()
        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.log_params.assert_called_once()
        mock_mlflow.log_metric.assert_called()
        mock_mlflow.set_tags.assert_called_with(
            {
                "model": "fake_estimator",
                "cv_years": (max_of_year_range - 1, max_of_year_range),
            }
        )
