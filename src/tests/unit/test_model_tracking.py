# pylint: disable=missing-module-docstring, missing-function-docstring
# pylint: disable=missing-class-docstring

from unittest.mock import patch, MagicMock
import re

from tests.fixtures.fake_estimator import (
    FakeEstimatorData,
    FakeEstimator,
    create_fake_pipeline,
)
from augury.model_tracking import (
    present_model_params,
    IRRELEVANT_PARAM_REGEX,
    BASE_PARAM_VALUE_TYPES,
    start_run,
)
from augury.settings import VALIDATION_YEAR_RANGE

FAKE_ML_MODELS = [{"name": "fake_estimator", "data_set": "fake_data"}]
ML_MODEL_NAME = "fake_estimator"


def test_present_model_params(kedro_session):
    context = kedro_session.load_context()
    estimator = context.catalog.load(ML_MODEL_NAME)
    trackable_params = present_model_params(estimator)

    assert isinstance(trackable_params, dict)

    param_keys = trackable_params.keys()

    assert any((key == "model" for key in param_keys))
    assert all((re.search(r"pipeline|^model$", key) for key in param_keys))
    assert not any((re.search(IRRELEVANT_PARAM_REGEX, key) for key in param_keys))

    param_values = trackable_params.values()
    assert all((isinstance(value, BASE_PARAM_VALUE_TYPES) for value in param_values))


@patch("augury.hooks.ProjectHooks.register_pipelines", {"fake": create_fake_pipeline()})
@patch("augury.model_tracking.mlflow")
def test_start_run(mock_mlflow):
    max_of_year_range = VALIDATION_YEAR_RANGE[1]

    model = FakeEstimator()
    model_data = FakeEstimatorData(
        train_year_range=(max_of_year_range - 2, max_of_year_range),
        max_year=(VALIDATION_YEAR_RANGE[1] - 1),
    )

    mock_mlflow.start_run = MagicMock()
    mock_mlflow.log_params = MagicMock()
    mock_mlflow.log_param = MagicMock()
    mock_mlflow.log_metric = MagicMock()
    mock_mlflow.set_tags = MagicMock()

    start_run(
        [
            (
                model,
                model_data,
                {
                    "tags": {"experiment": "fake_run"},
                    "params": {"experiment_value": "fake"},
                },
            )
        ],
        cv_year_range=(max_of_year_range - 1, max_of_year_range),
    )

    mock_mlflow.start_run.assert_called_once_with(run_name="fake_run_fake")
    mock_mlflow.log_params.assert_called_once()
    mock_mlflow.log_metric.assert_called()
    mock_mlflow.set_tags.assert_called_with(
        {
            "model": "fake_estimator",
            "cv": (max_of_year_range - 1, max_of_year_range),
            "experiment": "fake_run",
        }
    )
