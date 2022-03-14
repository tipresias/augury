# pylint: disable=missing-module-docstring, missing-function-docstring
# pylint: disable=missing-class-docstring, redefined-outer-name

from unittest.mock import MagicMock, patch
import os

import joblib
from freezegun import freeze_time
import pytest
from kedro.framework.session import get_current_session

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
MAX_YEAR = YEAR_RANGE[1] - 1


@pytest.fixture
def predictor():
    session = get_current_session()
    assert session is not None

    context = session.load_context()
    context.catalog.load = MagicMock(
        return_value=joblib.load(
            os.path.join(BASE_DIR, "src/tests/fixtures/fake_estimator.pkl")
        )
    )

    predictor = Predictor(YEAR_RANGE, context, PREDICTION_ROUND)
    return predictor


@pytest.mark.parametrize(
    "models,prediction_multiplier", [(FAKE_ML_MODELS, 2), (FAKE_ML_MODELS[1:], 1)]
)
@patch("augury.hooks.ProjectHooks.register_pipelines", {"fake": create_fake_pipeline()})
def test_make_predictions(models, prediction_multiplier, predictor):
    fake_data = FakeEstimatorData(max_year=MAX_YEAR)
    predicted_matches = fake_data.data.query(
        "year == @MAX_YEAR & round_number == @PREDICTION_ROUND"
    )
    predictor._data = fake_data  # pylint: disable=protected-access

    with freeze_time(f"{MAX_YEAR}-06-15"):
        model_predictions = predictor.make_predictions(models)

        assert len(model_predictions) == len(predicted_matches) * prediction_multiplier
        assert set(model_predictions.columns) == set(
            [
                "team",
                "year",
                "round_number",
                "at_home",
                "oppo_team",
                "ml_model",
                "predicted_margin",
            ]
        )

        prediction_years = model_predictions["year"].drop_duplicates()
        assert len(prediction_years) == 1
        prediction_year = prediction_years.iloc[0]
        assert prediction_year == [YEAR_RANGE[0]]
