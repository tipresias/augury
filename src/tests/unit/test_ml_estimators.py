# pylint: disable=missing-module-docstring, missing-function-docstring
# pylint: disable=missing-class-docstring

import pytest

from augury.ml_estimators.base_ml_estimator import BaseMLEstimator
from augury import settings


@pytest.mark.parametrize(
    "model_name", [ml_model["name"] for ml_model in settings.ML_MODELS]
)
def test_model_pickle_file_compatibility(model_name, kedro_session):
    context = kedro_session.load_context()

    estimator = context.catalog.load(model_name)
    assert isinstance(estimator, BaseMLEstimator)
