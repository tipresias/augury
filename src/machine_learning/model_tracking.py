"""Helper functions for working with MLFlow, particularly the tracking module"""

from typing import Union, List, Tuple
import re

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate
from sklearn.metrics import get_scorer
from baikal import Model
import mlflow
import numpy as np

from machine_learning.ml_data import MLData
from machine_learning.sklearn import year_cv_split, match_accuracy_scorer
from machine_learning.ml_estimators.base_ml_estimator import BaseMLEstimator
from machine_learning.settings import CV_YEAR_RANGE, SEED
from machine_learning.types import YearRange


np.random.seed(SEED)

GenericModel = Union[BaseEstimator, BaseMLEstimator, Model]

STATIC_TRANSFORMERS = [
    "onehotencoder",
    "columnselector",
    "columndropper",
    "teammatchtomatchconverter",
    "columntransformer",
    "standardscaler",
]
# cols_to_drop isn't technically static, but is calculated by the transformer
# rather than as a manually-defined argument
STATIC_COL_PARAMS = ["cols_to_keep", "match_cols", "cols_to_drop", "pipeline__steps"]
STATIC_PARAMS = ["verbosity", "verbose", "missing$", "n_jobs", "random_state"]
IRRELEVANT_PARAMS = STATIC_TRANSFORMERS + STATIC_COL_PARAMS + STATIC_PARAMS
IRRELEVANT_PARAM_REGEX = re.compile("|".join(IRRELEVANT_PARAMS))
BASE_PARAM_VALUE_TYPES = (str, int, float, list)

CV_LABELS = {
    "test_neg_mean_absolute_error": "mean_absolute_error",
    "test_match_accuracy": "match_accuracy",
    "fit_time": "fit_time",
    "score_time": "score_time",
}


def _is_relevant_param(key, value):
    return (
        # 'pipeline' means we're keeping underlying model params rather than params
        # from the wrapping class.
        "pipeline" in key
        and not re.search(IRRELEVANT_PARAM_REGEX, key)
        # We check the value type to avoid logging higher-level params like the model class instance
        # or top-level pipeline object
        and isinstance(value, BASE_PARAM_VALUE_TYPES)
    )


def present_model_params(model: GenericModel):
    """
    Filter model parameters, so MLFlow only tracks the ones relevant to param tuning
    """
    return {k: v for k, v in model.get_params().items() if _is_relevant_param(k, v)}


def _track_metric(scores: np.ndarray, metric_name: str, is_negative=False):
    multiplier = -1 if is_negative else 1
    mean_score = (sum(scores) / len(scores)) * multiplier
    mlflow.log_metric(metric_name, mean_score)


def _track_model(
    loaded_model: GenericModel, model_data: MLData, cv_year_range: YearRange
):
    X_train, _ = model_data.train_data

    with mlflow.start_run():
        mlflow.log_params(present_model_params(loaded_model))

        cv_scores = cross_validate(
            loaded_model,
            *model_data.train_data,
            cv=year_cv_split(X_train, cv_year_range),
            scoring={
                "neg_mean_absolute_error": get_scorer("neg_mean_absolute_error"),
                "match_accuracy": match_accuracy_scorer,
            },
            n_jobs=-1
        )

        for score_name, metric_name in CV_LABELS.items():
            _track_metric(
                cv_scores[score_name], metric_name, is_negative=("neg" in score_name)
            )

        mlflow.set_tags({"model": loaded_model.name, "cv": cv_year_range})


def start_run(
    experiment_name: str,
    ml_models: List[Tuple[GenericModel, MLData]],
    cv_year_range=CV_YEAR_RANGE,
):
    mlflow.set_experiment(experiment_name)

    for ml_model, model_data in ml_models:
        _track_model(ml_model, model_data, cv_year_range=cv_year_range)
