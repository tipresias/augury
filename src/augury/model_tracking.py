"""Helper functions for working with MLFlow, particularly the tracking module"""

from typing import Union, List, Tuple, Optional, Dict, Callable, Any
import re

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate
from sklearn.metrics import get_scorer
from baikal import Model
import mlflow
import numpy as np

from augury.ml_data import MLData
from augury.sklearn import year_cv_split, match_accuracy_scorer
from augury.ml_estimators.base_ml_estimator import BaseMLEstimator
from augury.settings import CV_YEAR_RANGE, SEED
from augury.types import YearRange


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
STATIC_PARAMS = [
    "verbosity",
    "verbose",
    "missing$",
    "n_jobs",
    "random_state",
    "missing_values",
    "copy",
    "add_indicator",
]
# We'll use 'nonce' in names of transformers that only exist for running
# a particular experiment in order to avoid polluting param lists
IRRELEVANT_PARAMS = STATIC_TRANSFORMERS + STATIC_COL_PARAMS + STATIC_PARAMS + ["nonce"]
IRRELEVANT_PARAM_REGEX = re.compile("|".join(IRRELEVANT_PARAMS))
BASE_PARAM_VALUE_TYPES = (str, int, float, list)

CV_LABELS = {
    "test_neg_mean_absolute_error": "mean_absolute_error",
    "test_match_accuracy": "match_accuracy",
    "test_bits": "bits",
    "fit_time": "fit_time",
    "score_time": "score_time",
}

DOUBLE_UNDERSCORE = "__"


def score_model(
    model,
    data,
    cv_year_range=CV_YEAR_RANGE,
    scoring={"neg_mean_absolute_error": get_scorer("neg_mean_absolute_error")},
    n_jobs=None,
):
    cv_scoring = {**{"match_accuracy": match_accuracy_scorer}, **scoring}

    data.train_year_range = (max(cv_year_range),)
    X_train, _ = data.train_data

    return cross_validate(
        model,
        *data.train_data,
        cv=year_cv_split(X_train, cv_year_range),
        scoring=cv_scoring,
        n_jobs=n_jobs,
        verbose=2,
    )


def _is_experimental_param(key):
    """
    This is to record any params added as part of an experiment,
    so we don't have to add them by hand each time.
    """

    return "pipeline" not in key and key != "name" and DOUBLE_UNDERSCORE not in key


def _is_relevant_type(value):
    if not isinstance(value, BASE_PARAM_VALUE_TYPES):
        return False

    if isinstance(value, list) and any(
        [not isinstance(element, BASE_PARAM_VALUE_TYPES) for element in value]
    ):
        return False

    return True


def _is_relevant_param(key, value):
    return (
        # 'pipeline' means we're keeping underlying model params rather than params
        # from the wrapping class.
        ("pipeline" in key or _is_experimental_param(key))
        and not re.search(IRRELEVANT_PARAM_REGEX, key)
        # We check the value type to avoid logging higher-level params like the model class instance
        # or top-level pipeline object
        and _is_relevant_type(value)
    )


def present_model_params(model: GenericModel):
    """
    Filter model parameters, so MLFlow only tracks the ones relevant to param tuning
    """

    try:
        model_name = model.name
    except AttributeError:
        model_name = model.model.name

    return {
        "model": model_name,
        **{k: v for k, v in model.get_params().items() if _is_relevant_param(k, v)},
    }


def _track_metric(scores: np.ndarray, metric_name: str, is_negative=False):
    multiplier = -1 if is_negative else 1
    mean_score = (sum(scores) / len(scores)) * multiplier
    mlflow.log_metric(metric_name, mean_score)


def _track_model(
    loaded_model: GenericModel,
    model_data: MLData,
    run_label: str,
    cv_year_range: YearRange,
    scoring: Dict[str, Callable],
    n_jobs=None,
    **run_tags,
) -> Dict[str, Any]:

    with mlflow.start_run():
        mlflow.log_params(present_model_params(loaded_model))
        mlflow.log_param("label", run_label)

        cv_scores = score_model(
            loaded_model,
            model_data,
            cv_year_range=cv_year_range,
            scoring=scoring,
            n_jobs=n_jobs,
        )

        for score_name, metric_name in CV_LABELS.items():
            if score_name in cv_scores.keys():
                _track_metric(
                    cv_scores[score_name],
                    metric_name,
                    is_negative=("neg" in score_name),
                )

        mlflow.set_tags({"model": loaded_model.name, "cv": cv_year_range, **run_tags})

    return {"model": loaded_model.name, **cv_scores}


def start_run(
    ml_models: List[Tuple[GenericModel, MLData, str]],
    cv_year_range=CV_YEAR_RANGE,
    experiment: Optional[str] = None,
    scoring: Dict[str, Callable] = {
        "neg_mean_absolute_error": get_scorer("neg_mean_absolute_error")
    },
    n_jobs=None,
    **run_tags,
) -> List[Dict[str, Any]]:
    """
    Perform cros-validation of models, recording params and metrics
    with mlflow's tracking module.
    """

    if experiment is not None:
        mlflow.set_experiment(experiment)

    return [
        _track_model(
            ml_model,
            model_data,
            run_label,
            cv_year_range=cv_year_range,
            scoring=scoring,
            n_jobs=n_jobs,
            **run_tags,
        )
        for ml_model, model_data, run_label in ml_models
    ]
