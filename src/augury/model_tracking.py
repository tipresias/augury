"""Helper functions for working with MLFlow, particularly the tracking module."""

from typing import Union, List, Tuple, Dict, Callable, Any
import re
import os

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from augury.ml_data import MLData
from augury.sklearn.model_selection import year_cv_split
from augury.sklearn.metrics import match_accuracy_scorer
from augury.ml_estimators.base_ml_estimator import BaseMLEstimator
from augury.settings import CV_YEAR_RANGE, SEED, BASE_DIR


np.random.seed(SEED)

GenericModel = Union[BaseEstimator, BaseMLEstimator, Pipeline]
SKLearnScorer = Callable[
    [BaseEstimator, Union[pd.DataFrame, np.array], Union[pd.DataFrame, np.array]],
    Union[float, int],
]

STATIC_TRANSFORMERS = [
    "onehotencoder",
    "columnselector",
    "columndropper",
    "teammatchtomatchconverter",
    "columntransformer",
    "standardscaler",
    "dataframeconverter",
]
# cols_to_drop isn't technically static, but is calculated by the transformer
# rather than as a manually-defined argument
STATIC_COL_PARAMS = ["cols_to_keep", "match_cols", "cols_to_drop", "__steps$"]
STATIC_PARAMS = [
    "verbosity",
    "verbose",
    "missing$",
    "n_jobs",
    "random_state",
    "missing_values",
    "copy",
    "add_indicator",
    "__memory$",
    "refit",
    "store_train_meta_features",
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
    model: GenericModel,
    data: MLData,
    cv_year_range=CV_YEAR_RANGE,
    scoring: Dict[str, SKLearnScorer] = {
        "neg_mean_absolute_error": get_scorer("neg_mean_absolute_error"),
        "match_accuracy": match_accuracy_scorer,
    },
    n_jobs=None,
) -> Dict[str, np.array]:
    """
    Perform cross-validation on the given model without logging to mlflow.

    This uses a range of years to create incrementing time-series folds
    for cross-validation rather than random k-folds to avoid data leakage.

    Params
    ------
    model: The model to cross-validate.
    cv_year_range: Year range for generating time-series folds for cross-validation.
    scoring: Any Scikit-learn scorers that can calculate a metric from predictions.
        This is in addition to `match_accuracy`, which is always used.
    n_jobs: Number of processes to use.

    Returns
    -------
    cv_scores: A dictionary whose values are arrays of metrics per Scikit-learn's
        `cross_validate` function.
    """
    train_year_range = data.train_year_range

    assert min(train_year_range) < min(cv_year_range) or len(train_year_range) == 1, (
        "Must have at least one year of data before first test fold. Training data "
        f"only goes back to {min(train_year_range)}, and first test fold is for "
        f"{min(cv_year_range)}"
    )
    assert max(train_year_range) >= max(cv_year_range), (
        "Training data must cover all years used for cross-validation. The last year "
        f"of data is {max(train_year_range)}, but the last test fold is for "
        f"{max(cv_year_range)}"
    )

    X_train, _ = data.train_data

    return cross_validate(
        model,
        *data.train_data,
        cv=year_cv_split(X_train, cv_year_range),
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=2,
        error_score="raise",
    )


def _is_experimental_param(key):
    """
    Record any params added to the base estimator class as part of an experiment.

    This is so we don't have to manually add experimental params to the inclusion list
    each time.
    """
    # Double underscores indicate params of pipeline steps or something similar,
    # and we only want params from the wrapper class
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
        ("pipeline" in key or "^meta_" in key or _is_experimental_param(key))
        and not re.search(IRRELEVANT_PARAM_REGEX, key)
        # We check the value type to avoid logging higher-level params like the model class instance
        # or top-level pipeline object
        and _is_relevant_type(value)
    )


def present_model_params(model: GenericModel):
    """Select relevant model parameters to include in MLFLow tracking data."""
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
    run_info: Dict[str, Any],
    run_tags,
    **score_model_kwargs,
) -> Dict[str, Any]:
    cv_scores = score_model(loaded_model, model_data, **score_model_kwargs,)

    run_tags = run_info.get("tags") or {}
    run_params = run_info.get("params") or {}

    run_experiment = run_tags.get("experiment")
    run_value = run_params.get("experiment_value")
    separator = "" if run_experiment is None or run_value is None else "_"
    run_name = (run_experiment or "") + separator + (run_value or "")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**present_model_params(loaded_model), **run_params})

        for score_name, metric_name in CV_LABELS.items():
            if score_name in cv_scores.keys():
                _track_metric(
                    cv_scores[score_name],
                    metric_name,
                    is_negative=("neg" in score_name),
                )

        mlflow.set_tags(
            {
                "model": loaded_model.name,
                "cv": score_model_kwargs.get("cv_year_range") or CV_YEAR_RANGE,
                **run_tags,
            }
        )

    return {"model": loaded_model.name, **cv_scores}


def start_run(
    ml_models: List[Tuple[GenericModel, MLData, Dict[str, Any]]],
    experiment=None,
    run_tags={},
    **score_model_kwargs,
) -> List[Dict[str, Any]]:
    """Perform cross-validation of models. Record params and metrics with MLFlow."""
    mlflow.set_tracking_uri("sqlite:///" + os.path.join(BASE_DIR, "db/mlflow.db"))

    if experiment is not None:
        mlflow.set_experiment(experiment)

    return [
        _track_model(ml_model, model_data, run_info, run_tags, **score_model_kwargs,)
        for ml_model, model_data, run_info in ml_models
    ]


def graph_tf_model_history(history, metrics: List[str] = []) -> None:
    """Visualize loss and metric values per epoch during Keras model training.

    Params
    ------
    history: Keras model history object.
    metrics: List of metric names that the model tracks in addition to loss.

    Returns
    -------
    None, but displays the generated charts.
    """
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(len(loss))

    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()

    for metric in metrics:
        plt.figure()

        metric_train = history.history[metric]
        metric_val = history.history[f"val_{metric}"]
        plt.plot(epochs, metric_train, "bo", label=f"Training {metric}")
        plt.plot(epochs, metric_val, "b", label=f"Validation {metric}")
        plt.title(f"Training and validation {metric}")
        plt.legend()

    plt.show()
