"""Helper functions for scoring ML models and displaying performance metrics."""

from typing import Union, List, Dict, Callable

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from augury.ml_data import MLData
from augury.sklearn.model_selection import year_cv_split
from augury.sklearn.metrics import match_accuracy_scorer
from augury.ml_estimators.base_ml_estimator import BaseMLEstimator
from augury.settings import CV_YEAR_RANGE, SEED


np.random.seed(SEED)

GenericModel = Union[BaseEstimator, BaseMLEstimator, Pipeline]
SKLearnScorer = Callable[
    [BaseEstimator, Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray]],
    Union[float, int],
]


def score_model(
    model: GenericModel,
    data: MLData,
    cv_year_range=CV_YEAR_RANGE,
    scoring: Dict[str, SKLearnScorer] = {
        "neg_mean_absolute_error": get_scorer("neg_mean_absolute_error"),
        "match_accuracy": match_accuracy_scorer,
    },
    n_jobs=None,
) -> Dict[str, np.ndarray]:
    """
    Perform cross-validation on the given model.

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


def _graph_accuracy_scores(performance_data_frame, sort):
    data = (
        performance_data_frame.sort_values("match_accuracy", ascending=False)
        if sort
        else performance_data_frame
    )

    plt.figure(figsize=(15, 7))
    sns.barplot(
        x="model",
        y="match_accuracy",
        data=data,
    )
    plt.ylim(bottom=0.55)
    plt.title("Model accuracy for cross-validation\n", fontsize=18)
    plt.ylabel("Accuracy", fontsize=14)
    plt.xlabel("", fontsize=14)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12, rotation=90)
    plt.legend(fontsize=14)

    plt.show()


def _graph_mae_scores(performance_data_frame, sort):
    data = (
        performance_data_frame.sort_values("mae", ascending=True)
        if sort
        else performance_data_frame
    )

    plt.figure(figsize=(15, 7))
    sns.barplot(x="model", y="mae", data=data)
    plt.ylim(bottom=20)
    plt.title("Model mean absolute error for cross-validation\n", fontsize=18)
    plt.ylabel("MAE", fontsize=14)
    plt.xlabel("", fontsize=14)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12, rotation=90)
    plt.legend(fontsize=14)

    plt.show()


def graph_cv_model_performance(performance_data_frame, sort=True):
    """Display accuracy and MAE scores for the given of models."""
    _graph_accuracy_scores(performance_data_frame, sort=sort)
    _graph_mae_scores(performance_data_frame, sort=sort)
