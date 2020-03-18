"""Metric, objective, and loss functions for use with models."""

from typing import Union, Tuple
from functools import partial, update_wrapper
import math

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
import tensorflow as tf
from tensorflow import keras

# For regressors that might try to predict negative values or 0,
# we need a slightly positive minimum to not get errors when calculating logarithms
MIN_LOG_VAL = 1 * 10 ** -10
LOSS = 0
DRAW = 0.5
WIN = 1


def _calculate_team_margin(team_margin, oppo_margin):
    # We want True to be 1 and False to be -1
    team_margin_multiplier = ((team_margin > oppo_margin).astype(int) * 2) - 1

    return (
        pd.Series(
            ((team_margin.abs() + oppo_margin.abs()) / 2) * team_margin_multiplier
        )
        .reindex(team_margin.index)
        .sort_index()
    )


def _calculate_match_accuracy(X, y_true, y_pred):
    """Scikit-learn metric function for calculating tipping accuracy."""
    team_match_data_frame = X.assign(y_true=y_true, y_pred=y_pred)
    home_match_data_frame = team_match_data_frame.query("at_home == 1").sort_index()
    away_match_data_frame = (
        team_match_data_frame.query("at_home == 0")
        .set_index(["oppo_team", "year", "round_number"])
        .rename_axis([None, None, None])
        .sort_index()
    )

    home_margin = _calculate_team_margin(
        home_match_data_frame["y_true"], away_match_data_frame["y_true"]
    )
    home_pred_margin = _calculate_team_margin(
        home_match_data_frame["y_pred"], away_match_data_frame["y_pred"]
    )

    return (
        # Any zero margin (i.e. a draw) is counted as correct per usual tipping rules.
        # Predicted margins should never be zero, but since we don't want to encourage
        # any wayward models, we'll count a predicted margin of zero as incorrect
        ((home_margin >= 0) & (home_pred_margin > 0))
        | ((home_margin <= 0) & (home_pred_margin < 0))
    ).mean()


def create_match_accuracy(X):
    """Return Scikit-learn metric function for calculating tipping accuracy."""
    return update_wrapper(
        partial(_calculate_match_accuracy, X), _calculate_match_accuracy
    )


def match_accuracy_scorer(estimator, X, y):
    """Scikit-learn scorer function for calculating tipping accuracy of an estimator."""
    y_pred = estimator.predict(X)

    return _calculate_match_accuracy(X, y, y_pred)


def _positive_pred(y_pred):
    return np.maximum(y_pred, np.repeat(MIN_LOG_VAL, len(y_pred)))


def _draw_bits(y_pred):
    return 1 + (0.5 * np.log2(_positive_pred(y_pred * (1 - y_pred))))


def _win_bits(y_pred):
    return 1 + np.log2(_positive_pred(y_pred))


def _loss_bits(y_pred):
    return 1 + np.log2(_positive_pred(1 - y_pred))


# Raw bits calculations per http://probabilistic-footy.monash.edu/~footy/about.shtml
def _calculate_bits(y_true, y_pred):
    return np.where(
        y_true == DRAW,
        _draw_bits(y_pred),
        np.where(y_true == WIN, _win_bits(y_pred), _loss_bits(y_pred)),
    )


def bits_scorer(
    estimator: BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    y: pd.Series,
    proba=False,
    n_years=1,
) -> float:
    """Scikit-learn scorer for the bits metric.

    Mostly for use in calls to cross_validate. Calculates a score
    based on the the model's predicted probability of a given result. For this metric,
    higher scores are better.

    We simplify calculations by using Numpy math functions. This has the benefit
    of not require a lot of reshaping based on categorical features, but gives
    final values that deviate a little from what is correct, because this scorer
    calculates bits per team-match combination rather than per match,
    which is how the official bits score will be calculated.

    Params
    ------
    estimator: The estimator being scored.
    X: Model features.
    y: Model labels.
    proba: Whether to use the `predict_proba` method to get predictions.
    """

    try:
        y_pred = estimator.predict_proba(X)[:, -1] if proba else estimator.predict(X)
    # TF/Keras models don't use predict_proba, so for classifiers, we pass proba=True,
    # then rescue and call predict.
    except AttributeError:
        if proba:
            y_pred = estimator.predict(X)[:, -1]
        else:
            raise

    if isinstance(X, pd.DataFrame) and "year" in X.columns:
        n_years = X["year"].drop_duplicates().count()

    # For tipping competitions, bits are summed across the season.
    # We divide by number of seasons for easier comparison with other models.
    # We divide by two to get a rough per-match bits value.
    return _calculate_bits(y, y_pred).sum() / n_years / 2


def _draw_bits_hessian(y_pred):
    return (y_pred ** 2 - y_pred + 0.5) / (
        math.log(2) * y_pred ** 2 * (y_pred - 1) ** 2
    )


def _win_bits_hessian(y_pred):
    return 1 / (math.log(2) * y_pred ** 2)


def _loss_bits_hessian(y_pred):
    return 1 / (math.log(2) * (1 - y_pred) ** 2)


def _bits_hessian(y_true, y_pred):
    return np.where(
        y_true == DRAW,
        _draw_bits_hessian(y_pred),
        np.where(y_true == WIN, _win_bits_hessian(y_pred), _loss_bits_hessian(y_pred)),
    )


def _draw_bits_gradient(y_pred):
    return (y_pred - 0.5) / (math.log(2) * (y_pred - y_pred ** 2))


def _win_bits_gradient(y_pred):
    return -1 / (math.log(2) * y_pred)


def _loss_bits_gradient(y_pred):
    return 1 / (math.log(2) * (1 - y_pred))


def _bits_gradient(y_true, y_pred):
    return np.where(
        y_true == DRAW,
        _draw_bits_gradient(y_pred),
        np.where(
            y_true == WIN, _win_bits_gradient(y_pred), _loss_bits_gradient(y_pred),
        ),
    )


def bits_objective(y_true, y_pred) -> Tuple[np.array, np.array]:
    """Objective function for XGBoost estimators.

    The gradient and hessian formulas are based on the formula for the bits error
    function rather than the bits metric to make the math more consistent
    with other objective and error functions.

    Params
    ------
    y_true [array-like, (n_observations,)]: Data labels.
    y_pred [array-like, (n_observations, n_label_classes)]: Model predictions.
        In the case of binary classification, the shape is (n_observations,)

    Returns
    -------
    gradient, hessian [tuple of array-like, (n_observations * n_classes,)]:
        gradient function is the derivative of the loss function, and hessian function
        is the derivative of the gradient function.
    """
    # Since y_pred can be 1- or 2-dimensional, we should only reshape y_true
    # when the latter is the case.
    y_true_matrix = (
        y_true.reshape(-1, 1) if len(y_true.shape) != len(y_pred.shape) else y_true
    )

    # Sometimes during training, the confidence estimator will get frisky
    # and give a team a 100% chance of winning, which results
    # in some divide-by-zero errors, so we make the maximum just a little less than 1.
    MAX_PROBA = 1 - MIN_LOG_VAL
    normalized_y_pred = np.minimum(y_pred, np.full_like(y_pred, MAX_PROBA))

    return (
        _bits_gradient(y_true_matrix, normalized_y_pred).flatten(),
        _bits_hessian(y_true_matrix, normalized_y_pred).flatten(),
    )


def _bits_error(y_true, y_pred):
    # We adjust bits calculation to make a valid ML error formula such that 0
    # represents a correct prediction, and the further off the prediction
    # the higher the error value.
    return np.where(
        y_true == DRAW,
        -1 * _draw_bits(y_pred),
        np.where(y_true == WIN, 1 - _win_bits(y_pred), 1 + (-1 * _loss_bits(y_pred))),
    )


def bits_metric(y_pred, y_true_matrix) -> Tuple[str, float]:
    """Metric function for internal model evaluation in XGBoost estimators.

    Note that the order of params, per the xgboost documentation, is y_pred, y_true
    as opposed to the usual y_true, y_pred for Scikit-learn metric functions.

    Params
    ------
    y_pred: Model predictions.
    y_true: Data labels.

    Returns
    -------
    Tuple of the metric name and mean bits error.
    """
    y_true = y_true_matrix.get_label()

    return "mean_bits_error", _bits_error(y_true, y_pred).mean()


def _positive_pred_tensor(y_pred):
    return tf.where(
        tf.math.less_equal(y_pred, tf.constant(0.0)), tf.constant(MIN_LOG_VAL), y_pred
    )


def _log2(x):
    return tf.math.divide(
        tf.math.log(_positive_pred_tensor(x)), tf.math.log(tf.constant(2.0))
    )


def _draw_bits_tensor(y_pred):
    return tf.math.add(
        tf.constant(1.0),
        tf.math.scalar_mul(
            tf.constant(0.5),
            _log2(tf.math.multiply(y_pred, tf.math.subtract(tf.constant(1.0), y_pred))),
        ),
    )


def _win_bits_tensor(y_pred):
    return tf.math.add(tf.constant(1.0), _log2(y_pred))


def _loss_bits_tensor(y_pred):
    return tf.math.add(
        tf.constant(1.0), _log2(tf.math.subtract(tf.constant(1.0), y_pred))
    )


# Raw bits calculations per http://probabilistic-footy.monash.edu/~footy/about.shtml
def bits_loss(y_true, y_pred):
    """Loss function for Tensorflow models based on the bits metric."""
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_win = y_pred[:, -1:]

    # We adjust bits calculation to make a valid ML error formula such that 0
    # represents a correct prediction, and the further off the prediction
    # the higher the error value.
    return keras.backend.mean(
        tf.where(
            tf.math.equal(y_true_f, tf.constant(0.5)),
            tf.math.scalar_mul(tf.constant(-1.0), _draw_bits_tensor(y_pred_win)),
            tf.where(
                tf.math.equal(y_true_f, tf.constant(1.0)),
                tf.math.subtract(tf.constant(1.0), _win_bits_tensor(y_pred_win)),
                tf.math.add(
                    tf.constant(1.0),
                    tf.math.scalar_mul(
                        tf.constant(-1.0), _loss_bits_tensor(y_pred_win)
                    ),
                ),
            ),
        )
    )
