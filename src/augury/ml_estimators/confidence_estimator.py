"""Model for predicting percent confidence that a team will win."""

from typing import Union

from sklearn.pipeline import make_pipeline, Pipeline
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

from augury.sklearn.metrics import bits_objective
from augury.settings import SEED
from .base_ml_estimator import BaseMLEstimator, BASE_ML_PIPELINE


BEST_PARAMS = {
    "pipeline__correlationselector__threshold": 0.04559726786512616,
    "xgbclassifier__booster": "gbtree",
    "xgbclassifier__colsample_bylevel": 0.8240329295611285,
    "xgbclassifier__colsample_bytree": 0.8683759333432803,
    "xgbclassifier__learning_rate": 0.10367196263253768,
    "xgbclassifier__max_depth": 8,
    "xgbclassifier__n_estimators": 136,
    "xgbclassifier__reg_alpha": 0.0851828929690012,
    "xgbclassifier__reg_lambda": 0.11896695316349301,
    "xgbclassifier__subsample": 0.8195668321302003,
}

PIPELINE = make_pipeline(
    BASE_ML_PIPELINE,
    XGBClassifier(
        random_state=SEED,
        objective=bits_objective,
        use_label_encoder=False,
        verbosity=0,
    ),
).set_params(**BEST_PARAMS)


class ConfidenceEstimator(BaseMLEstimator):
    """Model for predicting percent confidence that a team will win.

    Predictions must be in the form of floats between 0 and 1, representing
    the predicted probability of a given team winning.
    """

    def __init__(
        self, pipeline: Pipeline = PIPELINE, name: str = "confidence_estimator"
    ):
        """Instantiate a ConfidenceEstimator object.

        Params
        ------
        pipeline: Pipeline of Scikit-learn estimators ending in a regressor
            or classifier.
        name: Name of the estimator for reference by Kedro data sets and filenames.
        """
        super().__init__(pipeline=pipeline, name=name)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit estimator to the data."""
        # Binary classification (win vs loss) performs significantly better
        # than multi-class (win, draw, loss), so we'll arbitrarily round draws
        # down to losses and move on with our lives.
        y_enc = y.astype(int)

        self.pipeline.set_params(**{"pipeline__correlationselector__labels": y_enc})

        return super().fit(X, y_enc)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict the probability of each class being the correct label."""
        return self.pipeline.predict_proba(X)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict the probability of each team winning a given match.

        The purpose of the ConfidenceEstimator is to predict confidence
        rather than classify wins and losses like a typical classifier would.
        """

        return self.predict_proba(X)[:, -1]
