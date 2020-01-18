"""Model for predicting percent confidence that a team will win."""

from typing import Union

from sklearn.pipeline import make_pipeline, Pipeline
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

from augury.sklearn import bits_objective
from augury.settings import SEED
from .base_ml_estimator import BaseMLEstimator, BASE_ML_PIPELINE


PIPELINE = make_pipeline(
    BASE_ML_PIPELINE, XGBClassifier(random_state=SEED, objective=bits_objective),
)


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
