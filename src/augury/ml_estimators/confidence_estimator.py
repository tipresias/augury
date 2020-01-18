"""Model for predicting percent confidence that a team will win."""

from typing import Union

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

from augury.sklearn import (
    bits_objective,
    DataFrameConverter,
    ColumnDropper,
    CorrelationSelector,
)
from augury.settings import (
    TEAM_NAMES,
    ROUND_TYPES,
    VENUES,
    CATEGORY_COLS,
    SEED,
)
from .base_ml_estimator import BaseMLEstimator


ELO_MODEL_COLS = [
    "prev_match_oppo_team",
    "oppo_prev_match_oppo_team",
    "prev_match_at_home",
    "oppo_prev_match_at_home",
    "date",
]

PIPELINE = make_pipeline(
    DataFrameConverter(),
    ColumnDropper(cols_to_drop=ELO_MODEL_COLS),
    CorrelationSelector(cols_to_keep=CATEGORY_COLS),
    ColumnTransformer(
        [
            (
                "onehotencoder",
                OneHotEncoder(
                    categories=[TEAM_NAMES, TEAM_NAMES, ROUND_TYPES, VENUES],
                    sparse=False,
                    handle_unknown="ignore",
                ),
                CATEGORY_COLS,
            )
        ],
        remainder=StandardScaler(),
    ),
    XGBClassifier(random_state=SEED, objective=bits_objective),
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

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.array]):
        """Fit estimator to the data."""
        # Binary classification (win vs loss) performs significantly better
        # than multi-class (win, draw, loss), so we'll arbitrarily round draws
        # down to losses and move on with our lives.
        y_enc = y.astype(int)

        if "dataframeconverter__columns" in self.pipeline.get_params().keys():
            self.pipeline.set_params(
                **{
                    "dataframeconverter__columns": X.columns,
                    "dataframeconverter__index": X.index,
                }
            )

        if "correlationselector__labels" in self.pipeline.get_params().keys():
            self.pipeline.set_params(**{"correlationselector__labels": y_enc})

        return super().fit(X, y_enc)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict the probability of each class being the correct label."""
        if "dataframeconverter__columns" in self.pipeline.get_params().keys():
            self.pipeline.set_params(
                **{
                    "dataframeconverter__columns": X.columns,
                    "dataframeconverter__index": X.index,
                }
            )

        return self.pipeline.predict_proba(X)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict the probability of each team winning a given match.

        The purpose of the ConfidenceEstimator is to predict confidence
        rather than classify wins and losses like a typical classifier would.
        """

        return self.predict_proba(X)[:, -1]
