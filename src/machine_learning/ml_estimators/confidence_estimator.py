"""Model for predicting percent confidence that a team will win"""

from sklearn.pipeline import make_pipeline, Pipeline

from machine_learning.sklearn import EloRegressor, TeammatchToMatchConverter
from .base_ml_estimator import BaseMLEstimator

PIPELINE = make_pipeline(TeammatchToMatchConverter(), EloRegressor())


class ConfidenceEstimator(BaseMLEstimator):
    """
    Model for predicting percent confidence that a team will win. Predictions must be
    in the form of floats between 0 and 1, representing the predicted probability
    of a given team winning.
    """

    def __init__(
        self, pipeline: Pipeline = PIPELINE, name: str = "confidence_estimator"
    ):
        super().__init__(pipeline=pipeline, name=name)
