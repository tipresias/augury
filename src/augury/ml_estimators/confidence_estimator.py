"""Model for predicting percent confidence that a team will win."""

from sklearn.pipeline import make_pipeline, Pipeline

from augury.sklearn import EloRegressor, TeammatchToMatchConverter
from .base_ml_estimator import BaseMLEstimator

PIPELINE = make_pipeline(TeammatchToMatchConverter(), EloRegressor())


class ConfidenceEstimator(BaseMLEstimator):
    """Model for predicting percent confidence that a team will win.

    Predictions must be in the form of floats between 0 and 1, representing
    the predicted probability of a given team winning.
    """

    def __init__(
        self, pipeline: Pipeline = PIPELINE, name: str = "confidence_estimator"
    ):
        """Instantiate a ConfidenceEstimator object.

        Params:
            pipeline: Pipeline of Scikit-learn estimators ending in a regressor
                or classifier.
            name: Name of the estimator for reference by Kedro data sets and filenames.
        """
        super().__init__(pipeline=pipeline, name=name)
