"""Estimator class for non-ensemble model pipelines."""

from typing import Union

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import Ridge

from .base_ml_estimator import BaseMLEstimator, BASE_ML_PIPELINE


BEST_PARAMS = {"min_year": 1965}

PIPELINE = make_pipeline(BASE_ML_PIPELINE, Ridge())


class BasicEstimator(BaseMLEstimator):
    """Estimator class for non-ensemble model pipelines."""

    def __init__(
        self,
        pipeline: Union[Pipeline, BaseEstimator] = None,
        name: str = "basic_estimator",
        min_year: int = BEST_PARAMS["min_year"],
    ) -> None:
        """Instantiate a StackingEstimator object.

        Params
        ------
        pipeline: Pipeline of Scikit-learn estimators ending in a regressor
            or classifier.
        name: Name of the estimator for reference by Kedro data sets and filenames.
        min_year: Minimum year for data used in training (inclusive).
        """
        pipeline = PIPELINE if pipeline is None else pipeline
        super().__init__(pipeline, name=name)

        self.min_year = min_year
