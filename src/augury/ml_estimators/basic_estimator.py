"""Estimator class for non-ensemble model pipelines."""

from typing import Union

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import Ridge

from .base_ml_estimator import BaseMLEstimator, BASE_ML_PIPELINE


BEST_PARAMS = {
    "pipeline__correlationselector__threshold": 0.04308980248526492,
    "ridge__alpha": 0.06355835028602363,
}

PIPELINE = make_pipeline(BASE_ML_PIPELINE, Ridge()).set_params(**BEST_PARAMS)


class BasicEstimator(BaseMLEstimator):
    """Estimator class for non-ensemble model pipelines."""

    def __init__(
        self,
        pipeline: Union[Pipeline, BaseEstimator] = None,
        name: str = "basic_estimator",
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
