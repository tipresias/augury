"""Base ML model and data classes."""

import os
from typing import Optional, Union, Type

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
import pandas as pd
import numpy as np

from augury.sklearn import ColumnDropper, CorrelationSelector
from augury.settings import BASE_DIR, TEAM_NAMES, ROUND_TYPES, VENUES, CATEGORY_COLS
from augury.types import R


ELO_MODEL_COLS = [
    "prev_match_oppo_team",
    "oppo_prev_match_oppo_team",
    "prev_match_at_home",
    "oppo_prev_match_at_home",
    "date",
]

BASE_ML_PIPELINE = make_pipeline(
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
)


class BaseMLEstimator(_BaseComposition, RegressorMixin):
    """Base ML model class."""

    def __init__(
        self, pipeline: Union[Pipeline, BaseEstimator], name: Optional[str] = None,
    ) -> None:
        """Instantiate a BaseMLEstimator object.

        Params
        ------
        pipeline: Pipeline of Scikit-learn estimators ending in a regressor
            or classifier.
        name: Name of the estimator for reference by Kedro data sets and filenames.
        """
        super().__init__()

        self._name = name
        self.pipeline = pipeline

    @property
    def name(self) -> str:
        """Return the name of the model."""
        return self._name or self.__class__.__name__

    @property
    def pickle_filepath(self) -> str:
        """Return the filepath to the model's saved pickle file."""
        return os.path.join(self._default_directory(), f"{self.name}.pkl")

    def dump(self, filepath: str = None) -> None:
        """Save the model as a pickle file."""
        save_path = filepath or self.pickle_filepath

        joblib.dump(self, save_path)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Type[R]:
        """Fit estimator to the data."""
        if self.pipeline is None:
            raise TypeError("pipeline must be a scikit learn estimator but is None")

        self.pipeline.fit(X, y)

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions based on the data input."""
        if self.pipeline is None:
            raise TypeError("pipeline must be a scikit learn estimator but is None")

        return self.pipeline.predict(X)

    @staticmethod
    def _default_directory() -> str:
        return os.path.abspath(os.path.join(BASE_DIR, "data/06_models"))
