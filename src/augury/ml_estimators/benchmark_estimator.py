"""Class for benchmark model without any fancy ensembling."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from xgboost import XGBRegressor

from augury.settings import (
    TEAM_NAMES,
    ROUND_TYPES,
    VENUES,
    SEED,
    CATEGORY_COLS,
)
from augury.sklearn.preprocessing import ColumnDropper
from .base_ml_estimator import BaseMLEstimator, ELO_MODEL_COLS


np.random.seed(SEED)

PIPELINE = make_pipeline(
    ColumnDropper(cols_to_drop=ELO_MODEL_COLS),
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
    XGBRegressor(objective="reg:squarederror"),
)


class BenchmarkEstimator(BaseMLEstimator):
    """Basic estimator based on a single Scikit-learn pipeline."""

    def __init__(
        self, pipeline: Pipeline = PIPELINE, name: str = "benchmark_estimator"
    ) -> None:
        """Instantiate a BenchmarkEstimator object.

        Params
        ------
        pipeline: Pipeline of Scikit-learn estimators ending in a regressor
            or classifier.
        name: Name of the estimator for reference by Kedro data sets and filenames.
        """
        super().__init__(pipeline=pipeline, name=name)
