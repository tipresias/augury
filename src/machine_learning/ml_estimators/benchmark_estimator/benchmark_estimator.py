"""Class for model trained on all AFL data and its associated data class"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from xgboost import XGBRegressor

from machine_learning.settings import (
    TEAM_NAMES,
    ROUND_TYPES,
    VENUES,
    SEED,
    CATEGORY_COLS,
)
from machine_learning.ml_estimators.sklearn import ColumnDropper
from .. import BaseMLEstimator


np.random.seed(SEED)

PIPELINE = make_pipeline(
    ColumnDropper(cols_to_drop=["prev_match_oppo_team", "prev_match_at_home"]),
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
    """Create pipeline for fitting/predicting with model trained on all AFL data"""

    def __init__(
        self, pipeline: Pipeline = PIPELINE, name: str = "benchmark_estimator"
    ) -> None:
        super().__init__(pipeline=pipeline, name=name)
