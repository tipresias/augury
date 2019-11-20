"""Class for model trained on all AFL data and its associated data class"""

from typing import Optional, Union, Type

from mlxtend.regressor import StackingRegressor
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingRegressor
from xgboost import XGBRegressor

from machine_learning.settings import (
    TEAM_NAMES,
    ROUND_TYPES,
    VENUES,
    CATEGORY_COLS,
    SEED,
)
from machine_learning.ml_estimators.sklearn import (
    CorrelationSelector,
    ColumnDropper,
    TeammatchToMatchConverter,
    EloRegressor,
)
from machine_learning.types import R
from .. import BaseMLEstimator

np.random.seed(SEED)

BEST_ML_PARAMS = {
    "baggingregressor__base_estimator__booster": "dart",
    "baggingregressor__base_estimator__colsample_bylevel": 0.9593085973720467,
    "baggingregressor__base_estimator__colsample_bytree": 0.8366869579732328,
    "baggingregressor__base_estimator__learning_rate": 0.13118764001091077,
    "baggingregressor__base_estimator__max_depth": 6,
    "baggingregressor__base_estimator__n_estimators": 149,
    "baggingregressor__base_estimator__reg_alpha": 0.07296244459829336,
    "baggingregressor__base_estimator__reg_lambda": 0.11334834444556088,
    "baggingregressor__base_estimator__subsample": 0.8285733635843882,
    "baggingregressor__base_estimator__objective": "reg:squarederror",
    "baggingregressor__n_estimators": 7,
    "correlationselector__threshold": 0.030411689885916048,
}

ML_PIPELINE = make_pipeline(
    ColumnDropper(cols_to_drop=["prev_match_oppo_team", "prev_match_at_home"]),
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
    BaggingRegressor(base_estimator=XGBRegressor(seed=SEED)),
).set_params(**BEST_ML_PARAMS)

ELO_PIPELINE = make_pipeline(TeammatchToMatchConverter(), EloRegressor())

META_REGRESSOR = make_pipeline(StandardScaler(), XGBRegressor())

PIPELINE = make_pipeline(
    StackingRegressor(
        regressors=[ML_PIPELINE, ELO_PIPELINE], meta_regressor=META_REGRESSOR
    )
)


class StackingEstimator(BaseMLEstimator):
    """
    Stacked ensemble model with lower-level model predictions feeding
    into a meta estimator.
    """

    def __init__(
        self, pipeline: Pipeline = PIPELINE, name: Optional[str] = "stacking_estimator"
    ) -> None:
        super().__init__(pipeline=pipeline, name=name)

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> Type[R]:
        """Fit estimator to the data"""

        assert (
            self.pipeline is not None
        ), "pipeline must be a scikit learn estimator but is None"

        assert X.index.is_monotonic, (
            "X must be sorted by index values. Otherwise, we risk mismatching rows "
            "being passed from lower estimators to the meta estimator."
        )

        pipeline_params = self.pipeline.get_params().keys()

        if (
            "stackingregressor__pipeline-1__correlationselector__labels"
            in pipeline_params
        ):
            self.pipeline.set_params(
                **{"stackingregressor__pipeline-1__correlationselector__labels": y}
            )

        return super().fit(X, y)
