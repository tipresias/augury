"""Class for model trained on all AFL data and its associated data class"""

from typing import Optional, Union, Type, List

from mlxtend.regressor import StackingRegressor
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
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


NAME_IDX = 0
ELO_MODEL_COLS = [
    "prev_match_oppo_team",
    "oppo_prev_match_oppo_team",
    "prev_match_at_home",
    "oppo_prev_match_at_home",
    "date",
]

ENCODED_CATEGORY_COLS = {
    "team": TEAM_NAMES,
    "oppo_team": ["oppo_" + team_name for team_name in TEAM_NAMES],
    "round_type": ROUND_TYPES,
    "venue": VENUES,
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
    XGBRegressor(objective="reg:squarederror", seed=SEED),
)

ELO_PIPELINE = make_pipeline(TeammatchToMatchConverter(), EloRegressor())

META_REGRESSOR = make_pipeline(
    StandardScaler(), XGBRegressor(objective="reg:squarederror", seed=SEED)
)

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

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray], regressor: str = "meta"
    ) -> pd.DataFrame:
        """
        Transform data using the given pipeline's transformers

        Args:
            X (array-like): Data input for the model
            regressor (str, int): Either the string 'meta' to generate transformed
                data inputs for the meta-regressor. Or the index for a
                first-level regressor pipeline to generate the transformed data input
                for that regressor.
        """

        assert (
            self.pipeline is not None
        ), "pipeline must be a scikit learn estimator but is None"

        if regressor == "meta":
            return self._transform_meta_features(X)

        sub_pipeline = self._get_sub_pipeline(regressor)

        assert (
            sub_pipeline is not None
        ), f"Could not find first-level pipeline with regressor named {regressor}."

        return self._transform_sub_features(X, sub_pipeline)

    def get_internal_regressor(
        self, regressor_name: str = "meta"
    ) -> Optional[BaseEstimator]:
        """Get an internal regressor from the StackingRegressor ensemble"""

        if regressor_name == "meta":
            return self._meta_pipeline[-1]

        sub_pipeline = self._get_sub_pipeline(regressor_name)

        if sub_pipeline is None:
            return None

        return sub_pipeline[-1]

    @property
    def _meta_pipeline(self) -> Pipeline:
        assert (
            self.pipeline is not None
        ), "pipeline must be a scikit learn estimator but is None"

        return self.pipeline[-1].meta_regr_

    def _get_sub_pipeline(self, regressor_name: str) -> Optional[Pipeline]:
        assert (
            self.pipeline is not None
        ), "pipeline must be a scikit learn estimator but is None"

        for regr in self._sub_pipelines:
            if regressor_name in regr.get_params():
                return regr

        return None

    @property
    def _sub_pipelines(self) -> List[Pipeline]:
        assert (
            self.pipeline is not None
        ), "pipeline must be a scikit learn estimator but is None"

        return self.pipeline[-1].regr_

    def _transform_meta_features(self, X):
        # This assumes that the main pipeline doesn't have any relevant transformers
        sub_predictions = np.array([pl.predict(X) for pl in self._sub_pipelines]).T
        # This assumes that the meta regressor is a pipeline with
        # at least one transformer
        transformer_pipeline = self._meta_pipeline[:-1]

        # This assumes that all first-level regressors are pipelines
        feature_names = [regr.steps[-1][NAME_IDX] for regr in self._sub_pipelines]

        return pd.DataFrame(
            transformer_pipeline.transform(sub_predictions), columns=feature_names
        )

    @staticmethod
    def _transform_sub_features(X, sub_pipeline):
        if "correlationselector" in sub_pipeline.get_params():
            # Need to run data through the trained column selector
            # to get the columns actually used by the model
            numeric_columns = (
                sub_pipeline["correlationselector"]
                .transform(X)
                .drop(CATEGORY_COLS, axis=1)
                .columns
            )
        else:
            numeric_columns = X.drop(CATEGORY_COLS, axis=1).columns

        if "columntransformer__onehotencoder" in sub_pipeline.get_params():
            # This assumes that none of the category columns are dropped
            # (they are currently excluded from the ColumnSelector above)
            category_columns = [
                col_val
                for col in CATEGORY_COLS
                for col_val in ENCODED_CATEGORY_COLS[col]
            ]
        else:
            category_columns = CATEGORY_COLS

        feature_names = category_columns + list(numeric_columns)

        # This assumes that all first-level regressors are pipelines with
        # at least one transformer
        transformer_pipeline = sub_pipeline[:-1]

        return pd.DataFrame(transformer_pipeline.transform(X), columns=feature_names)
