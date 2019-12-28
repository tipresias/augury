"""Class for model trained on all AFL data and its associated data class"""

from typing import Optional, Union, Type

from baikal import Input, Model
from baikal.steps import Stack
from baikal.steps.merge import Concatenate
from baikal.sklearn import SKLearnWrapper
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from augury.steps import (
    StandardScalerStep,
    OneHotEncoderStep,
    ColumnSelectorStep,
    XGBRegressorStep,
    CorrelationSelectorStep,
    ColumnDropperStep,
    TeammatchToMatchConverterStep,
    EloRegressorStep,
)
from augury.settings import (
    TEAM_NAMES,
    ROUND_TYPES,
    VENUES,
    CATEGORY_COLS,
    SEED,
)
from augury.sklearn import year_chunk_cv_split
from augury.types import R
from .base_ml_estimator import BaseMLEstimator

np.random.seed(SEED)


ELO_MODEL_COLS = [
    "prev_match_oppo_team",
    "oppo_prev_match_oppo_team",
    "prev_match_at_home",
    "oppo_prev_match_at_home",
    "date",
]


def _build_pipeline(sub_model=None):
    yt = Input("yt")
    X = Input("X")

    X_trans = X if sub_model is None else sub_model(X)

    z_ml = ColumnDropperStep(cols_to_drop=ELO_MODEL_COLS, name="columndropper_elo")(
        X_trans
    )
    z_ml = CorrelationSelectorStep(
        cols_to_keep=CATEGORY_COLS, name="correlationselector"
    )(z_ml, yt)

    z_cat = ColumnSelectorStep(cols=CATEGORY_COLS, name="columnselector")(z_ml)
    z_cat = OneHotEncoderStep(
        categories=[TEAM_NAMES, TEAM_NAMES, ROUND_TYPES, VENUES],
        sparse=False,
        handle_unknown="ignore",
        name="onehotencoder",
    )(z_cat)

    z_num = ColumnDropperStep(cols_to_drop=CATEGORY_COLS, name="columndropper_cat")(
        z_ml
    )
    z_num = StandardScalerStep(name="standardscaler_sub")(z_num)

    ml_features = Concatenate(name="concatenate")([z_cat, z_num])
    y1 = XGBRegressorStep(
        objective="reg:squarederror", random_state=SEED, name="xgbregressor_sub"
    )(ml_features, yt)

    z_elo = TeammatchToMatchConverterStep(name="teammatchtomatchconverter")(X_trans)
    y2 = EloRegressorStep(name="eloregressor")(z_elo)

    ensemble_features = Stack(name="stack")([y1, y2])
    z = StandardScalerStep(name="standardscaler_meta")(ensemble_features)
    y = XGBRegressorStep(
        objective="reg:squarederror", random_state=SEED, name="xgbregressor_meta"
    )(z, yt)

    return Model(X, y, yt)


class StackingEstimator(BaseMLEstimator):
    """
    Stacked ensemble model with lower-level model predictions feeding
    into a meta estimator.
    """

    def __init__(
        self,
        # Need to use SKLearnWrapper for this to work with Scikit-learn
        # cross-validation
        # pipeline: Union[Model, SKLearnWrapper] = PIPELINE,
        sub_pipelines=[],
        meta_pipeline=None,
        cv=5,
        name: Optional[str] = "stacking_estimator",
    ) -> None:
        super().__init__(name=name)
        self._name = name
        self.sub_pipelines = sub_pipelines
        self.meta_pipeline = meta_pipeline
        self.cv = cv

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> Type[R]:
        """Fit estimator to the data"""

        assert X.index.is_monotonic, (
            "X must be sorted by index values. Otherwise, we risk mismatching rows "
            "being passed from lower estimators to the meta estimator."
        )

        sub_preds = []

        for pipeline in self.sub_pipelines:
            split_preds = []

            for train, test in year_chunk_cv_split(X, cv=self.cv):
                X_train, y_train = X[train], y[train]
                X_test = X[test]

                pipeline.fit(X_train, y_train)
                split_preds.append(pipeline.predict(X_test))

            sub_preds.append(np.concatenate(split_preds))
            pipeline.fit(X, y)

        self.meta_pipeline.fit(np.array(sub_preds).transpose(), y)

        # return super().fit(X, y)
        return self

    def predict(self, X):
        sub_preds = []

        for pipeline in self.sub_pipelines:
            sub_preds.append(pipeline.predict(X))

        return self.meta_pipeline.predict(np.array(sub_preds).transpose())

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray], step_name: str
    ) -> pd.DataFrame:
        """
        Transform data up to the given step in the pipeline.

        Args:
            X (array-like): Data input for the model
            step_name (str, int): The name of the step which will generate
                the final output.
        """

        # Some internal naming convention causes baikal to append '/0'
        # to the step names defined at instantiation.
        return self._model.predict(X, output_names=f"{step_name}/0")

    def get_step(self, step_name: str) -> BaseEstimator:
        """Get a step object from the pipeline by name."""

        return self._model.get_step(step_name)

    @property
    def _model(self) -> Model:
        if isinstance(self.pipeline, SKLearnWrapper):
            return self.pipeline.model

        return self.pipeline
