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
DEFAULT_MIN_YEAR = 1965


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
    y2 = EloRegressorStep(name="eloregressor")(z_elo, yt)

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
        pipeline: Union[Model, SKLearnWrapper] = SKLearnWrapper(_build_pipeline),
        name: Optional[str] = "stacking_estimator",
        min_year=DEFAULT_MIN_YEAR,
    ) -> None:
        self.min_year = min_year
        super().__init__(pipeline, name=name)

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> Type[R]:
        """Fit estimator to the data"""

        assert X.index.is_monotonic, (
            "X must be sorted by index values. Otherwise, we risk mismatching rows "
            "being passed from lower estimators to the meta estimator."
        )

        return super().fit(self._filter_X(X), self._filter_y(y),)

    def predict(self, X):
        return super().predict(self._filter_X(X))

    def transform(self, X: pd.DataFrame, step_name: str,) -> pd.DataFrame:
        """
        Transform data up to the given step in the pipeline.

        Args:
            X (array-like): Data input for the model
            step_name (str, int): The name of the step which will generate
                the final output.
        """

        # Some internal naming convention causes baikal to append '/0'
        # to the step names defined at instantiation.
        return self._model.predict(self._filter_X(X), output_names=f"{step_name}/0")

    def get_step(self, step_name: str) -> BaseEstimator:
        """Get a step object from the pipeline by name."""

        return self._model.get_step(step_name)

    @property
    def _model(self) -> Model:
        if isinstance(self.pipeline, SKLearnWrapper):
            return self.pipeline.model

        return self.pipeline

    def _filter_X(self, X: pd.DataFrame) -> pd.DataFrame:  # pylint: disable=no-self-use
        return X.query("year >= @self.min_year")

    def _filter_y(self, y: pd.Series) -> pd.Series:
        return y.loc[(slice(None), slice(self.min_year, None), slice(None))]
