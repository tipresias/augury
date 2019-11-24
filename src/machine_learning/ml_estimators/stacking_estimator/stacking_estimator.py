"""Class for model trained on all AFL data and its associated data class"""

from typing import Optional, Union, Type, Callable

from baikal import Input, Model
from baikal.steps import Stack
from baikal.steps.merge import Concatenate
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from machine_learning.steps import (
    StandardScalerStep,
    OneHotEncoderStep,
    ColumnSelectorStep,
    XGBRegressorStep,
    CorrelationSelectorStep,
    ColumnDropperStep,
    TeammatchToMatchConverterStep,
    EloRegressorStep,
)
from machine_learning.settings import (
    TEAM_NAMES,
    ROUND_TYPES,
    VENUES,
    CATEGORY_COLS,
    SEED,
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


def _build_pipeline():
    X = Input("X")
    yt = Input("yt")

    z_ml = ColumnDropperStep(cols_to_drop=ELO_MODEL_COLS, name="columndropper_elo")(X)
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

    z_elo = TeammatchToMatchConverterStep(name="teammatchtomatchconverter")(X)
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
        pipeline: Callable[[], Model] = _build_pipeline,
        name: Optional[str] = "stacking_estimator",
    ) -> None:
        super().__init__(pipeline(), name=name)
        self._pipeline_func = _build_pipeline

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> Type[R]:
        """Fit estimator to the data"""

        assert X.index.is_monotonic, (
            "X must be sorted by index values. Otherwise, we risk mismatching rows "
            "being passed from lower estimators to the meta estimator."
        )

        return super().fit(X, y)

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
        return self.pipeline.predict(X, output_names=f"{step_name}/0")

    def get_step(self, step_name: str) -> BaseEstimator:
        """Get a step object from the pipeline by name."""

        return self.pipeline.get_step(step_name)
