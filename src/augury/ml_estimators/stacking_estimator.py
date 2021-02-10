"""Class for model trained on all AFL data and its associated data class."""

from typing import Optional, Union, Type

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from mlxtend.regressor import StackingRegressor
import statsmodels.api as sm

from augury.sklearn.preprocessing import (
    TeammatchToMatchConverter,
    DataFrameConverter,
)
from augury.sklearn.models import EloRegressor, TimeSeriesRegressor
from augury.settings import SEED
from augury.types import R
from .base_ml_estimator import BaseMLEstimator, BASE_ML_PIPELINE

np.random.seed(SEED)


BEST_PARAMS = {
    "min_year": 1965,
    "ml_pipeline": {
        "extratreesregressor__max_depth": 45,
        "extratreesregressor__max_features": 0.9493692952,
        "extratreesregressor__min_samples_leaf": 2,
        "extratreesregressor__min_samples_split": 3,
        "extratreesregressor__n_estimators": 113,
        "pipeline__correlationselector__threshold": 0.0376827797,
    },
    "elo_pipeline": {
        "eloregressor__home_ground_advantage": 7,
        "eloregressor__k": 23.5156358583,
        "eloregressor__m": 131.54906178,
        "eloregressor__s": 257.5770727802,
        "eloregressor__season_carryover": 0.5329064035,
        "eloregressor__x": 0.6343992255,
    },
    "arima_pipeline": {
        "timeseriesregressor__exog_cols": ["at_home", "oppo_cum_percent"],
        "timeseriesregressor__fit_method": "css",
        "timeseriesregressor__fit_solver": "bfgs",
        "timeseriesregressor__order": (8, 0, 1),
    },
    "meta_pipeline": {
        "extratreesregressor__max_depth": 41,
        "extratreesregressor__min_samples_leaf": 1,
        "extratreesregressor__min_samples_split": 3,
        "extratreesregressor__n_estimators": 172,
    },
}

ML_PIPELINE = make_pipeline(
    DataFrameConverter(), BASE_ML_PIPELINE, ExtraTreesRegressor(random_state=SEED)
).set_params(**BEST_PARAMS["ml_pipeline"])

ELO_PIPELINE = make_pipeline(
    DataFrameConverter(), TeammatchToMatchConverter(), EloRegressor()
).set_params(**BEST_PARAMS["elo_pipeline"])

ARIMA_PIPELINE = make_pipeline(
    DataFrameConverter(),
    TimeSeriesRegressor(
        sm.tsa.ARIMA,
        order=(6, 0, 1),
        exog_cols=["at_home", "oppo_cum_percent"],
    ),
).set_params(**BEST_PARAMS["arima_pipeline"])

META_PIPELINE = make_pipeline(
    StandardScaler(), ExtraTreesRegressor(random_state=SEED)
).set_params(**BEST_PARAMS["meta_pipeline"])

PIPELINE = StackingRegressor(
    regressors=[ML_PIPELINE, ELO_PIPELINE, ARIMA_PIPELINE], meta_regressor=META_PIPELINE
)


class StackingEstimator(BaseMLEstimator):
    """Stacked ensemble model based on `mlxtend`'s `StackingRegressor`."""

    def __init__(
        self,
        pipeline: Union[Pipeline, BaseEstimator] = PIPELINE,
        name: Optional[str] = "stacking_estimator",
        min_year=BEST_PARAMS["min_year"],
    ) -> None:
        """Instantiate a StackingEstimator object.

        Params
        ------
        pipeline: Pipeline of Scikit-learn estimators ending in a regressor
            or classifier.
        name: Name of the estimator for reference by Kedro data sets and filenames.
        min_year: Minimum year for data used in training (inclusive).
        """
        super().__init__(pipeline, name=name)

        self.min_year = min_year

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> Type[R]:
        """Fit estimator to the data."""
        X_filtered, y_filtered = (
            self._filter_by_min_year(X),
            self._filter_by_min_year(y),
        )

        assert X_filtered.index.is_monotonic, (
            "X must be sorted by index values. Otherwise, we risk mismatching rows "
            "being passed from lower estimators to the meta estimator."
        )

        for regr in self.pipeline.regressors:
            if "dataframeconverter__columns" in regr.get_params().keys():
                regr.set_params(
                    **{
                        "dataframeconverter__columns": X_filtered.columns,
                        "dataframeconverter__index": X_filtered.index,
                    }
                )

        self.pipeline.set_params(
            **{"pipeline-1__pipeline__correlationselector__labels": y_filtered}
        )

        return super().fit(X_filtered, y_filtered)

    def predict(self, X):
        """Make predictions."""
        X_filtered = self._filter_by_min_year(X)

        # On fit, StackingRegressor reassigns the defined regressors to regr_,
        # which it uses internally to fit/predict. Calling set_params doesn't update
        # the regr_ attribute, which means without this little hack,
        # we would be predicting with outdated params.
        for regr in self.pipeline.regr_:
            regr.set_params(
                **{
                    "dataframeconverter__columns": X_filtered.columns,
                    "dataframeconverter__index": X_filtered.index,
                }
            )

        return super().predict(X_filtered)

    def _filter_by_min_year(
        self, data: Union[pd.DataFrame, pd.Series]
    ) -> Union[pd.DataFrame, pd.Series]:
        row_slice = (slice(None), slice(self.min_year, None), slice(None))

        if isinstance(data, pd.Series):
            return data.loc[row_slice]

        return data.loc[row_slice, :]
