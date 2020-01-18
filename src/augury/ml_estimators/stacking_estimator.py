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

from augury.sklearn import (
    TeammatchToMatchConverter,
    EloRegressor,
    DataFrameConverter,
    TimeSeriesRegressor,
)
from augury.settings import SEED
from augury.types import R
from .base_ml_estimator import BaseMLEstimator, BASE_ML_PIPELINE

np.random.seed(SEED)


DEFAULT_MIN_YEAR = 1965

ML_PIPELINE = make_pipeline(
    DataFrameConverter(), BASE_ML_PIPELINE, ExtraTreesRegressor(random_state=SEED)
)

ELO_PIPELINE = make_pipeline(
    DataFrameConverter(), TeammatchToMatchConverter(), EloRegressor()
)

ARIMA_PIPELINE = make_pipeline(
    DataFrameConverter(),
    TimeSeriesRegressor(
        sm.tsa.ARIMA, order=(6, 0, 1), exog_cols=["at_home", "oppo_cum_percent"]
    ),
)

META_PIPELINE = make_pipeline(StandardScaler(), ExtraTreesRegressor(random_state=SEED))

PIPELINE = StackingRegressor(
    regressors=[ML_PIPELINE, ELO_PIPELINE, ARIMA_PIPELINE], meta_regressor=META_PIPELINE
)


class StackingEstimator(BaseMLEstimator):
    """Stacked ensemble model based on `mlxtend`'s `StackingRegressor`."""

    def __init__(
        self,
        pipeline: Union[Pipeline, BaseEstimator] = PIPELINE,
        name: Optional[str] = "stacking_estimator",
        min_year=DEFAULT_MIN_YEAR,
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
