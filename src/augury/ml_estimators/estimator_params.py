"""Params for different versions of estimators."""

from mlxtend.regressor import StackingRegressor
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
import statsmodels.api as sm
from xgboost import XGBClassifier

from augury.settings import SEED
from augury.sklearn.metrics import bits_objective
from augury.sklearn.models import EloRegressor, TimeSeriesRegressor
from augury.sklearn.preprocessing import (
    TeammatchToMatchConverter,
    DataFrameConverter,
)
from .base_ml_estimator import BASE_ML_PIPELINE


np.random.seed(SEED)


tipresias_margin_2020 = {
    "name": "tipresias_margin_2020",
    "min_year": 1965,
    "pipeline": StackingRegressor(
        regressors=[
            make_pipeline(
                DataFrameConverter(),
                BASE_ML_PIPELINE,
                ExtraTreesRegressor(random_state=SEED),
            ).set_params(
                **{
                    "extratreesregressor__max_depth": 45,
                    "extratreesregressor__max_features": 0.9493692952,
                    "extratreesregressor__min_samples_leaf": 2,
                    "extratreesregressor__min_samples_split": 3,
                    "extratreesregressor__n_estimators": 113,
                    "pipeline__correlationselector__threshold": 0.0376827797,
                }
            ),
            make_pipeline(
                DataFrameConverter(), TeammatchToMatchConverter(), EloRegressor()
            ).set_params(
                **{
                    "eloregressor__home_ground_advantage": 7,
                    "eloregressor__k": 23.5156358583,
                    "eloregressor__m": 131.54906178,
                    "eloregressor__s": 257.5770727802,
                    "eloregressor__season_carryover": 0.5329064035,
                    "eloregressor__x": 0.6343992255,
                }
            ),
            make_pipeline(
                DataFrameConverter(),
                TimeSeriesRegressor(
                    sm.tsa.ARIMA,
                    order=(6, 0, 1),
                    exog_cols=["at_home", "oppo_cum_percent"],
                ),
            ).set_params(
                **{
                    "timeseriesregressor__exog_cols": ["at_home", "oppo_cum_percent"],
                    "timeseriesregressor__fit_method": "css",
                    "timeseriesregressor__fit_solver": "bfgs",
                    "timeseriesregressor__order": (8, 0, 1),
                }
            ),
        ],
        meta_regressor=make_pipeline(
            StandardScaler(), ExtraTreesRegressor(random_state=SEED)
        ).set_params(
            **{
                "extratreesregressor__max_depth": 41,
                "extratreesregressor__min_samples_leaf": 1,
                "extratreesregressor__min_samples_split": 3,
                "extratreesregressor__n_estimators": 172,
            },
        ),
    ),
}

tipresias_proba_2020 = {
    "name": "tipresias_proba_2020",
    "pipeline": make_pipeline(
        BASE_ML_PIPELINE,
        XGBClassifier(
            random_state=SEED,
            objective=bits_objective,
            use_label_encoder=False,
            verbosity=0,
        ),
    ),
}
