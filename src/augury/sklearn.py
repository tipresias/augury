"""Classes and functions based on existing Scikit-learn functionality."""

from typing import Sequence, Type, List, Union, Optional, Any, Tuple, Dict, Callable
import re
import copy
from functools import partial, update_wrapper
import warnings
import tempfile
import math

import pandas as pd
import numpy as np
from sklearn.base import (
    BaseEstimator,
    RegressorMixin,
    TransformerMixin,
    ClassifierMixin,
)
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.preprocessing import LabelEncoder
from mypy_extensions import TypedDict
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from scipy.stats import norm
from tensorflow import keras
import tensorflow as tf

from augury.types import R, T
from augury.nodes.base import _validate_required_columns
from augury.nodes import common
from augury.settings import TEAM_NAMES


EloDictionary = TypedDict(
    "EloDictionary",
    {
        "prev_team_elo_ratings": np.ndarray,
        "current_team_elo_ratings": np.ndarray,
        "year": int,
        "round_number": int,
    },
)


MATCH_INDEX_COLS = ["year", "round_number"]
MATCH_COLS = ["date", "venue", "round_type"]
OPPO_REGEX = re.compile("^oppo_")

ELO_INDEX_COLS = ["home_team", "year", "round_number"]
NULL_TEAM_NAME = "0"

BASE_RATING = 1000
K = 35.6
ELO_X = 0.49
M = 130
HOME_GROUND_ADVANTAGE = 9
S = 250
SEASON_CARRYOVER = 0.575

MATRIX_COLS = [
    "year",
    "round_number",
    "home_team",
    "home_prev_match_at_home",
    "home_prev_match_oppo_team",
    "home_prev_match_margin",
    "away_team",
    "away_prev_match_at_home",
    "away_prev_match_oppo_team",
    "away_prev_match_margin",
]
YEAR_IDX = MATRIX_COLS.index("year")
ROUND_NUMBER_IDX = MATRIX_COLS.index("round_number")
HOME_TEAM_IDX = MATRIX_COLS.index("home_team")
AWAY_TEAM_IDX = MATRIX_COLS.index("away_team")
PREV_AT_HOME_OFFSET = 1
PREV_OPPO_OFFSET = 2
PREV_MARGIN_OFFSET = 3

YEAR_LVL = 1
ROUND_NUMBER_LVL = 2

# Got these default p, d, q values from a combination of statistical tests
# (mostly for d and q) and trial-and-error (mostly for p)
DEFAULT_ORDER = (6, 0, 1)
# Minimum number of training years when using exogeneous variables
# in a time-series model to avoid the error "On entry to DLASCL parameter number 4
# had an illegal value". Arrived at through trial-and-error.
MIN_YEARS_FOR_DLASCL = 3
# Minimum number of training years when using exogeneous variables
# in a time-series model to avoid the warning "HessianInversionWarning:
# Inverting hessian failed, no bse or cov_params available".
# Arrived at through trial-and-error.
MIN_YEARS_FOR_HESSIAN = 6
MIN_TIME_SERIES_YEARS = max(MIN_YEARS_FOR_DLASCL, MIN_YEARS_FOR_HESSIAN)
# For regressors that might try to predict negative values or 0,
# we need a slightly positive minimum to not get errors when calculating logarithms
MIN_LOG_VAL = 1 * 10 ** -10
LOSS = 0
DRAW = 0.5
WIN = 1


class AveragingRegressor(_BaseComposition, RegressorMixin):
    """Scikit-Learn-style ensemble regressor for averaging regressors' predictions."""

    def __init__(
        self,
        estimators: Sequence[Tuple[str, BaseEstimator]],
        weights: Optional[List[float]] = None,
    ) -> None:
        """Instantiate an AveragingRegressor object.

        Params
        ------
        estimators: Scikit-learn estimators (and their names) for generating
            base predictions that will be averaged.
        weights: Multipliers for individual base predictions to weight their impact
            on the final prediction.
        """
        super().__init__()

        self.estimators = estimators
        self.weights = weights

        self.__validate_estimators_weights_equality()

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> Type[R]:
        """Fit estimators to the data."""
        self.__validate_estimators_weights_equality()

        for _, estimator in self.estimators:
            estimator.fit(X, y)

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict with each estimator, then average the predictions."""
        self.__validate_estimators_weights_equality()

        predictions = [estimator.predict(X) for _, estimator in self.estimators]

        return np.average(np.array(predictions), axis=0, weights=self.weights)

    # The params Dict is way too complicated to try properly typing it
    def get_params(self, deep=True) -> Dict[str, Any]:
        """Get the params dictionary comprised of all estimators."""
        return super()._get_params("estimators", deep=deep)

    def set_params(self, **params) -> BaseEstimator:
        """Set params on any estimators."""
        super()._set_params("estimators", **params)

        return self

    def __validate_estimators_weights_equality(self):
        if self.weights is not None and len(self.estimators) != len(self.weights):
            raise ValueError(
                f"Received {len(self.estimators)} estimators and {len(self.weights)}"
                "weight values, but they must have the same number."
            )


class CorrelationSelector(BaseEstimator, TransformerMixin):
    """Transformer for filtering out features that are less correlated with labels."""

    def __init__(
        self,
        cols_to_keep: List[str] = [],
        threshold: Optional[float] = None,
        labels=pd.Series(),
    ) -> None:
        """Instantiate a CorrelationSelector transformer.

        Params
        ------
        cols_to_keep: List of feature names to always keep in the data set.
        threshold: Minimum correlation value (exclusive) for keeping a feature.
        labels: Label values from the training data set for calculating
            correlations.
        """
        self.threshold = threshold
        self.labels = labels
        self._cols_to_keep = cols_to_keep
        self._above_threshold_columns = cols_to_keep

    def transform(self, X: pd.DataFrame, _y=None) -> pd.DataFrame:
        """Filter out features with weak correlation with the labels."""
        return X[self._above_threshold_columns]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Type[T]:
        """Calculate feature/label correlations and save high-correlation features."""
        if not any(self.labels) and y is not None:
            self.labels = y

        assert any(
            self.labels
        ), "Need labels argument for calculating feature correlations."

        data_frame = pd.concat([X, self.labels], axis=1).drop(self.cols_to_keep, axis=1)
        label_correlations = data_frame.corr().fillna(0)[self.labels.name].abs()

        if self.threshold is None:
            correlated_columns = data_frame.columns
        else:
            correlated_columns = data_frame.columns[label_correlations > self.threshold]

        self._above_threshold_columns = self.cols_to_keep + [
            col for col in correlated_columns if col in X.columns
        ]

        return self

    @property
    def cols_to_keep(self) -> List[str]:
        """List columns that never filtered out."""
        return self._cols_to_keep

    @cols_to_keep.setter
    def cols_to_keep(self, cols_to_keep: List[str]) -> None:
        """Set the list of columns to always keep.

        Also resets the overall list of columns to keep to the given list.
        """
        self._cols_to_keep = cols_to_keep
        self._above_threshold_columns = self._cols_to_keep


class EloRegressor(BaseEstimator, RegressorMixin):
    """Elo regression model with a scikit-learn interface."""

    def __init__(
        self,
        k=K,
        x=ELO_X,
        m=M,
        home_ground_advantage=HOME_GROUND_ADVANTAGE,
        s=S,
        season_carryover=SEASON_CARRYOVER,
    ):
        """
        Instantiate an EloRegressor object.

        Params
        ------
        k: Elo model param for regulating for how long match results affect Elo ratings.
        x: Elo model param.
        m: Elo model param.
        home_ground_advantage: Elo model param for how many points an average home team
            is expected to win by.
        s: Elo model param.
        season_carryover: The percentage of a team's end-of-season Elo score
            that is kept for the next season.
        """
        self.k = k
        self.x = x
        self.m = m
        self.home_ground_advantage = home_ground_advantage
        self.s = s
        self.season_carryover = season_carryover
        self._running_elo_ratings = {
            "previous_elo": np.array([]),
            "current_elo": np.array([]),
            "year": 0,
            "round_number": 0,
        }
        self._fitted_elo_ratings = copy.deepcopy(self._running_elo_ratings)
        self._first_fitted_year = 0

        self._team_encoder = LabelEncoder()
        # Have to fit encoder on all team names to not make it dependent
        # on the teams in the train set being a superset of those in the test set.
        # Have to add '0' to team names to account for filling in prev_match_oppo_team
        # for a new team's first match
        self._team_encoder.fit(np.append(np.array(TEAM_NAMES), NULL_TEAM_NAME))
        self._null_team = self._team_encoder.transform([NULL_TEAM_NAME])[0]

    def fit(self, X: pd.DataFrame, _y: pd.Series = None) -> Type[R]:
        """Fit estimator to data."""
        REQUIRED_COLS = set(ELO_INDEX_COLS) | set(MATRIX_COLS)
        _validate_required_columns(REQUIRED_COLS, X.columns)

        data_frame: pd.DataFrame = (
            X.set_index(ELO_INDEX_COLS, drop=False)
            .rename_axis([None] * len(ELO_INDEX_COLS))
            .assign(
                home_team=lambda df: self._team_encoder.transform(df["home_team"]),
                away_team=lambda df: self._team_encoder.transform(df["away_team"]),
                home_prev_match_oppo_team=lambda df: self._team_encoder.transform(
                    df["home_prev_match_oppo_team"].astype(str)
                ),
                away_prev_match_oppo_team=lambda df: self._team_encoder.transform(
                    df["away_prev_match_oppo_team"].astype(str)
                ),
            )
            .sort_index(level=[YEAR_LVL, ROUND_NUMBER_LVL], ascending=True)
        )

        self._reset_elo_state()

        elo_matrix = (data_frame.loc[:, MATRIX_COLS]).to_numpy()

        for match_row in elo_matrix:
            self._update_current_elo_ratings(match_row)

        self._fitted_elo_ratings = copy.deepcopy(self._running_elo_ratings)
        self._first_fitted_year = data_frame["year"].min()

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.

        Data set used for predictions must follow the training set chronologically.
        Otherwise, an error is raised to avoid making invalid predictions.
        """
        REQUIRED_COLS = set(ELO_INDEX_COLS) | {"away_team"}

        _validate_required_columns(REQUIRED_COLS, X.columns)

        assert X.index.is_monotonic, (
            "Test data must be sorted by index before passing it to the model, ",
            "because calculating Elo ratings requires multiple resorts, ",
            "and we need to be able to make the final predictions match the sorting ",
            "of the input data.",
        )

        data_frame = (
            X.assign(
                home_team=lambda df: self._team_encoder.transform(df["home_team"]),
                away_team=lambda df: self._team_encoder.transform(df["away_team"]),
                home_prev_match_oppo_team=lambda df: self._team_encoder.transform(
                    df["home_prev_match_oppo_team"].astype(str)
                ),
                away_prev_match_oppo_team=lambda df: self._team_encoder.transform(
                    df["away_prev_match_oppo_team"].astype(str)
                ),
            )
            .set_index(ELO_INDEX_COLS, drop=False)
            .sort_index(level=[YEAR_LVL, ROUND_NUMBER_LVL], ascending=True)
        )

        elo_matrix = (data_frame.loc[:, MATRIX_COLS]).to_numpy()

        if data_frame["year"].min() == self._first_fitted_year:
            self._reset_elo_state()
        else:
            self._running_elo_ratings = copy.deepcopy(self._fitted_elo_ratings)

        elo_predictions = [
            self._calculate_current_elo_predictions(match_row)
            for match_row in elo_matrix
        ]
        # Need to zip predictions to group by home/away instead of match,
        # then transpose to convert from rows to columns
        home_away_columns = np.array(list(zip(*elo_predictions))).T

        elo_data_frame = pd.DataFrame(
            home_away_columns,
            columns=["home_elo_prediction", "away_elo_prediction"],
            index=data_frame.index,
        ).sort_index()

        match_data_frame = pd.concat(
            [elo_data_frame, data_frame.sort_index().loc[:, ["away_team", "date"]]],
            axis=1,
        ).reset_index(drop=False)

        # Have to convert to team-match rows to make shape consistent with other
        # model predictions
        return (
            common.convert_match_rows_to_teammatch_rows(match_data_frame)
            # Need to make sure sorting matches wider data set conventions
            .sort_index()
            .loc[:, "elo_prediction"]
            .to_numpy()
        )

    def _calculate_current_elo_predictions(self, match_row: np.ndarray):
        home_team = int(match_row[HOME_TEAM_IDX])
        away_team = int(match_row[AWAY_TEAM_IDX])

        self._update_current_elo_ratings(match_row)

        home_elo_rating = self._running_elo_ratings["current_elo"][home_team]
        away_elo_rating = self._running_elo_ratings["current_elo"][away_team]

        home_elo_prediction = self._calculate_team_elo_prediction(
            home_elo_rating, away_elo_rating, True
        )
        away_elo_prediction = self._calculate_team_elo_prediction(
            away_elo_rating, home_elo_rating, False
        )

        return home_elo_prediction, away_elo_prediction

    # Assumes df sorted by year & round_number with ascending=True in order to calculate
    # correct Elo ratings
    def _update_current_elo_ratings(self, match_row: np.ndarray) -> None:
        home_team = int(match_row[HOME_TEAM_IDX])
        away_team = int(match_row[AWAY_TEAM_IDX])

        self._update_prev_elo_ratings(match_row)

        home_elo_rating = self._calculate_current_elo_rating(HOME_TEAM_IDX, match_row)
        away_elo_rating = self._calculate_current_elo_rating(AWAY_TEAM_IDX, match_row)

        self._running_elo_ratings["current_elo"][home_team] = home_elo_rating
        self._running_elo_ratings["current_elo"][away_team] = away_elo_rating

    def _update_prev_elo_ratings(self, match_row: np.ndarray):
        match_year = match_row[YEAR_IDX]
        match_round = match_row[ROUND_NUMBER_IDX]

        self._validate_consecutive_rounds(match_year, match_round)

        # Need to wait till new round to update prev_team_elo_ratings to avoid
        # updating an Elo rating from the previous round before calculating
        # a relevant team's Elo for the current round
        if match_round != self._running_elo_ratings["round_number"]:
            self._running_elo_ratings["previous_elo"] = np.copy(
                self._running_elo_ratings["current_elo"]
            )

            self._running_elo_ratings["round_number"] = match_round

        # It's typical for Elo models to do a small adjustment toward the baseline
        # between seasons
        if match_year != self._running_elo_ratings["year"]:
            self._running_elo_ratings["previous_elo"] = (
                self._running_elo_ratings["previous_elo"] * self.season_carryover
            ) + BASE_RATING * (1 - self.season_carryover)

            self._running_elo_ratings["year"] = match_year

    def _calculate_current_elo_rating(self, team_idx: int, match_row: np.ndarray):
        team = int(match_row[team_idx])
        was_at_home = int(match_row[team_idx + PREV_AT_HOME_OFFSET])
        prev_oppo_team = int(match_row[team_idx + PREV_OPPO_OFFSET])
        prev_margin = match_row[team_idx + PREV_MARGIN_OFFSET]

        # If a previous oppo team has the null team value, that means this is
        # the given team's first match, so they start with the default Elo rating
        if prev_oppo_team == self._null_team:
            return self._running_elo_ratings["previous_elo"][team]

        prev_elo_rating = self._running_elo_ratings["previous_elo"][team]
        prev_oppo_elo_rating = self._running_elo_ratings["previous_elo"][prev_oppo_team]
        prev_elo_prediction = self._calculate_team_elo_prediction(
            prev_elo_rating, prev_oppo_elo_rating, was_at_home
        )

        return self._calculate_team_elo_rating(
            prev_elo_rating, prev_elo_prediction, prev_margin
        )

    # Basing Elo calculations on:
    # http://www.matterofstats.com/mafl-stats-journal/2013/10/13/building-your-own-team-rating-system.html
    def _calculate_team_elo_prediction(
        self, elo_rating: int, oppo_elo_rating: int, at_home: int
    ) -> float:
        home_ground_advantage = (
            self.home_ground_advantage
            if at_home == 1
            else self.home_ground_advantage * -1
        )

        return 1 / (
            1 + 10 ** ((oppo_elo_rating - elo_rating - home_ground_advantage) / self.s)
        )

    def _calculate_team_elo_rating(
        self, elo_rating: float, elo_prediction: float, margin: int
    ) -> float:
        actual_outcome = self.x + 0.5 - self.x ** (1 + (margin / M))

        return elo_rating + (self.k * (actual_outcome - elo_prediction))

    def _reset_elo_state(self):
        self._running_elo_ratings["previous_elo"] = np.full(
            len(TEAM_NAMES) + 1, BASE_RATING
        )
        self._running_elo_ratings["current_elo"] = np.full(
            len(TEAM_NAMES) + 1, BASE_RATING
        )
        self._running_elo_ratings["year"] = 0
        self._running_elo_ratings["round_number"] = 0

    def _validate_consecutive_rounds(self, match_year: int, match_round: int):
        is_start_of_data = (
            self._running_elo_ratings["year"] == 0
            and self._running_elo_ratings["round_number"] == 0
        )
        is_new_year = (
            match_year - self._running_elo_ratings["year"] == 1 and match_round == 1
        )
        is_same_round = match_round == self._running_elo_ratings["round_number"]
        is_next_round = match_round - self._running_elo_ratings["round_number"] == 1

        assert is_start_of_data or is_new_year or is_same_round or is_next_round, (
            "For Elo calculations to be valid, we must update ratings for each round. "
            f"The current year/round of {match_year}/{match_round} seems to skip "
            f"the last-calculated year/round of {self._running_elo_ratings['year']}/"
            f"{self._running_elo_ratings['round_number']}"
        )


class TeammatchToMatchConverter(BaseEstimator, TransformerMixin):
    """Transformer for converting data frames to be organised by match."""

    def __init__(self, match_cols=MATCH_COLS):
        """
        Instantiate a TeammatchToMatchConverter transformer.

        Params
        ------
        match_cols (list of strings,
            default=["date", "venue", "round_type"]):
            List of match columns that are team neutral (e.g. round_number, venue).
            These won't be renamed with 'home_' or 'away_' prefixes.
        """
        self.match_cols = match_cols
        self._match_cols = list(set(match_cols + MATCH_INDEX_COLS))

    def fit(self, _X, _y=None):
        """Include for consistency with the Scikit-learn interface."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data from being organised by team-match to match.

        This means that input has two rows per match (one row each
        for home and away teams), and output has one row per match (with separate
        columns for home and away team data).
        """
        self._validate_required_columns(X)

        return (
            pd.concat(
                [
                    self._match_data_frame(X, at_home=True),
                    self._match_data_frame(X, at_home=False),
                ],
                axis=1,
            )
            .reset_index()
            .set_index(["home_team"] + MATCH_INDEX_COLS, drop=False)
            .rename_axis([None] * (len(MATCH_INDEX_COLS) + 1))
            .sort_index()
        )

    def _validate_required_columns(self, data_frame: pd.DataFrame):
        required_cols: List[str] = ["team", "oppo_team", "at_home"] + self._match_cols

        _validate_required_columns(required_cols, data_frame.columns)

    def _match_data_frame(
        self, data_frame: pd.DataFrame, at_home: bool = True
    ) -> pd.DataFrame:
        home_index = "team" if at_home else "oppo_team"
        away_index = "oppo_team" if at_home else "team"
        # We drop oppo stats cols, because we end up with both teams' stats per match
        # when we join home and away teams. We keep 'oppo_team' and add the renamed column
        # to the index for convenience
        oppo_stats_cols = [
            col
            for col in data_frame.columns
            if re.match(OPPO_REGEX, col) and col != "oppo_team"
        ]

        return (
            data_frame.query(f"at_home == {int(at_home)}")
            # We index match rows by home_team, year, round_number
            .rename(columns={home_index: "home_team", away_index: "away_team"})
            .drop(["at_home"] + oppo_stats_cols, axis=1)
            # We add all match cols to the index, because they don't affect the upcoming
            # concat, and it's easier than creating a third data frame for match cols
            .set_index(["home_team", "away_team"] + self._match_cols)
            .rename(columns=self._replace_col_names(at_home))
            .sort_index()
        )

    @staticmethod
    def _replace_col_names(at_home: bool):
        team_label = "home" if at_home else "away"
        oppo_label = "away" if at_home else "home"

        return (
            lambda col: col.replace("oppo_", f"{oppo_label}_", 1)
            if re.match(OPPO_REGEX, col)
            else f"{team_label}_{col}"
        )


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Transformer that drops named columns from data frames."""

    def __init__(self, cols_to_drop: List[str] = []):
        """Instantiate a ColumbnDropper transformer.

        Params
        ------
        cols_to_drop: List of column names to drop.
        """
        self.cols_to_drop = cols_to_drop

    def fit(self, _X, y=None):  # pylint: disable=unused-argument
        """Include for consistency with Scikit-learn interface."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop the given columns from the data."""
        return X.drop(self.cols_to_drop, axis=1, errors="ignore")


class DataFrameConverter(BaseEstimator, TransformerMixin):
    """Transformer that converts numpy arrays into DataFrames with named axes.

    Resulting data frame is assigned named columns and indices per the initial data sets
    passed to fit/predict. This is mostly for cases when classes from packages
    convert DataFrames to numpy arrays without asking, and later transformers depend
    on named indices/columns to work.
    """

    def __init__(
        self,
        columns: Optional[Union[List[str], pd.Index]] = None,
        index: Optional[Union[List[str], pd.Index]] = None,
    ):
        """Instantiate a DataFrameConverter transformer.

        Params
        ------
        columns: List of column names or a pd.Index to assign as columns.
        index: List of row names or a pd.Index to assign as the index.
        """
        self.columns = columns
        self.index = index

    def fit(self, X, y=None):  # pylint: disable=unused-argument
        """Include for consistency with Scikit-learn interface."""
        return self

    def transform(self, X: Union[pd.DataFrame, np.array]):
        """Convert data into a pandas DataFrame with the given columns and index."""
        if self.columns is not None:
            assert X.shape[1] == len(self.columns), (
                f"X must have the same number of columns {X.shape[1]} "
                f"as self.columns {len(self.columns)}."
            )

        if self.index is not None:
            assert X.shape[0] == len(self.index), (
                f"X must have the same number of rows {X.shape[0]} "
                f"as indicated by self.index {len(self.index)}."
            )

        return pd.DataFrame(X, columns=self.columns, index=self.index)


class TimeSeriesRegressor(BaseEstimator, RegressorMixin):
    """Wrapper class with Scikit-learn API for regressors from statsmodels.tsa."""

    def __init__(
        self,
        stats_model: TimeSeriesModel,
        order: Tuple[int, int, int] = DEFAULT_ORDER,
        exog_cols: List[str] = [],
        confidence=False,
        **sm_kwargs,
    ):
        """Instantiate a StatsModelsRegressor.

        Params
        ------
        stats_model: A model class from the statsmodels package. So far, has only been
            tested with models from the tsa module.
        order: The `order` param for ARIMA and similar models.
        exog_cols: Names of columns to use as exogeneous variables for ARIMA
            and similar models.
        sm_kwargs: Any other keyword arguments to pass directly to the instantiation
            of the given stats_model.
        """
        self.stats_model = stats_model
        self.order = order
        self.exog_cols = exog_cols
        self.confidence = confidence
        self.sm_kwargs = sm_kwargs
        self._team_models: Dict[str, TimeSeriesModel] = {}

    def fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, np.array]):
        """Fit the model to the training data."""
        time_series_df = X.assign(y=y, ts_date=X["date"].dt.date).sort_values(
            "ts_date", ascending=True
        )

        _ = [
            self._fit_team_model(team_name, team_df)
            for team_name, team_df in time_series_df.groupby("team")
        ]

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions."""
        team_predictions = [
            self._predict_with_team_model(X, team_name, team_model)
            for team_name, team_model in self._team_models.items()
        ]

        return pd.concat(team_predictions, axis=0).sort_index()

    def _fit_team_model(self, team_name, team_df):
        n_train_years = team_df["date"].dt.year.drop_duplicates().count()
        y = team_df.set_index("ts_date")["y"]

        # Need smaller p for teams with fewer seasons for training (looking at you GWS),
        # because higher p values raise weird calculation errors or warnings
        # deep in statsmodels.
        # The minimum number of years and the hard-coded lower p value are somewhat
        # arbitrary, but were arrived at after a fair bit of trial-and-error.
        p_param = (
            self.order[0]
            if n_train_years >= MIN_TIME_SERIES_YEARS
            else min(4, self.order[0])
        )
        order_param = (p_param, *self.order[1:])

        with warnings.catch_warnings():
            # Match dates are roughly weekly, but since they aren't consistent,
            # trying to coerce weekly frequency doesn't work. Ignoring the specific
            # dates and making up a consistent weekly schedule doesn't improve model
            # performance, so better to just ignore this unhelpful warning.
            warnings.filterwarnings(
                "ignore",
                message=(
                    "A date index has been provided, but it has no associated "
                    "frequency information and so will be ignored when "
                    "e.g. forecasting."
                ),
            )
            self._team_models[team_name] = self.stats_model(
                y, order=order_param, exog=self._exog_arg(team_df), **self.sm_kwargs
            ).fit()

    def _predict_with_team_model(
        self,
        X: pd.DataFrame,
        team_name: str,  # pylint: disable=unused-argument
        team_model: TimeSeriesModel,
    ) -> pd.Series:
        team_df = X.query("team == @team_name").sort_values("date")

        if not team_df.any().any():
            return pd.Series(name="yhat")

        team_df_index = team_df.index

        # TODO: Oversimplification of mapping of X dates onto forecast dates
        # that ignores bye weeks and any other scheduling irregularities,
        # but it's good enough for now.
        forecast, standard_error, _conf = team_model.forecast(
            steps=len(team_df_index), exog=self._exog_arg(team_df)
        )

        if self.confidence:
            standard_deviation = standard_error * (len(team_model.fittedvalues)) ** 0.5
            confidence_matrix = np.vstack([forecast, standard_deviation]).transpose()
            y_pred = [norm.cdf(0, *mean_std) for mean_std in confidence_matrix]
        else:
            y_pred = forecast

        return pd.Series(y_pred, name="yhat", index=team_df_index)

    def _exog_arg(self, data_frame: pd.DataFrame) -> Optional[np.array]:
        return data_frame[self.exog_cols].to_numpy() if any(self.exog_cols) else None


def _positive_pred_tensor(y_pred):
    return tf.where(
        tf.math.less_equal(y_pred, tf.constant(0.0)), tf.constant(MIN_LOG_VAL), y_pred
    )


def _log2(x):
    return tf.math.divide(
        tf.math.log(_positive_pred_tensor(x)), tf.math.log(tf.constant(2.0))
    )


def _draw_bits_tensor(y_pred):
    return tf.math.add(
        tf.constant(1.0),
        tf.math.scalar_mul(
            tf.constant(0.5),
            _log2(tf.math.multiply(y_pred, tf.math.subtract(tf.constant(1.0), y_pred))),
        ),
    )


def _win_bits_tensor(y_pred):
    return tf.math.add(tf.constant(1.0), _log2(y_pred))


def _loss_bits_tensor(y_pred):
    return tf.math.add(
        tf.constant(1.0), _log2(tf.math.subtract(tf.constant(1.0), y_pred))
    )


# Raw bits calculations per http://probabilistic-footy.monash.edu/~footy/about.shtml
def bits_loss(y_true, y_pred):
    """Loss function for Tensorflow models based on the bits metric."""
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_win = y_pred[:, -1:]

    # We adjust bits calculation to make a valid ML error formula such that 0
    # represents a correct prediction, and the further off the prediction
    # the higher the error value.
    return K_backend.mean(
        tf.where(
            tf.math.equal(y_true_f, tf.constant(0.5)),
            tf.math.scalar_mul(tf.constant(-1.0), _draw_bits_tensor(y_pred_win)),
            tf.where(
                tf.math.equal(y_true_f, tf.constant(1.0)),
                tf.math.subtract(tf.constant(1.0), _win_bits_tensor(y_pred_win)),
                tf.math.add(
                    tf.constant(1.0),
                    tf.math.scalar_mul(
                        tf.constant(-1.0), _loss_bits_tensor(y_pred_win)
                    ),
                ),
            ),
        ),
    )


class KerasClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper class for the KerasClassifier Scikit-learn wrapper class.

    This is mostly to override __getstate__ and __setstate__, because TensorFlow
    functions are not consistently picklable.
    """

    def __init__(
        self,
        model_func: Callable[[Any], Callable],
        n_hidden_layers: int = 2,
        n_cells: int = 25,
        dropout_rate: float = 0.1,
        label_activation: str = "softmax",
        n_labels: int = 2,
        loss: Callable = bits_loss,
        embed_dim: int = 4,
        epochs: int = 20,
        **kwargs,
    ):
        """Instantiate a KerasClassifier estimator.

        Params
        ------
        model_func: Function that returns a compiled Keras model.
        n_hidden_layers: Number of hidden layers to include between the input
            and output layers.
        n_cells: Number of cells to include in each hidden layer.
        dropout_rate: Dropout rate between layers. Passed directly to the Keras model.
        label_activation: Which activation function to use for the output.
        n_labels: Number of output columns.
        loss: Loss function to use. Passed directly to the Keras model.
        embed_dim: Number of columns produced by the embedding layer.
        """
        self.model_func = model_func
        self.n_hidden_layers = n_hidden_layers
        self.n_cells = n_cells
        self.dropout_rate = dropout_rate
        self.label_activation = label_activation
        self.n_labels = n_labels
        self.loss = loss
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.kwargs = kwargs

        self._create_model()

    def fit(self, X, y, validation_data=None):
        """Fit the model to the training data.

        Params
        ------
        X: Training features.
        y: Training labels.
        validation_data: Optional validation data sets for early stopping.
            Passed directly to the Keras model's fit method.
        """
        self._model.fit(X, y, validation_data=validation_data)

        return self

    def predict_proba(self, X):
        """Return predictions with class probabilities.

        Only works if the output has more than one column. Otherwise,
        returns the same predictions as the #predict method.
        """
        return self._model.predict(X)

    def predict(self, X):
        """Return predictions."""
        return (
            self.predict_proba(X)
            if self.n_labels == 1
            else np.argmax(self.predict_proba(X), axis=1)
        )

    def set_params(self, **params):
        """Set instance params."""
        # We call the parent's #set_params method to avoid an infinite loop.
        super().set_params(**params)

        # Since most of the params are passed to the model's build function,
        # we need to create a new instance with the updated params rather than delegate
        # to the internal model.
        self._create_model()

        return self

    @property
    def history(self) -> keras.callbacks.History:
        """Return the history object of the trained Keras model."""
        return self._model.model.history

    def _create_model(self):
        keras.backend.clear_session()

        # We use KerasRegressor, because KerasClassifier only works
        # with the Sequential model
        self._model = keras.wrappers.scikit_learn.KerasRegressor(
            build_fn=self.model_func(
                n_hidden_layers=self.n_hidden_layers,
                n_cells=self.n_cells,
                dropout_rate=self.dropout_rate,
                label_activation=self.label_activation,
                n_labels=self.n_labels,
                loss=self.loss,
                embed_dim=self.embed_dim,
                **self.kwargs,
            ),
            epochs=self.epochs,
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)],
        )

    # Adapted this code from: http://zachmoshe.com/2017/04/03/pickling-keras-models.html
    # Keras has since been updated to be picklable, but my custom tensorflow loss function is not
    # (at least I can figure out how to pickle it). So, this is necessary
    # for basic Scikit-learn functionality like grid search and multiprocessing.
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            keras.models.save_model(self.model, f.name, overwrite=True)
            model_str = f.read()
        d = {key: value for key, value in self.__dict__.items() if key != "model"}
        d.update({"model_str": model_str})
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            f.write(state["model_str"])
            f.flush()
            model = keras.models.load_model(f.name)
        d = {value: key for value, key in state.items() if key != "model_str"}
        d.update({"model": model})
        self.__dict__ = d # pylint: disable=attribute-defined-outside-init


def _calculate_team_margin(team_margin, oppo_margin):
    # We want True to be 1 and False to be -1
    team_margin_multiplier = ((team_margin > oppo_margin).astype(int) * 2) - 1

    return (
        pd.Series(
            ((team_margin.abs() + oppo_margin.abs()) / 2) * team_margin_multiplier
        )
        .reindex(team_margin.index)
        .sort_index()
    )


def _calculate_match_accuracy(X, y_true, y_pred):
    """Scikit-learn metric function for calculating tipping accuracy."""
    team_match_data_frame = X.assign(y_true=y_true, y_pred=y_pred)
    home_match_data_frame = team_match_data_frame.query("at_home == 1").sort_index()
    away_match_data_frame = (
        team_match_data_frame.query("at_home == 0")
        .set_index(["oppo_team", "year", "round_number"])
        .rename_axis([None, None, None])
        .sort_index()
    )

    home_margin = _calculate_team_margin(
        home_match_data_frame["y_true"], away_match_data_frame["y_true"]
    )
    home_pred_margin = _calculate_team_margin(
        home_match_data_frame["y_pred"], away_match_data_frame["y_pred"]
    )

    return (
        # Any zero margin (i.e. a draw) is counted as correct per usual tipping rules.
        # Predicted margins should never be zero, but since we don't want to encourage
        # any wayward models, we'll count a predicted margin of zero as incorrect
        ((home_margin >= 0) & (home_pred_margin > 0))
        | ((home_margin <= 0) & (home_pred_margin < 0))
    ).mean()


def create_match_accuracy(X):
    """Return Scikit-learn metric function for calculating tipping accuracy."""
    return update_wrapper(
        partial(_calculate_match_accuracy, X), _calculate_match_accuracy
    )


def match_accuracy_scorer(estimator, X, y):
    """Scikit-learn scorer function for calculating tipping accuracy of an estimator."""
    y_pred = estimator.predict(X)

    return _calculate_match_accuracy(X, y, y_pred)


def _positive_pred(y_pred):
    return np.maximum(y_pred, np.repeat(MIN_LOG_VAL, len(y_pred)))


def _draw_bits(y_pred):
    return 1 + (0.5 * np.log2(_positive_pred(y_pred * (1 - y_pred))))


def _win_bits(y_pred):
    return 1 + np.log2(_positive_pred(y_pred))


def _loss_bits(y_pred):
    return 1 + np.log2(_positive_pred(1 - y_pred))


# Raw bits calculations per http://probabilistic-footy.monash.edu/~footy/about.shtml
def _calculate_bits(y_true, y_pred):
    return np.where(
        y_true == DRAW,
        _draw_bits(y_pred),
        np.where(y_true == WIN, _win_bits(y_pred), _loss_bits(y_pred)),
    )


def bits_scorer(
    estimator: BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    y: pd.Series,
    proba=False,
    n_years=1,
) -> float:
    """Scikit-learn scorer for the bits metric.

    Mostly for use in calls to cross_validate. Calculates a score
    based on the the model's predicted probability of a given result. For this metric,
    higher scores are better.

    We simplify calculations by using Numpy math functions. This has the benefit
    of not require a lot of reshaping based on categorical features, but gives
    final values that deviate a little from what is correct, because this scorer
    calculates bits per team-match combination rather than per match,
    which is how the official bits score will be calculated.

    Params
    ------
    estimator: The estimator being scored.
    X: Model features.
    y: Model labels.
    proba: Whether to use the `predict_proba` method to get predictions.
    """

    try:
        y_pred = estimator.predict_proba(X)[:, -1] if proba else estimator.predict(X)
    # TF/Keras models don't use predict_proba, so for classifiers, we pass proba=True,
    # then rescue and call predict.
    except AttributeError:
        if proba:
            y_pred = estimator.predict(X)[:, -1]
        else:
            raise

    if isinstance(X, pd.DataFrame) and "year" in X.columns:
        n_years = X["year"].drop_duplicates().count()

    # For tipping competitions, bits are summed across the season.
    # We divide by number of seasons for easier comparison with other models.
    # We divide by two to get a rough per-match bits value.
    return _calculate_bits(y, y_pred).sum() / n_years / 2


def _draw_bits_hessian(y_pred):
    return (y_pred ** 2 - y_pred + 0.5) / (
        math.log(2) * y_pred ** 2 * (y_pred - 1) ** 2
    )


def _win_bits_hessian(y_pred):
    return 1 / (math.log(2) * y_pred ** 2)


def _loss_bits_hessian(y_pred):
    return 1 / (math.log(2) * (1 - y_pred) ** 2)


def _bits_hessian(y_true, y_pred):
    return np.where(
        y_true == DRAW,
        _draw_bits_hessian(y_pred),
        np.where(y_true == WIN, _win_bits_hessian(y_pred), _loss_bits_hessian(y_pred),),
    )


def _draw_bits_gradient(y_pred):
    return (y_pred - 0.5) / (math.log(2) * (y_pred - y_pred ** 2))


def _win_bits_gradient(y_pred):
    return -1 / (math.log(2) * y_pred)


def _loss_bits_gradient(y_pred):
    return 1 / (math.log(2) * (1 - y_pred))


def _bits_gradient(y_true, y_pred):
    return np.where(
        y_true == DRAW,
        _draw_bits_gradient(y_pred),
        np.where(
            y_true == WIN, _win_bits_gradient(y_pred), _loss_bits_gradient(y_pred),
        ),
    )


def bits_objective(y_true, y_pred) -> Tuple[np.array, np.array]:
    """Objective function for XGBoost estimators.

    The gradient and hessian formulas are based on the formula for the bits error
    function rather than the bits metric to make the math more consistent
    with other objective and error functions.

    Params
    ------
    y_true [array-like, (n_observations,)]: Data labels.
    y_pred [array-like, (n_observations, n_label_classes)]: Model predictions.
        In the case of binary classification, the shape is (n_observations,)

    Returns
    -------
    gradient, hessian [tuple of array-like, (n_observations * n_classes,)]:
        gradient function is the derivative of the loss function, and hessian function
        is the derivative of the gradient function.
    """
    # Since y_pred can be 1- or 2-dimensional, we should only reshape y_true
    # when the latter is the case.
    y_true_matrix = (
        y_true.reshape(-1, 1) if len(y_true.shape) != len(y_pred.shape) else y_true
    )

    return (
        _bits_gradient(y_true_matrix, y_pred).flatten(),
        _bits_hessian(y_true_matrix, y_pred).flatten(),
    )


def _bits_error(y_true, y_pred):
    # We adjust bits calculation to make a valid ML error formula such that 0
    # represents a correct prediction, and the further off the prediction
    # the higher the error value.
    return np.where(
        y_true == DRAW,
        -1 * _draw_bits(y_pred),
        np.where(y_true == WIN, 1 - _win_bits(y_pred), 1 + (-1 * _loss_bits(y_pred)),),
    )


def bits_metric(y_pred, y_true_matrix) -> Tuple[str, float]:
    """Metric function for internal model evaluation in XGBoost estimators.

    Note that the order of params, per the xgboost documentation, is y_pred, y_true
    as opposed to the usual y_true, y_pred for Scikit-learn metric functions.

    Params
    ------
    y_pred: Model predictions.
    y_true: Data labels.

    Returns
    -------
    Tuple of the metric name and mean bits error.
    """
    y_true = y_true_matrix.get_label()

    return "mean_bits_error", _bits_error(y_true, y_pred).mean()


def year_cv_split(X, year_range):
    """Split data by year for cross-validation for time-series data.

    Makes data from each year in the year_range a test set per split, with data
    from all earlier years being in the train split.
    """
    return [
        ((X["year"] < year).to_numpy(), (X["year"] == year).to_numpy())
        for year in range(*year_range)
    ]
