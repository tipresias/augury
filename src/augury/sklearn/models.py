# pylint: disable=too-many-lines

"""Classes and functions based on existing Scikit-learn functionality."""

from collections import defaultdict
from typing import Type, List, Union, Optional, Any, Tuple, Dict, Callable
import copy
import warnings
import tempfile

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from mypy_extensions import TypedDict
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from scipy.stats import norm
from tensorflow import keras
from scikeras.wrappers import KerasRegressor, KerasClassifier as ScikerasClassifier

from augury.types import R
from augury.pipelines.nodes.base import _validate_required_columns
from augury.pipelines.nodes import common
from augury.sklearn.metrics import bits_loss, regressor_team_match_accuracy
from augury.sklearn.model_selection import year_cv_split
from augury.sklearn.preprocessing import TimeStepReshaper, KerasInputLister
from augury.settings import TEAM_NAMES, VENUES, CATEGORY_COLS, ROUND_TYPES


# Default params for EloRegressor
DEFAULT_K = 35.6
DEFAULT_X = 0.49
DEFAULT_M = 130
DEFAULT_HOME_GROUND_ADVANTAGE = 9
DEFAULT_S = 250
DEFAULT_SEASON_CARRYOVER = 0.575

TEAM_LEVEL = 0
YEAR_LEVEL = 1


class EloRegressor(BaseEstimator, RegressorMixin):
    """Elo regression model with a scikit-learn interface."""

    EloDictionary = TypedDict(
        "EloDictionary",
        {
            "previous_elo": np.ndarray,
            "current_elo": np.ndarray,
            "year": int,
            "round_number": int,
        },
    )

    ELO_INDEX_COLS = ["home_team", "year", "round_number"]
    NULL_TEAM_NAME = "0"
    BASE_RATING = 1000

    # Constants for accessing data in the running_elo_ratings matrices
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

    def __init__(
        self,
        k=DEFAULT_K,
        x=DEFAULT_X,
        m=DEFAULT_M,
        home_ground_advantage=DEFAULT_HOME_GROUND_ADVANTAGE,
        s=DEFAULT_S,
        season_carryover=DEFAULT_SEASON_CARRYOVER,
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
        self._running_elo_ratings: self.EloDictionary = {
            "previous_elo": np.array([]),
            "current_elo": np.array([]),
            "year": 0,
            "round_number": 0,
        }
        self._fitted_elo_ratings: self.EloDictionary = copy.deepcopy(
            self._running_elo_ratings
        )
        self._first_fitted_year = 0

        self._team_encoder = LabelEncoder()
        # Have to fit encoder on all team names to not make it dependent
        # on the teams in the train set being a superset of those in the test set.
        # Have to add '0' to team names to account for filling in prev_match_oppo_team
        # for a new team's first match
        self._team_encoder.fit(np.append(np.array(TEAM_NAMES), self.NULL_TEAM_NAME))
        self._null_team = self._team_encoder.transform([self.NULL_TEAM_NAME])[0]

    def fit(self, X: pd.DataFrame, _y: pd.Series = None) -> Type[R]:
        """Fit estimator to data."""
        REQUIRED_COLS = set(self.ELO_INDEX_COLS) | set(self.MATRIX_COLS)
        _validate_required_columns(REQUIRED_COLS, X.columns)

        data_frame: pd.DataFrame = (
            X.set_index(self.ELO_INDEX_COLS, drop=False)
            .rename_axis([None] * len(self.ELO_INDEX_COLS))
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
            .sort_index(level=[self.YEAR_LVL, self.ROUND_NUMBER_LVL], ascending=True)
        )

        self._reset_elo_state()

        elo_matrix = (data_frame.loc[:, self.MATRIX_COLS]).to_numpy()

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
        REQUIRED_COLS = set(self.ELO_INDEX_COLS) | {"away_team"}

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
            .set_index(self.ELO_INDEX_COLS, drop=False)
            .sort_index(level=[self.YEAR_LVL, self.ROUND_NUMBER_LVL], ascending=True)
        )

        elo_matrix = (data_frame.loc[:, self.MATRIX_COLS]).to_numpy()

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
        home_team = int(match_row[self.HOME_TEAM_IDX])
        away_team = int(match_row[self.AWAY_TEAM_IDX])

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
        home_team = int(match_row[self.HOME_TEAM_IDX])
        away_team = int(match_row[self.AWAY_TEAM_IDX])

        self._update_prev_elo_ratings(match_row)

        home_elo_rating = self._calculate_current_elo_rating(
            self.HOME_TEAM_IDX, match_row
        )
        away_elo_rating = self._calculate_current_elo_rating(
            self.AWAY_TEAM_IDX, match_row
        )

        self._running_elo_ratings["current_elo"][home_team] = home_elo_rating
        self._running_elo_ratings["current_elo"][away_team] = away_elo_rating

    def _update_prev_elo_ratings(self, match_row: np.ndarray):
        match_year = match_row[self.YEAR_IDX]
        match_round = match_row[self.ROUND_NUMBER_IDX]

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
            ) + self.BASE_RATING * (1 - self.season_carryover)

            self._running_elo_ratings["year"] = match_year

    def _calculate_current_elo_rating(self, team_idx: int, match_row: np.ndarray):
        team = int(match_row[team_idx])
        was_at_home = int(match_row[team_idx + self.PREV_AT_HOME_OFFSET])
        prev_oppo_team = int(match_row[team_idx + self.PREV_OPPO_OFFSET])
        prev_margin = match_row[team_idx + self.PREV_MARGIN_OFFSET]

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
        actual_outcome = self.x + 0.5 - self.x ** (1 + (margin / self.m))

        return elo_rating + (self.k * (actual_outcome - elo_prediction))

    def _reset_elo_state(self):
        self._running_elo_ratings["previous_elo"] = np.full(
            len(TEAM_NAMES) + 1, self.BASE_RATING
        )
        self._running_elo_ratings["current_elo"] = np.full(
            len(TEAM_NAMES) + 1, self.BASE_RATING
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


class TimeSeriesRegressor(BaseEstimator, RegressorMixin):
    """Wrapper class with Scikit-learn API for regressors from statsmodels.tsa."""

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

    def __init__(
        self,
        stats_model: TimeSeriesModel,
        order: Tuple[int, int, int] = DEFAULT_ORDER,
        exog_cols: List[str] = [],
        fit_method: Optional[str] = None,
        fit_solver: Optional[str] = None,
        confidence=False,
        verbose=0,
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
        fit_method: Name of formula to maximise when fitting.
        fit_solver: Name of solver function passed to the statsmodel's fit method.
        confidence: Whether to return predictions as percentage confidence
            of an outcome (e.g. win) or float value (e.g. predicted margin).
        verbose: How much information to print during the fitting of the statsmodel.
        sm_kwargs: Any other keyword arguments to pass directly to the instantiation
            of the given stats_model.
        """
        self.stats_model = stats_model
        self.order = order
        self.exog_cols = exog_cols
        self.fit_method = fit_method
        self.fit_solver = fit_solver
        self.confidence = confidence
        self.sm_kwargs = sm_kwargs
        self.verbose = verbose
        self._team_models: Dict[str, TimeSeriesModel] = {}

    def fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, np.ndarray]):
        """Fit the model to the training data."""
        time_series_df = X.assign(
            y=y.astype(float), ts_date=X["date"].dt.date
        ).sort_values("ts_date", ascending=True)

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
            if n_train_years >= self.MIN_TIME_SERIES_YEARS
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

            fit_kwargs = {
                "solver": self.fit_solver,
                "method": self.fit_method,
                "disp": self.verbose,
            }
            fit_kwargs = {k: v for k, v in fit_kwargs.items() if v is not None}

            self._team_models[team_name] = self.stats_model(
                y, order=order_param, exog=self._exog_arg(team_df), **self.sm_kwargs
            ).fit(**fit_kwargs)

    def _predict_with_team_model(
        self,
        X: pd.DataFrame,
        team_name: str,  # pylint: disable=unused-argument
        team_model: TimeSeriesModel,
    ) -> pd.Series:
        team_df = X.query("team == @team_name").sort_values("date")

        if not team_df.size:
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

    def _exog_arg(self, data_frame: pd.DataFrame) -> Optional[np.ndarray]:
        return (
            data_frame[self.exog_cols].astype(float).to_numpy()
            if any(self.exog_cols)
            else None
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
        return self._model.predict_proba(X)

    def predict(self, X):
        """Return predictions."""
        return self._model.predict(X)

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
    def history(self) -> defaultdict:
        """Return the history object of the trained Keras model."""
        return self._model.history_

    def _create_model(self):
        keras.backend.clear_session()

        # We use KerasRegressor, because KerasClassifier only works
        # with the Sequential model
        # (2021-11-30: this might no longer be true with switch to scikeras)
        self._model = ScikerasClassifier(
            self.model_func(
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
            callbacks=[keras.callbacks.EarlyStopping(monitor="loss", patience=5)],
        )

    # Adapted this code from: http://zachmoshe.com/2017/04/03/pickling-keras-models.html
    # Keras has since been updated to be picklable, but my custom tensorflow
    # loss function is not (at least I can't figure out how to pickle it).
    # So, this is necessary for basic Scikit-learn functionality like grid search
    # and multiprocessing.
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            keras.models.save_model(self._model, f.name, overwrite=True)
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
        self.__dict__ = d  # pylint: disable=attribute-defined-outside-init


def rnn_model_func(
    n_teams: int = len(TEAM_NAMES),
    n_venues: int = len(VENUES),
    n_categories: int = len(CATEGORY_COLS),
    n_round_types: int = len(ROUND_TYPES),
    n_steps=None,
    n_features=None,
    round_type_dim=None,
    venue_dim=None,
    team_dim=None,
    n_cells=None,
    dropout=None,
    recurrent_dropout=None,
    n_hidden_layers=1,
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    loss=None,
    optimizer=None,
):
    """Function for creating a function that returns an RNN Keras model."""
    assert n_hidden_layers >= 1, "Must have at least one hidden layer"

    create_category_input = lambda name: keras.layers.Input(
        shape=(n_steps,), dtype="int32", name=name
    )
    create_team_embedding_layer = lambda name: keras.layers.Embedding(
        input_dim=n_teams * 2,
        output_dim=team_dim,
        input_length=n_steps,
        name=name,
    )

    team_input = create_category_input("team_input")
    oppo_team_input = create_category_input("oppo_team_input")
    round_type_input = create_category_input("round_type_input")
    venue_input = create_category_input("venue_input")

    numeric_input = keras.layers.Input(
        shape=(n_steps, n_features - n_categories),
        dtype="float32",
        name="numeric_input",
    )

    team_embed = create_team_embedding_layer("embedding_team")(team_input)
    oppo_team_embed = create_team_embedding_layer("embedding_oppo_team")(
        oppo_team_input
    )
    round_type_embed = keras.layers.Embedding(
        input_dim=n_round_types * 2,
        output_dim=round_type_dim,
        input_length=n_steps,
        name="embedding_round_type",
    )(round_type_input)
    venue_embed = keras.layers.Embedding(
        input_dim=n_venues * 2,
        output_dim=venue_dim,
        input_length=n_steps,
        name="embedding_venue",
    )(venue_input)

    concated_layers = keras.layers.concatenate(
        [team_embed, oppo_team_embed, round_type_embed, venue_embed, numeric_input]
    )

    # Have to define the first layer outside the loop due to limitations
    # in how Keras compiles models and being unable to handle
    # dynamically-defined inputs (e.g. concated_layers vs lstm).
    lstm = keras.layers.LSTM(
        n_cells[0],
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        return_sequences=n_hidden_layers - 1 > 0,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        name=f"lstm_{0}",
    )(concated_layers)

    # Allow for variable number of hidden layers, returning sequences to each
    # subsequent LSTM layer
    for idx in range(1, n_hidden_layers):
        lstm = keras.layers.LSTM(
            n_cells[idx],
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=idx < n_hidden_layers - 1,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            name=f"lstm_{idx}",
        )(lstm)

    output = keras.layers.Dense(2)(lstm)

    model = keras.models.Model(
        inputs=[
            team_input,
            oppo_team_input,
            round_type_input,
            venue_input,
            numeric_input,
        ],
        outputs=output,
    )
    model.compile(
        loss=loss, optimizer=optimizer, metrics=[regressor_team_match_accuracy]
    )

    return lambda: model


class RNNRegressor(BaseEstimator, RegressorMixin):
    """Wrapper class for a keras RNN regressor model."""

    def __init__(
        self,
        n_categories: int = len(CATEGORY_COLS),
        n_features: int = None,
        n_steps: int = 2,
        patience: int = 5,
        dropout: float = 0.2,
        recurrent_dropout=0,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        n_cells=50,
        batch_size=None,
        team_dim=4,
        round_type_dim=4,
        venue_dim=4,
        verbose=0,
        n_hidden_layers=1,
        epochs=20,
        optimizer="adam",
        metrics=None,
        loss="mean_absolute_error",
        model_func=rnn_model_func,
    ):
        """Initialise RNNRegressor.

        Params
        ------
        n_teams: Size of team vocab.
        n_years: Number of years in data set.
        n_categories: Total number of category features.
        n_features: Total number of features of X input.

        n_steps: Number of time steps (i.e. past observations) to include in the data.
            (This is the 2nd dimension of the input data.)
        patience: Number of epochs of declining performance before model stops early.
        dropout: Percentage of data that's dropped from inputs.
        recurrent_dropout: Percentage of data that's dropped from recurrent state.
        n_cells: Number of neurons per layer.
        team_dim: Output dimension of embedded team data.
        batch_size: Number of observations per batch (keras default is 32).
        team: Whether to include teams input in model.
        oppo_team: Whether to include oppo_teams input model.
        verbose: How frequently messages are printed during training.
        n_hidden_layers: How many hidden layers to include in the model.
        epochs: Max number of epochs to run during training.
        """
        assert not n_features is None
        assert (
            n_hidden_layers >= 1
        ), "The model assumes at least one hidden layer between the inputs and outputs."

        self.n_features = n_features

        self.n_categories = n_categories
        self.n_steps = n_steps

        self.n_cells = n_cells
        self.n_hidden_layers = n_hidden_layers
        self.venue_dim = venue_dim
        self.team_dim = team_dim
        self.round_type_dim = round_type_dim

        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer

        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose
        self.epochs = epochs
        self.optimizer = optimizer
        self.metrics = metrics or [regressor_team_match_accuracy]
        self.loss = loss

        self.model_func = model_func
        self._model = None

        # We have to include the time reshaper/encoder in the model instead of
        # separate pipelines for consistency during parameter tuning.
        # Both the model and reshaper take n_steps as a parameter and must use
        # the same n_steps value.
        # Also, sklearn really doesn't like 3D data sets.
        self.segment_col = 0
        self._X_reshaper = make_pipeline(
            TimeStepReshaper(n_steps=n_steps, segment_col=self.segment_col),
            KerasInputLister(n_inputs=(self.n_categories + 1)),
        )
        self._y_reshaper = TimeStepReshaper(
            n_steps=n_steps, are_labels=True, segment_col=self.segment_col
        )

        self._create_model()

    def fit(self, X, y):
        """Fit model to training data."""
        X_train, X_test, y_train, y_test = self._train_test_split(X, y)

        return self._model.fit(
            self._inputs(X_train),
            self._labels(y_train),
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            validation_data=(self._inputs(X_test), self._labels(y_test)),
            verbose=self.verbose,
            # Using loss instead of accuracy, because it bounces around less.
            # Also, accuracy tends to reach its max a little after loss reaches its min,
            # meaning the early-stopping delay improves performance.
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=self.patience
                )
            ],
        )

    def predict(self, X):
        """Predict label for each observation."""
        if self._model is None:
            return None

        return self._model.predict(self._inputs(X))

    def score(self, X, y, sample_weight=None):
        """Score the model based on the loss and metrics passed to the keras model."""
        if self._model is None:
            return None

        return self._model.evaluate(self._inputs(X), self._labels(y))

    @property
    def history(self) -> keras.callbacks.History:
        """Return the history object of the trained Keras model."""
        if self._model is None:
            return None

        return self._model.model.history

    def set_params(self, **params):
        """Set params for this model instance, and create a new keras model."""
        prev_params = self.get_params()
        # NOTE: Don't do an early return if params haven't changed, because it causes an
        # error when n_jobs > 1

        # Use parent set_params method to avoid infinite loop
        super().set_params(**params)

        # Only need to recreate reshapers if n_steps has changed
        if prev_params["n_steps"] != self.n_steps:
            self._X_reshaper.set_params(
                timestepreshaper__n_steps=self.n_steps,
                kerasinputlister__n_inputs=(self.n_categories + 1),
            )
            self._y_reshaper.set_params(n_steps=self.n_steps)

        # Need to recreate model after changing any relevant params
        self._create_model()

        return self

    @property
    def n_cells(self) -> List[int]:
        """Get the number of cells per layer."""
        n_cells = (
            self._n_cells
            if isinstance(self._n_cells, list)
            else [self._n_cells] * (self.n_hidden_layers)
        )

        assert len(n_cells) == self.n_hidden_layers, (
            "n_cells must be an integer or a list with length equal to number "
            f"of layers. n_cells has {len(n_cells)} values and there are "
            f"{self.n_hidden_layers} layers in this model."
        )

        return n_cells

    @n_cells.setter
    def n_cells(self, n_cells):
        self._n_cells = n_cells

    def _train_test_split(self, X, y):
        years = y.index.get_level_values(YEAR_LEVEL)
        validation_season = years.max()

        X_with_years = pd.DataFrame(X).assign(year=years)
        X_train_filter, X_test_filter = year_cv_split(
            X_with_years, (validation_season, validation_season + 1)
        )[0]

        y_with_years = pd.DataFrame(y).assign(year=years)
        y_train_filter, y_test_filter = year_cv_split(
            y_with_years, (validation_season, validation_season + 1)
        )[0]

        return X[X_train_filter], X[X_test_filter], y[y_train_filter], y[y_test_filter]

    def _inputs(self, X):
        """Reshape X to fit expected inputs for model."""
        return self._X_reshaper.fit_transform(X)

    def _labels(self, y):
        """Prepare y data array to fit expected input shape for the model."""
        y_with_segments = y.reset_index(TEAM_LEVEL)
        reshaped_y = self._y_reshaper.fit_transform(y_with_segments)
        return reshaped_y[:, 0, :].astype(int)

    def _create_model(self):
        keras.backend.clear_session()

        self._model = KerasRegressor(
            model=self.model_func(
                n_steps=self.n_steps,
                n_features=self.n_features,
                round_type_dim=self.round_type_dim,
                venue_dim=self.venue_dim,
                team_dim=self.team_dim,
                n_cells=self.n_cells,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
                n_hidden_layers=self.n_hidden_layers,
                kernel_regularizer=self.kernel_regularizer,
                recurrent_regularizer=self.recurrent_regularizer,
                bias_regularizer=self.bias_regularizer,
                activity_regularizer=self.activity_regularizer,
                loss=self.loss,
                optimizer=self.optimizer,
            )
        )

    def _load_model(self, saved_model):
        keras.backend.clear_session()

        return KerasRegressor(model=lambda: saved_model)

    # Adapted this code from: http://zachmoshe.com/2017/04/03/pickling-keras-models.html
    # Keras has since been updated to be picklable, but my custom tensorflow
    # loss function is not (at least I can't figure out how to pickle it).
    # So, this is necessary for basic Scikit-learn functionality like grid search
    # and multiprocessing.
    def __getstate__(self):
        model_str = ""

        if self._model and "model" in dir(self._model):
            with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
                keras.models.save_model(self._model.model, f.name, overwrite=True)
                model_str = f.read()

        dict_definition = {
            key: value for key, value in self.__dict__.items() if key != "_model"
        }
        dict_definition.update({"model_str": model_str})
        return dict_definition

    def __setstate__(self, state):
        model = None
        if state["model_str"] != "":
            with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
                f.write(state["model_str"])
                f.flush()
                model = keras.models.load_model(
                    f.name,
                    custom_objects={
                        "regressor_team_match_accuracy": regressor_team_match_accuracy
                    },
                )

        dict_definition = {
            value: key for value, key in state.items() if key != "model_str"
        }

        if model is not None:
            dict_definition.update({"_model": self._load_model(model)})

        self.__dict__ = (
            dict_definition  # pylint: disable=attribute-defined-outside-init
        )
