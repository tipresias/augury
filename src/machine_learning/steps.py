"""
Module for defining baikal Step subclasses based on sklearn BaseEstimator subclasses.
Due to how baikal defines step modules/names, this is necessary for pickling to work.
"""

from baikal import Step
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from mlxtend.feature_selection import ColumnSelector
from xgboost import XGBRegressor

from machine_learning.sklearn import (
    CorrelationSelector,
    ColumnDropper,
    TeammatchToMatchConverter,
    EloRegressor,
)


class StandardScalerStep(Step, StandardScaler):
    def __init__(self, name=None, function=None, n_outputs=1, trainable=True, **kwargs):
        super().__init__(
            name=name,
            function=function,
            n_outputs=n_outputs,
            trainable=trainable,
            **kwargs
        )


class OneHotEncoderStep(Step, OneHotEncoder):
    def __init__(self, name=None, function=None, n_outputs=1, trainable=True, **kwargs):
        super().__init__(
            name=name,
            function=function,
            n_outputs=n_outputs,
            trainable=trainable,
            **kwargs
        )


class ColumnSelectorStep(Step, ColumnSelector):
    def __init__(self, name=None, function=None, n_outputs=1, trainable=True, **kwargs):
        super().__init__(
            name=name,
            function=function,
            n_outputs=n_outputs,
            trainable=trainable,
            **kwargs
        )


class XGBRegressorStep(Step, XGBRegressor):
    def __init__(self, name=None, function=None, n_outputs=1, trainable=True, **kwargs):
        super().__init__(
            name=name,
            function=function,
            n_outputs=n_outputs,
            trainable=trainable,
            **kwargs
        )


class CorrelationSelectorStep(Step, CorrelationSelector):
    def __init__(self, name=None, function=None, n_outputs=1, trainable=True, **kwargs):
        super().__init__(
            name=name,
            function=function,
            n_outputs=n_outputs,
            trainable=trainable,
            **kwargs
        )


class ColumnDropperStep(Step, ColumnDropper):
    def __init__(self, name=None, function=None, n_outputs=1, trainable=True, **kwargs):
        super().__init__(
            name=name,
            function=function,
            n_outputs=n_outputs,
            trainable=trainable,
            **kwargs
        )


class TeammatchToMatchConverterStep(Step, TeammatchToMatchConverter):
    def __init__(self, name=None, function=None, n_outputs=1, trainable=True, **kwargs):
        super().__init__(
            name=name,
            function=function,
            n_outputs=n_outputs,
            trainable=trainable,
            **kwargs
        )


class EloRegressorStep(Step, EloRegressor):
    def __init__(self, name=None, function=None, n_outputs=1, trainable=True, **kwargs):
        super().__init__(
            name=name,
            function=function,
            n_outputs=n_outputs,
            trainable=trainable,
            **kwargs
        )
