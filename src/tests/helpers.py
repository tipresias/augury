"""Functions and classes to deduplicate and simplify test code"""

from typing import Union
import os

import pandas as pd
import numpy as np
from kedro.context import load_context

from augury.settings import BASE_DIR


def regression_accuracy(
    y: Union[pd.DataFrame, np.ndarray], y_pred: Union[pd.DataFrame, np.ndarray]
) -> np.ndarray:
    try:
        correct_preds = ((y >= 0) & (y_pred > 0)) | ((y <= 0) & (y_pred < 0))
    except ValueError:
        reset_y = y.reset_index(drop=True)
        reset_y_pred = y_pred.reset_index(drop=True)
        correct_preds = ((reset_y >= 0) & (reset_y_pred > 0)) | (
            (reset_y <= 0) & (reset_y_pred < 0)
        )

    return np.mean(correct_preds.astype(int))


class KedroContextMixin:
    """Mixin class for loading the kedro context in tests"""

    @staticmethod
    def load_context(**context_kwargs):
        """
        Load the kedro context, changing the environment variable for CI,
        so data sets will load correctly.
        """

        # Need to use production environment for loading data sets if in CI, because we
        # don't check data set files into source control
        kedro_env = (
            "production"
            if os.environ.get("CI") == "true"
            else os.environ.get("PYTHON_ENV")
        )

        return load_context(BASE_DIR, env=kedro_env, **context_kwargs)
