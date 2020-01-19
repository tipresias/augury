"""Script for creating default models, training them, then pickling them.

Necessary due to how frequently changes to modules or package versions
make old model files obsolete.
"""

from joblib import Parallel, delayed
import numpy as np

from tests.fixtures.fake_estimator import pickle_fake_estimator
from augury.ml_estimators import (
    BenchmarkEstimator,
    BaggingEstimator,
    StackingEstimator,
    ConfidenceEstimator,
)
from augury.ml_data import MLData
from augury.settings import SEED


np.random.seed(SEED)


def _train_save_model(model, **data_kwargs):
    data = MLData(**data_kwargs)
    model.fit(*data.train_data)
    model.dump()


def main():
    """Loop through models, training and saving each."""
    legacy_data_kwargs = {"data_set": "legacy_model_data", "pipeline": "legacy"}
    data_kwargs = {"data_set": "model_data", "pipeline": "full"}

    model_info = [
        (BenchmarkEstimator(), legacy_data_kwargs),
        (BaggingEstimator(name="tipresias_2019"), legacy_data_kwargs),
        (StackingEstimator(name="tipresias_2020"), data_kwargs),
        (ConfidenceEstimator(), {**data_kwargs, "label_col": "result"}),
    ]

    Parallel(n_jobs=-1)(
        delayed(_train_save_model)(model, **data_kwargs)
        for model, data_kwargs in model_info
    )

    pickle_fake_estimator()


if __name__ == "__main__":
    main()
