"""Script for creating default models, training them, then pickling them.

Necessary due to how frequently changes to modules or package versions
make old model files obsolete.
"""

import os

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
from augury.io import PickleGCStorageDataSet
from augury.settings import SEED


np.random.seed(SEED)

BUCKET_NAME = "afl_data"
TRAIN_YEAR_RANGE = (2020,)


def _train_save_model(model, **data_kwargs):
    data = MLData(**data_kwargs)
    model.fit(*data.train_data)
    model.dump()

    # For now, we're using a flat file structure in the data bucket,
    # so we just want the filename of the pickled model
    bucket_filepath = os.path.split(model.pickle_filepath)[-1]
    data_set = PickleGCStorageDataSet(filepath=bucket_filepath, bucket_name=BUCKET_NAME)
    data_set.save(model)


def main():
    """Loop through models, training and saving each."""
    legacy_data_kwargs = {
        "data_set": "legacy_model_data",
        "pipeline": "legacy",
        "train_year_range": TRAIN_YEAR_RANGE,
    }
    data_kwargs = {
        "data_set": "model_data",
        "pipeline": "full",
        "train_year_range": TRAIN_YEAR_RANGE,
    }

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
