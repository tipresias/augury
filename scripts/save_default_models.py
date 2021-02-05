"""Script for creating default models, training them, then pickling them.

Necessary due to how frequently changes to modules or package versions
make old model files obsolete.
"""

import os
from dateutil import parser

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from kedro.extras.datasets.pickle import PickleDataSet

from tests.fixtures.fake_estimator import pickle_fake_estimator
from augury.ml_estimators import (
    StackingEstimator,
    ConfidenceEstimator,
)
from augury.ml_data import MLData
from augury.settings import SEED, PREDICTION_DATA_START_DATE
from augury.context import load_project_context


np.random.seed(SEED)

BUCKET_NAME = "afl_data"
TRAIN_YEAR_RANGE = (2020,)
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")


def _train_save_model(model, **data_kwargs):
    data = MLData(**data_kwargs)
    model.fit(*data.train_data)
    model.dump()

    # For now, we're using a flat file structure in the data bucket,
    # so we just want the filename of the pickled model
    bucket_filename = os.path.split(model.pickle_filepath)[-1]
    data_set = PickleDataSet(
        filepath=f"gs://{BUCKET_NAME}/{bucket_filename}",
        backend="joblib",
        fs_args={"project": "tipresias"},
        credentials={"token": GOOGLE_APPLICATION_CREDENTIALS},
    )
    data_set.save(model)


def main():
    """Loop through models, training and saving each."""
    data_kwargs = {
        "data_set": "model_data",
        "train_year_range": TRAIN_YEAR_RANGE,
    }

    context = load_project_context()
    model_data = pd.DataFrame(context.catalog.load("model_data"))

    # Make sure we're using full data sets instead of truncated prod data sets
    assert model_data["year"].min() < parser.parse(PREDICTION_DATA_START_DATE).year

    model_info = [
        (ConfidenceEstimator(), {**data_kwargs, "label_col": "result"}),
        (StackingEstimator(name="tipresias_2020"), data_kwargs),
    ]

    Parallel(n_jobs=-1)(
        delayed(_train_save_model)(model, **data_kwargs)
        for model, data_kwargs in model_info
    )

    pickle_fake_estimator()


if __name__ == "__main__":
    main()
