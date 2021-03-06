"""Script for creating default models, training them, then pickling them.

Necessary due to how frequently changes to modules or package versions
make old model files obsolete.
"""

import os
import sys
from dateutil import parser

import numpy as np
import pandas as pd
from kedro.extras.datasets.pickle import PickleDataSet
from kedro.framework.session import KedroSession

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from tests.fixtures.fake_estimator import pickle_fake_estimator
from augury.ml_estimators import StackingEstimator, BasicEstimator, ConfidenceEstimator
from augury.ml_data import MLData
from augury.ml_estimators import estimator_params
from augury import settings


np.random.seed(settings.SEED)

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
        "train_year_range": settings.FULL_YEAR_RANGE,
    }

    with KedroSession.create(
        settings.PACKAGE_NAME, project_path=settings.BASE_DIR, env=settings.ENV
    ) as session:
        context = session.load_context()
        full_data = pd.DataFrame(context.catalog.load("full_data"))

        # Make sure we're using full data sets instead of truncated prod data sets
        assert (
            full_data["year"].min()
            < parser.parse(settings.PREDICTION_DATA_START_DATE).year
        )
        del full_data

        model_info = [
            (
                ConfidenceEstimator(**estimator_params.tipresias_proba_2020),
                {**data_kwargs, "data_set": "legacy_data", "label_col": "result"},
            ),
            (
                StackingEstimator(**estimator_params.tipresias_margin_2020),
                {**data_kwargs, "data_set": "legacy_data"},
            ),
            (
                BasicEstimator(name="tipresias_margin_2021"),
                {**data_kwargs, "data_set": "full_data"},
            ),
            (
                ConfidenceEstimator(name="tipresias_proba_2021"),
                {**data_kwargs, "data_set": "full_data", "label_col": "result"},
            ),
        ]

        for model, data_kwargs in model_info:
            _train_save_model(model, **data_kwargs)

        pickle_fake_estimator()


if __name__ == "__main__":
    main()
