"""
Script for creating default models, training them, then pickling them.
Necessary due to how frequently changes to modules or package versions
make old model files obsolete.
"""


from augury.ml_estimators import (
    BenchmarkEstimator,
    BaggingEstimator,
    StackingEstimator,
    ConfidenceEstimator,
)
from augury.ml_data import MLData
from tests.fixtures.fake_estimator import pickle_fake_estimator


def train_save_model(model, data_set_name):
    pipeline = "legacy" if "legacy" in data_set_name else "full"
    data = MLData(data_set=data_set_name, pipeline=pipeline)
    model.fit(*data.train_data)
    model.dump()


def main():
    model_info = [
        (BenchmarkEstimator(), "legacy_model_data"),
        (BaggingEstimator(name="tipresias_2019"), "legacy_model_data"),
        (StackingEstimator(), "model_data"),
        (ConfidenceEstimator(), "model_data"),
    ]

    _ = [train_save_model(*model_and_data) for model_and_data in model_info]

    pickle_fake_estimator()


if __name__ == "__main__":
    main()
