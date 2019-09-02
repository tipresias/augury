import os
from datetime import timezone, timedelta, date
import yaml


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data/01_raw/")
CASSETTE_LIBRARY_DIR = os.path.join(BASE_DIR, "src/tests/fixtures/cassettes")

HOURS_FROM_UTC_TO_MELBOURNE = 11
MELBOURNE_TIMEZONE = timezone(timedelta(hours=HOURS_FROM_UTC_TO_MELBOURNE))

# We calculate rolling sums/means for some features that can span over 5 seasons
# of data, so we're setting it to 10 to be on the safe side.
N_SEASONS_FOR_PREDICTION = 10
# We want to limit the amount of data loaded as much as possible,
# because we only need the full data set for model training and data analysis,
# and we want to limit memory usage and speed up data processing for tipping
PREDICTION_DATA_START_DATE = f"{date.today().year - N_SEASONS_FOR_PREDICTION}-01-01"

with open(os.path.join(BASE_DIR, "src/machine_learning/ml_models.yml"), "r") as file:
    ML_MODELS = yaml.safe_load(file).get("models", [])
