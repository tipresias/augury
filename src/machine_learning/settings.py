import os
from datetime import date
import pytz

import yaml


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data/01_raw/")
CASSETTE_LIBRARY_DIR = os.path.join(BASE_DIR, "src/tests/fixtures/cassettes")

# Using Melbourne for datetimes that aren't location specific, because these are usually
# start/end datetimes, and Melbourne will at least get us on the correct date
# (i.e. no weirdness around changing timezones resulting in datetimes
# just before or after midnight, and thus on a different day)
MELBOURNE_TIMEZONE = pytz.timezone("Australia/Melbourne")

# We calculate rolling sums/means for some features that can span over 5 seasons
# of data, so we're setting it to 10 to be on the safe side.
N_SEASONS_FOR_PREDICTION = 10
# We want to limit the amount of data loaded as much as possible,
# because we only need the full data set for model training and data analysis,
# and we want to limit memory usage and speed up data processing for tipping
PREDICTION_DATA_START_DATE = f"{date.today().year - N_SEASONS_FOR_PREDICTION}-01-01"

with open(os.path.join(BASE_DIR, "src/machine_learning/ml_models.yml"), "r") as file:
    ML_MODELS = yaml.safe_load(file).get("models", [])
