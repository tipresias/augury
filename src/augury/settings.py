"""App-wide constants for app and data configuration."""

from typing import Dict, Union, List
import os
from datetime import date
import pytz

import yaml

from augury.types import MLModelDict


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

with open(os.path.join(BASE_DIR, "src/augury/ml_models.yml"), "r") as file:
    ML_MODELS: List[MLModelDict] = yaml.safe_load(file).get("models", [])
    PREDICTION_TYPES: List[str] = list(
        {ml_model["prediction_type"] for ml_model in ML_MODELS}
    )

# TODO: Create an SQLite DB to store and handle logic for these hard-coded entities
# (i.e. there could be Team, City, Venue models with associations & model logic)
TEAM_TRANSLATIONS = {
    "Tigers": "Richmond",
    "Blues": "Carlton",
    "Demons": "Melbourne",
    "Giants": "GWS",
    "GWS Giants": "GWS",
    "Greater Western Sydney": "GWS",
    "Suns": "Gold Coast",
    "Bombers": "Essendon",
    "Swans": "Sydney",
    "Magpies": "Collingwood",
    "Kangaroos": "North Melbourne",
    "Crows": "Adelaide",
    "Bulldogs": "Western Bulldogs",
    "Footscray": "Western Bulldogs",
    "Dockers": "Fremantle",
    "Power": "Port Adelaide",
    "Saints": "St Kilda",
    "Eagles": "West Coast",
    "Lions": "Brisbane Lions",
    "Cats": "Geelong",
    "Hawks": "Hawthorn",
    "Adelaide Crows": "Adelaide",
    "Gold Coast Suns": "Gold Coast",
    "Geelong Cats": "Geelong",
    "West Coast Eagles": "West Coast",
    "Sydney Swans": "Sydney",
}

# For when we fetch upcoming matches in the fixture and need to make Footywire venue
# names consistent with AFL tables venue names
FOOTYWIRE_VENUE_TRANSLATIONS = {
    "AAMI Stadium": "Football Park",
    "ANZ Stadium": "Stadium Australia",
    "UTAS Stadium": "York Park",
    "Blacktown International": "Blacktown",
    "Blundstone Arena": "Bellerive Oval",
    "Domain Stadium": "Subiaco",
    "Etihad Stadium": "Docklands",
    "GMHBA Stadium": "Kardinia Park",
    "MCG": "M.C.G.",
    "Mars Stadium": "Eureka Stadium",
    "Metricon Stadium": "Carrara",
    "Optus Stadium": "Perth Stadium",
    "SCG": "S.C.G.",
    "Spotless Stadium": "Sydney Showground",
    "Showground Stadium": "Sydney Showground",
    "TIO Stadium": "Marrara Oval",
    "Westpac Stadium": "Wellington",  # Not copy-pasta: AFL Tables calls it Wellington
    "Marvel Stadium": "Docklands",
    "Canberra Oval": "Manuka Oval",
    "TIO Traeger Park": "Traeger Park",
    # Correct spelling is 'Traeger', but footywire.com is spelling it 'Traegar' in its
    # fixtures, so including both in case they eventually fix the misspelling
    "TIO Traegar Park": "Traeger Park",
    "GIANTS Stadium": "Sydney Showground",
}

CITIES: Dict[str, Dict[str, Union[str, float]]] = {
    "Adelaide": {
        "state": "SA",
        "lat": -34.9285,
        "long": 138.6007,
        "timezone": "Australia/Adelaide",
    },
    "Sydney": {
        "state": "NSW",
        "lat": -33.8688,
        "long": 151.2093,
        "timezone": "Australia/Sydney",
    },
    "Melbourne": {
        "state": "VIC",
        "lat": -37.8136,
        "long": 144.9631,
        "timezone": "Australia/Melbourne",
    },
    "Geelong": {
        "state": "VIC",
        "lat": -38.1499,
        "long": 144.3617,
        "timezone": "Australia/Melbourne",
    },
    "Perth": {
        "state": "WA",
        "lat": -31.9505,
        "long": 115.8605,
        "timezone": "Australia/Perth",
    },
    "Gold Coast": {
        "state": "QLD",
        "lat": -28.0167,
        "long": 153.4000,
        "timezone": "Australia/Brisbane",
    },
    "Brisbane": {
        "state": "QLD",
        "lat": -27.4698,
        "long": 153.0251,
        "timezone": "Australia/Brisbane",
    },
    "Launceston": {
        "state": "TAS",
        "lat": -41.4332,
        "long": 147.1441,
        "timezone": "Australia/Hobart",
    },
    "Canberra": {
        "state": "ACT",
        "lat": -35.2809,
        "long": 149.1300,
        "timezone": "Australia/Sydney",
    },
    "Hobart": {
        "state": "TAS",
        "lat": -42.8821,
        "long": 147.3272,
        "timezone": "Australia/Hobart",
    },
    "Darwin": {
        "state": "NT",
        "lat": -12.4634,
        "long": 130.8456,
        "timezone": "Australia/Darwin",
    },
    "Alice Springs": {
        "state": "NT",
        "lat": -23.6980,
        "long": 133.8807,
        "timezone": "Australia/Darwin",
    },
    "Wellington": {
        "state": "NZ",
        "lat": -41.2865,
        "long": 174.7762,
        "timezone": "Pacific/Auckland",
    },
    "Euroa": {
        "state": "VIC",
        "lat": -36.7500,
        "long": 145.5667,
        "timezone": "Australia/Melbourne",
    },
    "Yallourn": {
        "state": "VIC",
        "lat": -38.1803,
        "long": 146.3183,
        "timezone": "Australia/Melbourne",
    },
    "Cairns": {
        "state": "QLD",
        "lat": -6.9186,
        "long": 145.7781,
        "timezone": "Australia/Brisbane",
    },
    "Ballarat": {
        "state": "VIC",
        "lat": -37.5622,
        "long": 143.8503,
        "timezone": "Australia/Melbourne",
    },
    "Shanghai": {
        "state": "CHN",
        "lat": 31.2304,
        "long": 121.4737,
        "timezone": "Asia/Shanghai",
    },
    "Albury": {
        "state": "NSW",
        "lat": -36.0737,
        "long": 146.9135,
        "timezone": "Australia/Sydney",
    },
    "Townsville": {
        "state": "QLD",
        "lat": -19.2590,
        "long": 146.8169,
        "timezone": "Australia/Brisbane",
    },
}

TEAM_CITIES = {
    "Adelaide": "Adelaide",
    "Brisbane Lions": "Brisbane",
    "Brisbane Bears": "Brisbane",
    "Carlton": "Melbourne",
    "Collingwood": "Melbourne",
    "Essendon": "Melbourne",
    "Fitzroy": "Melbourne",
    "Western Bulldogs": "Melbourne",
    "Fremantle": "Perth",
    "GWS": "Sydney",
    "Geelong": "Geelong",
    "Gold Coast": "Gold Coast",
    "Hawthorn": "Melbourne",
    "Melbourne": "Melbourne",
    "North Melbourne": "Melbourne",
    "Port Adelaide": "Adelaide",
    "Richmond": "Melbourne",
    "St Kilda": "Melbourne",
    "Sydney": "Sydney",
    "University": "Melbourne",
    "West Coast": "Perth",
}

VENUE_CITIES = {
    # AFL Tables venues
    "Football Park": "Adelaide",
    "S.C.G.": "Sydney",
    "Windy Hill": "Melbourne",
    "Subiaco": "Perth",
    "Moorabbin Oval": "Melbourne",
    "M.C.G.": "Melbourne",
    "Kardinia Park": "Geelong",
    "Victoria Park": "Melbourne",
    "Waverley Park": "Melbourne",
    "Princes Park": "Melbourne",
    "Western Oval": "Melbourne",
    "W.A.C.A.": "Perth",
    "Carrara": "Gold Coast",
    "Gabba": "Brisbane",
    "Docklands": "Melbourne",
    "York Park": "Launceston",
    "Manuka Oval": "Canberra",
    "Sydney Showground": "Sydney",
    "Adelaide Oval": "Adelaide",
    "Bellerive Oval": "Hobart",
    "Marrara Oval": "Darwin",
    "Traeger Park": "Alice Springs",
    "Perth Stadium": "Perth",
    "Stadium Australia": "Sydney",
    "Wellington": "Wellington",
    "Lake Oval": "Melbourne",
    "East Melbourne": "Melbourne",
    "Corio Oval": "Geelong",
    "Junction Oval": "Melbourne",
    "Brunswick St": "Melbourne",
    "Punt Rd": "Melbourne",
    "Glenferrie Oval": "Melbourne",
    "Arden St": "Melbourne",
    "Olympic Park": "Melbourne",
    "Yarraville Oval": "Melbourne",
    "Toorak Park": "Melbourne",
    "Euroa": "Euroa",
    "Coburg Oval": "Melbourne",
    "Brisbane Exhibition": "Brisbane",
    "North Hobart": "Hobart",
    "Bruce Stadium": "Canberra",
    "Yallourn": "Yallourn",
    "Cazaly's Stadium": "Cairns",
    "Eureka Stadium": "Ballarat",
    "Blacktown": "Sydney",
    "Jiangwan Stadium": "Shanghai",
    "Albury": "Albury",
    "Riverway Stadium": "Townsville",
    # Footywire venues
    "AAMI Stadium": "Adelaide",
    "ANZ Stadium": "Sydney",
    "UTAS Stadium": "Launceston",
    "Blacktown International": "Sydney",
    "Blundstone Arena": "Hobart",
    "Domain Stadium": "Perth",
    "Etihad Stadium": "Melbourne",
    "GMHBA Stadium": "Geelong",
    "MCG": "Melbourne",
    "Mars Stadium": "Ballarat",
    "Metricon Stadium": "Gold Coast",
    "Optus Stadium": "Perth",
    "SCG": "Sydney",
    "Spotless Stadium": "Sydney",
    "TIO Stadium": "Darwin",
    "Westpac Stadium": "Wellington",
    "Marvel Stadium": "Melbourne",
    "Canberra Oval": "Canberra",
    "TIO Traeger Park": "Alice Springs",
    # Correct spelling is 'Traeger', but footywire.com is spelling it 'Traegar' in its
    # fixtures, so including both in case they eventually fix the misspelling
    "TIO Traegar Park": "Alice Springs",
}

DEFUNCT_TEAM_NAMES = ["Fitzroy", "University"]
TEAM_NAMES = sorted(DEFUNCT_TEAM_NAMES + list(set(TEAM_TRANSLATIONS.values())))

VENUES = list(set(VENUE_CITIES.keys()))
VENUE_TIMEZONES = {
    venue: CITIES[city]["timezone"] for venue, city in VENUE_CITIES.items()
}

ROUND_TYPES = ["Finals", "Regular"]
INDEX_COLS = ["team", "year", "round_number"]
SEED = 42
AVG_SEASON_LENGTH = 23
CATEGORY_COLS = ["team", "oppo_team", "round_type", "venue"]
TRAIN_YEAR_RANGE = (2017,)
VALIDATION_YEAR_RANGE = (2017, 2019)
CV_YEAR_RANGE = (2014, 2019)
TEST_YEAR_RANGE = (2019, 2020)
