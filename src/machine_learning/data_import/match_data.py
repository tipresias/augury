"""Module for fetching match data from afl_data service"""

from typing import List, Dict, Any
from datetime import date
import os
import json

from machine_learning.data_import.base_data import fetch_afl_data
from machine_learning.settings import RAW_DATA_DIR

FIRST_YEAR_OF_MATCH_DATA = 1897
END_OF_YEAR = f"{date.today().year}-12-31"


def fetch_match_data(
    start_date: str = f"{FIRST_YEAR_OF_MATCH_DATA}-01-01",
    end_date: str = END_OF_YEAR,
    verbose: int = 1,
) -> List[Dict[str, Any]]:
    """
    Get AFL match data for given date range.

    Args:
        start_date (string: YYYY-MM-DD): Earliest date for match data returned.
        end_date (string: YYYY-MM-DD): Latest date for match data returned.

    Returns
        list of dicts of match data.
    """

    if verbose == 1:
        print("Fetching match data from between " f"{start_date} and {end_date}...")

    data = fetch_afl_data(
        "/matches", params={"start_date": start_date, "end_date": end_date}
    )

    if verbose == 1:
        print("Match data received!")

    return data


def save_match_data(
    start_date: str = f"{FIRST_YEAR_OF_MATCH_DATA}-01-01",
    end_date: str = END_OF_YEAR,
    verbose: int = 1,
) -> None:
    """Save match data as a *.json file with name based on date range of data"""

    data = fetch_match_data(start_date=start_date, end_date=end_date, verbose=verbose)
    filepath = os.path.join(RAW_DATA_DIR, f"match-data_{start_date}_{end_date}.json")

    with open(filepath, "w") as json_file:
        json.dump(data, json_file, indent=2)

    if verbose == 1:
        print("Match data saved")


if __name__ == "__main__":
    last_year = date.today().year - 1
    end_of_last_year = f"{last_year}-12-31"

    # A bit arbitrary, but in general I prefer to keep the static, raw data up to the
    # end of last season, fetching more recent data as necessary
    save_match_data(end_date=end_of_last_year)
