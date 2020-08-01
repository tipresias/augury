"""Module for fetching match data from afl_data service."""

from typing import List, Dict, Any
from datetime import date
import os
import json

from augury.data_import.base_data import fetch_afl_data
from augury.settings import RAW_DATA_DIR, PREDICTION_DATA_START_DATE

FIRST_YEAR_OF_MATCH_DATA = 1897
END_OF_YEAR = f"{date.today().year}-12-31"
END_OF_LAST_YEAR = f"{date.today().year - 1}-12-31"


def fetch_match_data(
    start_date: str = f"{FIRST_YEAR_OF_MATCH_DATA}-01-01",
    end_date: str = END_OF_YEAR,
    verbose: int = 1,
) -> List[Dict[str, Any]]:
    """
    Get AFL match data for given date range.

    Params
    ------
    start_date (string: YYYY-MM-DD): Earliest date for match data returned.
    end_date (string: YYYY-MM-DD): Latest date for match data returned.

    Returns
    -------
    list of dicts of match data.
    """
    if verbose == 1:
        print("Fetching match data from between " f"{start_date} and {end_date}...")

    data = fetch_afl_data(
        "/matches",
        params={"start_date": start_date, "end_date": end_date, "fetch_data": True},
    )

    if verbose == 1:
        print("Match data received!")

    return data


def save_match_data(
    start_date: str = f"{FIRST_YEAR_OF_MATCH_DATA}-01-01",
    end_date: str = END_OF_LAST_YEAR,
    verbose: int = 1,
    for_prod: bool = False,
) -> None:
    """
    Save match data as a *.json file with name based on date range of data.

    Params
    ------
    start_date (string: YYYY-MM-DD): Earliest date for match data returned.
    end_date (string: YYYY-MM-DD): Latest date for match data returned.
    verbose (int): Whether to print info statements (1 means yes, 0 means no).
    for_prod (bool): Whether saved data set is meant for loading in production.
        If True, this overwrites the given start_date to limit the data set
        to the last 10 years to limit memory usage.

    Returns
    -------
    None
    """
    if for_prod:
        start_date = max(start_date, PREDICTION_DATA_START_DATE)

    data = fetch_match_data(start_date=start_date, end_date=end_date, verbose=verbose)
    filepath = os.path.join(RAW_DATA_DIR, f"match-data_{start_date}_{end_date}.json")

    with open(filepath, "w") as json_file:
        json.dump(data, json_file, indent=2)

    if verbose == 1:
        print("Match data saved")


def fetch_fixture_data(
    start_date: str = str(date.today()), end_date: str = END_OF_YEAR, verbose: int = 1
) -> List[Dict[str, Any]]:
    """
    Get AFL fixture data for given date range.

    Params
    ------
    start_date (string: YYYY-MM-DD): Earliest date for fixture data returned.
    end_date (string: YYYY-MM-DD): Latest date for fixture data returned.

    Returns
    -------
    list of dicts of fixture data.
    """
    if verbose == 1:
        print("Fetching fixture data from between " f"{start_date} and {end_date}...")

    data = fetch_afl_data(
        "/fixtures", params={"start_date": start_date, "end_date": end_date}
    )

    if verbose == 1:
        print("Fixture data received!")

    return data


def fetch_match_results_data(
    round_number: int, verbose: int = 1
) -> List[Dict[str, Any]]:
    """
    Get AFL match results for the given round.

    Params
    ------
    round_number: The round number for which to fetch match data.
    verbose: Whether to print status messages.

    Returns
    -------
    list of dicts of match results data
    """
    if verbose == 1:
        print(f"Fetching match results data for round {round_number}...")

    data = fetch_afl_data("/match_results", params={"round_number": round_number})

    if verbose == 1:
        print("Match results data received!")

    return data


if __name__ == "__main__":
    last_year = date.today().year - 1
    end_of_last_year = f"{last_year}-12-31"

    # A bit arbitrary, but in general I prefer to keep the static, raw data up to the
    # end of last season, fetching more recent data as necessary
    save_match_data(end_date=end_of_last_year)
