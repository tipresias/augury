"""Module for fetching player data from afl_data service."""

from typing import List, Dict, Any
from datetime import date, datetime, timedelta
import itertools
import math
from functools import partial
import os
import json

from machine_learning.data_import.base_data import fetch_afl_data
from machine_learning.settings import RAW_DATA_DIR, PREDICTION_DATA_START_DATE


# Player stats go back to 1897, but before 1965, there are only goals & behinds, which
# make for a very sparse data set and needlessly slow down training without adding
# much value
EARLIEST_SEASON_WITH_EXTENSIVE_PLAYER_STATS = "1965"
# This is the max number of season's worth of player data (give or take) that GCR
# can handle without blowing up
MAX_YEAR_COUNT_FOR_PLAYER_DATA = 3
END_OF_LAST_YEAR = f"{date.today().year - 1}-12-31"


def _date_range(
    start_date: datetime, end_date: datetime, time_spread: timedelta, period: int
):
    range_start = start_date + (time_spread * period)
    range_end = min(range_start + time_spread - timedelta(days=1), end_date)

    return (str(range_start.date()), str(range_end.date()))


def _fetch_player_stats_batch(
    start_date: str, end_date: str, verbose: int = 1
) -> List[Dict[str, Any]]:  # Just being lazy on the definition
    if verbose == 1:
        print(f"\tFetching player data from between {start_date} and " f"{end_date}...")

    data = fetch_afl_data(
        "/players", params={"start_date": start_date, "end_date": end_date}
    )

    if verbose == 1:
        print(f"\tPlayer data for {start_date} to {end_date} received!\n")

    return data


def _player_batch_date_ranges(start_date: str, end_date: str):
    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    time_spread = timedelta(days=(MAX_YEAR_COUNT_FOR_PLAYER_DATA * 365))
    year_spread = (end_date_dt - start_date_dt) / time_spread

    date_range = partial(_date_range, start_date_dt, end_date_dt, time_spread)

    return [date_range(period) for period in range(math.ceil(year_spread))]


def fetch_player_data(
    start_date: str = f"{EARLIEST_SEASON_WITH_EXTENSIVE_PLAYER_STATS}-01-01",
    end_date: str = str(date.today()),
    verbose: int = 1,
) -> List[Dict[str, Any]]:
    """
    Get player data from AFL tables

    Args:
        start_date (string: YYYY-MM-DD): Earliest date for match data returned.
        end_date (string: YYYY-MM-DD): Latest date for match data returned.
        verbose (int): Whether to print info statements (1 means yes, 0 means no).

    Returns:
    list of dicts of player data.
    """

    if verbose == 1:
        print(
            f"Fetching player data from between {start_date} and {end_date} "
            "in yearly baches..."
        )

    data_batch_date_ranges = _player_batch_date_ranges(start_date, end_date)
    partial_fetch_player_stats_batch = partial(
        _fetch_player_stats_batch, verbose=verbose
    )

    # Google Cloud Run cannot handle such a large data set in its response, so we
    # fetch it in batches. With the implementation of kedro pipelines, we should
    # usually read historical data from files or Google Cloud Storage, so the slowness
    # of this isn't much of an issue.
    data = itertools.chain.from_iterable(
        [
            partial_fetch_player_stats_batch(*date_pair)
            for date_pair in data_batch_date_ranges
        ]
    )

    if verbose == 1:
        print("All player data received!")

    return list(data)


def save_player_data(
    start_date: str = f"{EARLIEST_SEASON_WITH_EXTENSIVE_PLAYER_STATS}-01-01",
    end_date: str = END_OF_LAST_YEAR,
    verbose: int = 1,
    for_prod: bool = False,
) -> None:
    """
    Save match data as a *.json file with name based on date range of data

    Args:
        start_date (string: YYYY-MM-DD): Earliest date for match data returned.
        end_date (string: YYYY-MM-DD): Latest date for match data returned.
        verbose (int): Whether to print info statements (1 means yes, 0 means no).
        for_prod (bool): Whether saved data set is meant for loading in production.
            If True, this overwrites the given start_date to limit the data set
            to the last 10 years to limit memory usage.

    Returns:
        None
    """

    if for_prod:
        start_date = max(start_date, PREDICTION_DATA_START_DATE)

    data = fetch_player_data(start_date=start_date, end_date=end_date, verbose=verbose)
    filepath = os.path.join(RAW_DATA_DIR, f"player-data_{start_date}_{end_date}.json")

    with open(filepath, "w") as json_file:
        json.dump(data, json_file, indent=2)

    if verbose == 1:
        print("Player data saved")


def fetch_roster_data(
    round_number: int = None, verbose: int = 1
) -> List[Dict[str, Any]]:
    """
    Get player data from AFL tables

    Args:
        start_date (string: YYYY-MM-DD): Earliest date for match data returned.
        end_date (string: YYYY-MM-DD): Latest date for match data returned.
        verbose (int): Whether to print info statements (1 means yes, 0 means no).

    Returns:
    list of dicts of player data.
    """

    if verbose == 1:
        print(f"Fetching roster data for round {round_number}...")

    data = fetch_afl_data("/rosters", params={"round_number": round_number})

    if verbose == 1:
        if not any(data):
            print(
                "No roster data was received. It's likely that the team roster page "
                "hasn't been updated for the upcoming round."
            )
        else:
            print("Roster data received!")

    return data


if __name__ == "__main__":
    last_year = date.today().year - 1
    end_of_last_year = f"{last_year}-12-31"

    # A bit arbitrary, but in general I prefer to keep the static, raw data up to the
    # end of last season, fetching more recent data as necessary
    save_player_data(end_date=end_of_last_year)
