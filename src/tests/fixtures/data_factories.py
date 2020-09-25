"""Functions for generating dummy data for use in tests."""

from typing import List, Dict, Tuple, Any, Union, cast
from datetime import datetime, timedelta, date
import itertools

from faker import Faker
import numpy as np
import pandas as pd
from mypy_extensions import TypedDict

from augury.settings import (
    TEAM_NAMES,
    DEFUNCT_TEAM_NAMES,
    VENUE_CITIES,
    TEAM_TRANSLATIONS,
)


FixtureData = TypedDict(
    "FixtureData",
    {
        "date": str,
        "season": int,
        "round": int,
        "home_team": str,
        "away_team": str,
        "venue": str,
    },
)

CleanFixtureData = TypedDict(
    "CleanFixtureData",
    {
        "date": datetime,
        "season": int,
        "round": int,
        "home_team": str,
        "away_team": str,
        "venue": str,
    },
)

CleanedMatchData = TypedDict(
    "CleanedMatchData",
    {
        "date": datetime,
        "year": int,
        "round_number": int,
        "team": str,
        "oppo_team": str,
        "score": int,
        "oppo_score": int,
    },
)

MatchData = TypedDict(
    "MatchData",
    {
        "date": Union[datetime, pd.Timestamp],
        "season": int,
        "round": str,
        "round_number": int,
        "home_team": str,
        "away_team": str,
        "venue": str,
        "home_score": int,
        "away_score": int,
        "match_id": int,
        "crowd": int,
    },
)

FIRST = 1
SECOND = 2
JAN = 1
DEC = 12
THIRTY_FIRST = 31
FAKE = Faker()
CONTEMPORARY_TEAM_NAMES = [
    name for name in TEAM_NAMES if name not in DEFUNCT_TEAM_NAMES
]
BASELINE_BET_PAYOUT = 1.92

ROSTER_COLS = [
    "player_name",
    "playing_for",
    "home_team",
    "away_team",
    "date",
    "match_id",
    "season",
]

MATCH_RESULTS_COLS = [
    "date",
    "tz",
    "updated",
    "round",
    "roundname",
    "year",
    "hbehinds",
    "hgoals",
    "hscore",
    "hteam",
    "hteamid",
    "abehinds",
    "agoals",
    "ascore",
    "ateam",
    "ateamid",
    "winner",
    "winnerteamid",
    "is_grand_final",
    "complete",
    "is_final",
    "id",
    "venue",
]


class CyclicalTeamNames:
    """Cycles through valid team names for use in data fixtures.

    Sometimes necessary due to team-name validations.
    """

    def __init__(self, team_names: List[str] = CONTEMPORARY_TEAM_NAMES):
        """Instantiate a CyclicalTeamNames object.

        Params
            team_names: List of team names to cycle through.
        """
        self.team_names = team_names
        self.cyclical_team_names = (name for name in self.team_names)

    def next_team(self) -> str:
        """Return the next available team name."""
        try:
            return next(self.cyclical_team_names)
        except StopIteration:
            self.cyclical_team_names = (name for name in self.team_names)

            return next(self.cyclical_team_names)


def _min_max_datetimes_by_year(
    year: int, force_future: bool = False
) -> Dict[str, datetime]:
    # About as early as matches ever start
    MIN_MATCH_HOUR = 12
    # About as late as matches ever start
    MAX_MATCH_HOUR = 20

    if force_future:
        today = datetime.now()

        # Running tests on 28 Feb of a leap year breaks them, because the given year
        # generally won't be a leap year (e.g. 2018-2-29 doesn't exist),
        # so we retry with two days in the future (e.g. 2018-3-1).
        try:
            tomorrow = today + timedelta(hours=24)
            datetime_start = datetime(
                year, tomorrow.month, tomorrow.day, MIN_MATCH_HOUR
            )
        except ValueError:
            tomorrow = today + timedelta(hours=48)
            datetime_start = datetime(
                year, tomorrow.month, tomorrow.day, MIN_MATCH_HOUR
            )
    else:
        datetime_start = datetime(year, JAN, FIRST, MIN_MATCH_HOUR)

    return {
        "datetime_start": datetime_start,
        "datetime_end": datetime(year, DEC, THIRTY_FIRST, MAX_MATCH_HOUR),
    }


def _raw_match_data(year: int, team_names: Tuple[str, str], idx: int) -> MatchData:
    return cast(
        MatchData,
        {
            "date": str(
                FAKE.date_time_between_dates(**_min_max_datetimes_by_year(year))
            ),
            "season": year,
            "round": "R1",
            "round_number": round(idx / (len(CONTEMPORARY_TEAM_NAMES) / 2)) + 1,
            "home_team": team_names[0],
            "away_team": team_names[1],
            "venue": np.random.choice(list(VENUE_CITIES.keys())),
            "home_score": np.random.randint(50, 150),
            "away_score": np.random.randint(50, 150),
            "match_id": FAKE.ean(),
            "crowd": np.random.randint(10000, 30000),
        },
    )


def _match_data(year: int, team_names: Tuple[str, str], idx: int) -> CleanedMatchData:
    return cast(
        CleanedMatchData,
        {
            "date": FAKE.date_time_between_dates(**_min_max_datetimes_by_year(year)),
            "year": year,
            "round_number": round(idx / (len(CONTEMPORARY_TEAM_NAMES) / 2)) + 1,
            "team": team_names[0],
            "oppo_team": team_names[1],
            "score": np.random.randint(50, 150),
            "oppo_score": np.random.randint(50, 150),
        },
    )


def _matches_by_round(
    match_count_per_year: int, year: int, raw=False
) -> Union[List[CleanedMatchData], List[MatchData]]:
    team_names = CyclicalTeamNames()

    if raw:
        return [
            _raw_match_data(year, (team_names.next_team(), team_names.next_team()), idx)
            for idx in range(match_count_per_year)
        ]

    return [
        _match_data(year, (team_names.next_team(), team_names.next_team()), idx)
        for idx in range(match_count_per_year)
    ]


def _matches_by_year(
    match_count_per_year: int, year_range: Tuple[int, int], raw=False
) -> List[Union[List[CleanedMatchData], List[MatchData]]]:
    return [
        _matches_by_round(match_count_per_year, year, raw=raw)
        for year in range(*year_range)
    ]


def _oppo_match_data(team_match: CleanedMatchData) -> CleanedMatchData:
    return cast(
        CleanedMatchData,
        {
            **team_match,
            **{
                "team": team_match["oppo_team"],
                "oppo_team": team_match["team"],
                "score": team_match["oppo_score"],
                "oppo_score": team_match["score"],
            },
        },
    )


def _add_oppo_rows(match_data: List[CleanedMatchData]) -> List[CleanedMatchData]:
    data = [[match, _oppo_match_data(match)] for match in match_data]

    return list(itertools.chain.from_iterable(data))


def fake_match_results_data(
    match_data: pd.DataFrame, round_number: int  # pylint: disable=unused-argument
) -> pd.DataFrame:
    """
    Generate dummy data that replicates match results data from the Squiggle API.

    Params
    ------
    row_count: Number of match rows to return

    Returns
    -------
    DataFrame of match results data
    """
    assert (
        len(match_data["season"].drop_duplicates()) == 1
    ), "Match results data is fetched one season at a time."

    return (
        match_data.query("round_number == @round_number")
        .assign(
            updated=lambda df: pd.to_datetime(df["date"]) + timedelta(hours=3),
            tz="+10:00",
            # AFLTables match_results already have a 'round' column,
            # so we have to replace rather than rename.
            round=lambda df: df["round_number"],
            roundname=lambda df: "Round " + df["round_number"].astype(str),
            hteam=lambda df: df["home_team"].map(
                lambda team: TEAM_TRANSLATIONS.get(team, team)
            ),
            ateam=lambda df: df["away_team"].map(
                lambda team: TEAM_TRANSLATIONS.get(team, team)
            ),
            hteamid=lambda df: df["hteam"].map(TEAM_NAMES.index),
            ateamid=lambda df: df["ateam"].map(TEAM_NAMES.index),
            winner=lambda df: np.where(df["margin"] >= 0, df["hteam"], df["ateam"]),
            winnerteamid=lambda df: df["winner"].map(TEAM_NAMES.index),
            is_grand_final=0,
            complete=100,
            is_final=0,
        )
        .astype({"updated": str})
        .reset_index(drop=False)
        .rename(
            columns={
                "index": "id",
                "season": "year",
                "home_behinds": "hbehinds",
                "home_goals": "hgoals",
                "away_behinds": "abehinds",
                "away_goals": "agoals",
                "home_points": "hscore",
                "away_points": "ascore",
            }
        )
    ).loc[:, MATCH_RESULTS_COLS]


def _players_by_match(
    match_data: CleanedMatchData, n_players: int, idx: int
) -> List[Dict[str, Any]]:
    # Assumes that both team and oppo_team rows are present and that they alternate
    # in order to evenly split players between the two
    playing_for = match_data["team"] if idx % 2 == 0 else match_data["oppo_team"]

    return [
        {
            **match_data,
            **{
                "player_id": FAKE.ean(),
                "player_name": FAKE.name(),
                "playing_for": playing_for,
            },
        }
        for _ in range(n_players)
    ]


def fake_roster_data(match_count: int, n_players_per_team: int) -> pd.DataFrame:
    """Generate dummy data that replicates clean roster data.

    This represents player data for future matches after it has passed through
    the initial cleaning node `player.clean_roster_data`.
    """
    this_year = date.today().year
    match_data = cast(
        List[List[CleanedMatchData]],
        _matches_by_year(match_count, (this_year, this_year + 1)),
    )
    reduced_match_data = list(itertools.chain.from_iterable(match_data))

    roster_data = [
        _players_by_match(match_data, n_players_per_team, idx)
        for idx, match_data in enumerate(_add_oppo_rows(reduced_match_data))
    ]
    reduced_roster_data = list(itertools.chain.from_iterable(roster_data))

    return pd.DataFrame(reduced_roster_data).loc[:, ROSTER_COLS].sort_index()
