"""Pipeline construction."""

from typing import Dict

from kedro.pipeline import Pipeline, node

from machine_learning.settings import CATEGORY_COLS
from .nodes import betting, common, match, player, feature_calculation


MATCH_OPPO_COLS = [
    "team",
    "year",
    "round_number",
    "score",
    "oppo_score",
    "team_goals",
    "oppo_team_goals",
    "team_behinds",
    "oppo_team_behinds",
    # TODO: I have to omit these columns, because I accidentally left them in
    # when building betting features, and I need the columns to be the same
    # in order not to retrain my saved models.
    # "result",
    # "oppo_result",
    "margin",
    "oppo_margin",
    "elo_rating",
    "oppo_elo_rating",
    "out_of_state",
    "at_home",
    "oppo_team",
    "venue",
    "round_type",
    "date",
]

PLAYER_MATCH_STATS_COLS = [
    "at_home",
    "score",
    "oppo_score",
    "team",
    "oppo_team",
    "year",
    "round_number",
    "date",
]


def create_pipelines(start_date, end_date, **_kwargs) -> Dict[str, Pipeline]:
    """Create the project's pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """

    return {
        "__default__": Pipeline([]),
        "betting": betting_pipeline(start_date, end_date),
        "match": match_pipeline(start_date, end_date),
        "player": player_pipeline(start_date, end_date),
        "full": create_full_pipeline(start_date, end_date),
        "fake": fake_estimator_pipeline(),
    }


def betting_pipeline(start_date: str, end_date: str, **_kwargs):
    """Kedro pipeline for loading and transforming betting data"""

    return Pipeline(
        [
            node(
                common.convert_to_data_frame,
                ["betting_data", "remote_betting_data"],
                ["betting_data_frame", "remote_betting_data_frame"],
            ),
            node(
                common.combine_data(axis=0),
                ["betting_data_frame", "remote_betting_data_frame"],
                "combined_betting_data",
            ),
            node(betting.clean_data, "combined_betting_data", "clean_betting_data"),
            node(
                common.filter_by_date(start_date, end_date),
                "clean_betting_data",
                "filtered_betting_data",
            ),
            node(
                common.convert_match_rows_to_teammatch_rows,
                "filtered_betting_data",
                "stacked_betting_data",
            ),
            node(
                betting.add_betting_pred_win, ["stacked_betting_data"], "betting_data_a"
            ),
            node(
                feature_calculation.feature_calculator(
                    [
                        (
                            feature_calculation.calculate_rolling_rate,
                            [("betting_pred_win",)],
                        )
                    ]
                ),
                ["betting_data_a"],
                "betting_data_b",
            ),
            node(
                common.add_oppo_features(
                    oppo_feature_cols=[
                        "betting_pred_win",
                        "rolling_betting_pred_win_rate",
                    ]
                ),
                ["betting_data_b"],
                "betting_data_c",
            ),
            node(common.finalize_data, ["betting_data_c"], "final_betting_data"),
        ]
    )


def create_past_match_pipeline():
    """Kedro pipeline for match data to the end of last year"""

    return Pipeline(
        [
            node(
                common.convert_to_data_frame,
                ["match_data", "remote_match_data"],
                ["match_data_frame", "remote_match_data_frame"],
            ),
            node(
                common.combine_data(axis=0),
                ["match_data_frame", "remote_match_data_frame"],
                "combined_past_match_data",
            ),
            node(
                match.clean_match_data,
                "combined_past_match_data",
                "clean_past_match_data",
            ),
        ]
    )


def match_pipeline(
    start_date: str,
    end_date: str,
    past_match_pipeline=create_past_match_pipeline(),
    **_kwargs
):
    """Kedro pipeline for loading and transforming match data"""

    upcoming_match_pipeline = Pipeline(
        [
            node(common.convert_to_data_frame, "fixture_data", "fixture_data_frame"),
            node(match.clean_fixture_data, "fixture_data_frame", "clean_fixture_data"),
        ]
    )

    return Pipeline(
        [
            past_match_pipeline,
            upcoming_match_pipeline,
            node(
                common.combine_data(axis=0),
                ["clean_past_match_data", "clean_fixture_data"],
                "combined_match_data",
            ),
            node(
                common.filter_by_date(start_date, end_date),
                "combined_match_data",
                "filtered_past_match_data",
            ),
            # add_elo_rating depends on DF still being organized per-match
            # with home_team/away_team columns
            node(match.add_elo_rating, "filtered_past_match_data", "match_data_a"),
            node(
                common.convert_match_rows_to_teammatch_rows,
                "match_data_a",
                "match_data_b",
            ),
            node(match.add_out_of_state, "match_data_b", "match_data_c"),
            node(match.add_travel_distance, "match_data_c", "match_data_d"),
            node(match.add_result, "match_data_d", "match_data_e"),
            node(match.add_margin, "match_data_e", "match_data_f"),
            node(
                match.add_shifted_team_features(
                    shift_columns=[
                        "score",
                        "oppo_score",
                        "result",
                        "margin",
                        "team_goals",
                        "team_behinds",
                    ]
                ),
                "match_data_f",
                "match_data_g",
            ),
            node(match.add_cum_win_points, "match_data_g", "match_data_h"),
            node(match.add_win_streak, "match_data_h", "match_data_i"),
            node(
                feature_calculation.feature_calculator(
                    [
                        (
                            feature_calculation.calculate_rolling_rate,
                            [("prev_match_result",)],
                        ),
                        (
                            feature_calculation.calculate_rolling_mean_by_dimension,
                            [
                                ("oppo_team", "margin"),
                                ("oppo_team", "result"),
                                ("oppo_team", "score"),
                                ("venue", "margin"),
                                ("venue", "result"),
                                ("venue", "score"),
                            ],
                        ),
                    ]
                ),
                "match_data_i",
                "match_data_j",
            ),
            node(
                common.add_oppo_features(match_cols=MATCH_OPPO_COLS),
                "match_data_j",
                "match_data_k",
            ),
            # Features dependent on oppo columns
            node(match.add_cum_percent, "match_data_k", "match_data_l"),
            node(match.add_ladder_position, "match_data_l", "match_data_m"),
            node(match.add_elo_pred_win, "match_data_m", "match_data_n"),
            node(
                feature_calculation.feature_calculator(
                    [
                        (
                            feature_calculation.calculate_rolling_rate,
                            [("elo_pred_win",)],
                        ),
                        (
                            feature_calculation.calculate_division,
                            [("elo_rating", "ladder_position")],
                        ),
                    ]
                ),
                "match_data_n",
                "match_data_o",
            ),
            node(
                common.add_oppo_features(
                    oppo_feature_cols=["cum_percent", "ladder_position"]
                ),
                "match_data_o",
                "match_data_p",
            ),
            node(common.finalize_data, "match_data_p", "final_match_data"),
        ]
    )


def player_pipeline(
    start_date: str, end_date: str, past_match_pipeline=Pipeline([]), **_kwargs
):
    """
    Kedro pipeline for loading and transforming player data.

    Args:
        start_date (str): Stringified date (yyyy-mm-dd)
    """

    past_player_pipeline = Pipeline(
        [
            node(
                common.convert_to_data_frame,
                ["player_data", "remote_player_data"],
                ["player_data_frame", "remote_player_data_frame"],
            ),
            node(
                common.combine_data(axis=0),
                ["player_data_frame", "remote_player_data_frame"],
                "combined_past_player_data",
            ),
            node(
                player.clean_player_data,
                ["combined_past_player_data", "combined_past_match_data"],
                "clean_player_data",
            ),
        ]
    )

    roster_pipeline = Pipeline(
        [
            node(common.convert_to_data_frame, "roster_data", "roster_data_frame"),
            node(
                player.clean_roster_data,
                ["roster_data_frame", "clean_player_data"],
                "clean_roster_data",
            ),
        ]
    )

    return Pipeline(
        [
            past_match_pipeline,
            past_player_pipeline,
            roster_pipeline,
            node(
                common.combine_data(axis=0),
                ["clean_player_data", "clean_roster_data"],
                "combined_player_data",
            ),
            node(
                common.filter_by_date(start_date, end_date),
                "combined_player_data",
                "filtered_player_data",
            ),
            node(
                player.convert_player_match_rows_to_player_teammatch_rows,
                "filtered_player_data",
                "stacked_player_data",
            ),
            node(
                player.add_last_year_brownlow_votes,
                "stacked_player_data",
                "player_data_a",
            ),
            node(player.add_rolling_player_stats, "player_data_a", "player_data_b"),
            node(player.add_cum_matches_played, "player_data_b", "player_data_c"),
            node(
                feature_calculation.feature_calculator(
                    [
                        (
                            feature_calculation.calculate_addition,
                            [
                                (
                                    "rolling_prev_match_goals",
                                    "rolling_prev_match_behinds",
                                )
                            ],
                        )
                    ]
                ),
                "player_data_c",
                "player_data_d",
            ),
            node(
                feature_calculation.feature_calculator(
                    [
                        (
                            feature_calculation.calculate_division,
                            [
                                (
                                    "rolling_prev_match_goals",
                                    "rolling_prev_match_goals_plus_rolling_prev_match_behinds",
                                )
                            ],
                        )
                    ]
                ),
                "player_data_d",
                "player_data_e",
            ),
            node(
                player.aggregate_player_stats_by_team_match(
                    ["sum", "max", "min", "skew", "std"]
                ),
                "player_data_e",
                "aggregated_player_data",
            ),
            node(
                common.add_oppo_features(match_cols=PLAYER_MATCH_STATS_COLS),
                "aggregated_player_data",
                "oppo_player_data",
            ),
            node(common.finalize_data, "oppo_player_data", "final_player_data"),
        ]
    )


def create_full_pipeline(start_date: str, end_date: str, **_kwargs):
    return Pipeline(
        [
            betting_pipeline(start_date, end_date),
            match_pipeline(start_date, end_date),
            player_pipeline(start_date, end_date),
            node(
                common.combine_data(axis=1),
                ["final_betting_data", "final_match_data", "final_player_data"],
                "joined_data",
            ),
            node(
                feature_calculation.feature_calculator(
                    [
                        (
                            feature_calculation.calculate_division,
                            [("elo_rating", "win_odds")],
                        ),
                        (
                            feature_calculation.calculate_multiplication,
                            [("win_odds", "ladder_position")],
                        ),
                    ]
                ),
                "joined_data",
                "data_a",
            ),
            node(
                common.sort_data_frame_columns(category_cols=CATEGORY_COLS),
                "data_a",
                "data_b",
            ),
            node(common.finalize_joined_data, "data_b", "data"),
        ]
    )


def fake_estimator_pipeline(*_args, **_kwargs):
    """Kedro pipeline for loading and transforming match data for test estimator"""

    return Pipeline(
        [
            node(common.convert_to_data_frame, "fake_match_data", "match_data_frame"),
            node(match.clean_match_data, "match_data_frame", "clean_match_data"),
            node(
                common.convert_match_rows_to_teammatch_rows,
                "clean_match_data",
                "match_data_b",
            ),
            node(match.add_out_of_state, "match_data_b", "match_data_c"),
            node(match.add_travel_distance, "match_data_c", "match_data_d"),
            node(match.add_result, "match_data_d", "match_data_e"),
            node(match.add_margin, "match_data_e", "match_data_f"),
            node(
                match.add_shifted_team_features(
                    shift_columns=[
                        "score",
                        "oppo_score",
                        "result",
                        "margin",
                        "team_goals",
                        "team_behinds",
                    ]
                ),
                "match_data_f",
                "match_data_g",
            ),
            node(match.add_cum_win_points, "match_data_g", "match_data_h"),
            node(match.add_win_streak, "match_data_h", "data"),
        ]
    )
