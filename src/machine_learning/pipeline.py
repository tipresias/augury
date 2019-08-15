"""Pipeline construction."""

from kedro.pipeline import Pipeline, node

from machine_learning.data_processors.feature_calculation import (
    feature_calculator,
    calculate_rolling_rate,
    calculate_rolling_mean_by_dimension,
    calculate_division,
)
from .nodes import betting, common, match, player

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
            node(
                common.filter_by_date(start_date, end_date),
                "combined_betting_data",
                "filtered_betting_data",
            ),
            node(betting.clean_data, "filtered_betting_data", "clean_betting_data"),
            node(
                common.convert_match_rows_to_teammatch_rows,
                ["clean_betting_data"],
                "stacked_betting_data",
            ),
            node(
                betting.add_betting_pred_win, ["stacked_betting_data"], "betting_data_a"
            ),
            node(
                feature_calculator([(calculate_rolling_rate, [("betting_pred_win",)])]),
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
            node(betting.finalize_data, ["betting_data_c"], "data"),
        ]
    )


def match_pipeline(start_date: str, end_date: str, **_kwargs):
    """Kedro pipeline for loading and transforming match data"""

    past_match_pipeline = Pipeline(
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
                common.filter_by_date(start_date, end_date),
                "combined_past_match_data",
                "filtered_past_match_data",
            ),
            node(
                match.clean_match_data,
                "filtered_past_match_data",
                "clean_past_match_data",
            ),
        ]
    )

    upcoming_match_pipeline = Pipeline(
        [
            node(common.convert_to_data_frame, "fixture_data", "fixture_data_frame"),
            node(
                common.filter_by_date(start_date, end_date),
                "fixture_data_frame",
                "filtered_fixture_data_frame",
            ),
            node(
                match.clean_fixture_data,
                "filtered_fixture_data_frame",
                "clean_fixture_data",
            ),
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
            # add_elo_rating depends on DF still being organized per-match
            # with home_team/away_team columns
            node(match.add_elo_rating, "combined_match_data", "match_data_a"),
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
                feature_calculator(
                    [
                        (calculate_rolling_rate, [("prev_match_result",)]),
                        (
                            calculate_rolling_mean_by_dimension,
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
                feature_calculator(
                    [
                        (calculate_rolling_rate, [("elo_pred_win",)]),
                        (calculate_division, [("elo_rating", "ladder_position")]),
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
                "data",
            ),
        ]
    )


def player_pipeline(start_date: str, end_date: str, **_kwargs):
    """Kedro pipeline for loading and transforming player data"""

    past_match_pipeline = Pipeline(
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
                common.filter_by_date(start_date, end_date),
                "combined_past_match_data",
                "filtered_past_match_data",
            ),
        ]
    )

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
                common.filter_by_date(start_date, end_date),
                "combined_past_player_data",
                "filtered_past_player_data",
            ),
            node(
                player.clean_player_data,
                ["filtered_past_player_data", "filtered_past_match_data"],
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
        ]
    )


def fake_estimator_pipeline():
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
