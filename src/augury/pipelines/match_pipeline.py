"""Functions for creating Kedro pipelines for match-specific data."""

from kedro.pipeline import Pipeline, node

from augury.nodes import common, match, feature_calculation


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
    "result",
    "oppo_result",
    "margin",
    "oppo_margin",
    "out_of_state",
    "at_home",
    "oppo_team",
    "venue",
    "round_type",
    "date",
]


def create_past_match_pipeline():
    """Create Kedro pipeline for match data to the end of last year."""
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


def create_future_match_pipeline():
    """Create a pipeline for loading and cleaning fixture (i.e. future matches) data."""
    return Pipeline(
        [
            node(common.convert_to_data_frame, "fixture_data", "fixture_data_frame"),
            node(match.clean_fixture_data, "fixture_data_frame", "clean_fixture_data"),
        ]
    )


def create_match_pipeline(
    start_date: str,
    end_date: str,
    past_match_pipeline=create_past_match_pipeline(),
    **_kwargs
):
    """
    Create a Kedro pipeline for loading and transforming match data.

    Args:
        start_date (str, YYYY-MM-DD format): Earliest date for included data.
        end_date (str, YYYY-MM-DD format): Latest date for included data.
        past_match_pipeline (kedro.pipeline.Pipeline): Pipeline for loading and
            cleaning data for past matches.
    """
    return Pipeline(
        [
            past_match_pipeline,
            create_future_match_pipeline(),
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
            node(
                common.convert_match_rows_to_teammatch_rows,
                "filtered_past_match_data",
                "match_data_a",
            ),
            node(match.add_out_of_state, "match_data_a", "match_data_b"),
            node(match.add_travel_distance, "match_data_b", "match_data_c"),
            node(match.add_result, "match_data_c", "match_data_d"),
            node(match.add_margin, "match_data_d", "match_data_e"),
            node(
                match.add_shifted_team_features(
                    shift_columns=[
                        "score",
                        "oppo_score",
                        "result",
                        "margin",
                        "team_goals",
                        "team_behinds",
                        "oppo_team",
                        "at_home",
                    ]
                ),
                "match_data_e",
                "shifted_match_data",
            ),
            node(match.add_cum_win_points, "shifted_match_data", "match_data_g"),
            node(match.add_win_streak, "match_data_g", "match_data_h"),
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
                "match_data_h",
                "match_data_i",
            ),
            node(
                common.add_oppo_features(match_cols=MATCH_OPPO_COLS),
                "match_data_i",
                "match_data_j",
            ),
            # Features dependent on oppo columns
            node(match.add_cum_percent, "match_data_j", "match_data_k"),
            node(match.add_ladder_position, "match_data_k", "match_data_l"),
            node(
                common.add_oppo_features(
                    oppo_feature_cols=["cum_percent", "ladder_position"]
                ),
                "match_data_l",
                "match_data_m",
            ),
            node(common.finalize_data, "match_data_m", "prefinal_match_data"),
            node(common.convert_to_json, "prefinal_match_data", "final_match_data"),
        ]
    )


def create_legacy_match_pipeline(
    start_date: str,
    end_date: str,
    past_match_pipeline=create_past_match_pipeline(),
    **_kwargs
):
    """
    Create a pipeline for match data with Elo features included.

    Only relevant for generating predictions from older models.

    Args:
        start_date (str, YYYY-MM-DD format): Earliest date for included data.
        end_date (str, YYYY-MM-DD format): Latest date for included data.
        past_match_pipeline (kedro.pipeline.Pipeline): Pipeline for loading and
            cleaning data for past matches.
    """
    return Pipeline(
        [
            past_match_pipeline,
            create_future_match_pipeline(),
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
                common.add_oppo_features(
                    match_cols=MATCH_OPPO_COLS + ["elo_rating", "oppo_elo_rating"]
                ),
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
            node(common.finalize_data, "match_data_p", "prefinal_legacy_match_data"),
            node(
                common.convert_to_json,
                "prefinal_legacy_match_data",
                "final_legacy_match_data",
            ),
        ]
    )
