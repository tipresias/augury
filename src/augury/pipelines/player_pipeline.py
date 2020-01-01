"""Functions for creating Kedro pipelines that load and process AFL player data."""

from kedro.pipeline import Pipeline, node

from augury.nodes import common, player, feature_calculation


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


def create_past_player_pipeline():
    """Create a pipeline that loads and cleans historical player data."""
    return Pipeline(
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
                ["combined_past_player_data", "clean_past_match_data"],
                "clean_player_data",
            ),
        ]
    )


def create_roster_pipeline():
    """Create a pipeline that loads and cleans player data for future matches."""
    return Pipeline(
        [
            node(common.convert_to_data_frame, "roster_data", "roster_data_frame"),
            node(
                player.clean_roster_data,
                ["roster_data_frame", "clean_player_data"],
                "clean_roster_data",
            ),
        ]
    )


def create_player_pipeline(
    start_date: str, end_date: str, past_match_pipeline=Pipeline([]), **_kwargs
):
    """
    Create a Kedro pipeline for loading and transforming player data.

    Args:
        start_date (str): Stringified date (yyyy-mm-dd)
        end_date (str): Stringified date (yyyy-mm-dd)
        past_match_pipeline (kedro.pipeline.Pipeline): Pipeline for past match data,
            required for player data cleaning if player pipeline is run in isolation.
    """
    return Pipeline(
        [
            past_match_pipeline,
            create_past_player_pipeline(),
            create_roster_pipeline(),
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
            node(common.finalize_data, "oppo_player_data", "prefinal_player_data"),
            node(common.convert_to_json, "prefinal_player_data", "final_player_data"),
        ]
    )
