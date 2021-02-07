"""Pipelines for loading and processing joined data from other pipelines."""

from kedro.pipeline import Pipeline, node

from augury.settings import CATEGORY_COLS
from ..nodes import common, feature_calculation
from .. import player, betting, match


DEFAULT_FEATURE_CALCS = [
    (feature_calculation.calculate_multiplication, [("win_odds", "ladder_position")])
]


def create_pipeline(
    start_date: str,
    end_date: str,
    match_data_set="final_match_data",
    match_pipeline_func=match.create_pipeline,
    feature_calcs=DEFAULT_FEATURE_CALCS,
    final_data_set="model_data",
    category_cols=CATEGORY_COLS + ["prev_match_oppo_team", "oppo_prev_match_oppo_team"],
):
    """Create a pipeline that joins all data-source-specific pipelines."""
    return Pipeline(
        [
            betting.create_pipeline(start_date, end_date),
            match_pipeline_func(start_date, end_date),
            player.create_pipeline(start_date, end_date),
            node(
                common.clean_full_data,
                ["final_betting_data", match_data_set, "final_player_data"],
                [
                    "clean_betting_data_df",
                    "clean_match_data_df",
                    "clean_player_data_df",
                ],
            ),
            node(
                common.combine_data(axis=1),
                [
                    "clean_betting_data_df",
                    "clean_match_data_df",
                    "clean_player_data_df",
                ],
                "joined_data",
                name="joined_data",
            ),
            node(
                common.filter_by_date(start_date, end_date),
                "joined_data",
                "filtered_data",
            ),
            node(
                feature_calculation.feature_calculator(feature_calcs),
                "filtered_data",
                "data_a",
            ),
            node(
                common.sort_data_frame_columns(category_cols=category_cols),
                "data_a",
                "sorted_data",
            ),
            node(common.finalize_data, "sorted_data", final_data_set),
        ],
    )
