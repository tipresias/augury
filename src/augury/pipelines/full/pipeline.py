"""Pipelines for loading and processing joined data from other pipelines."""

from kedro.pipeline import Pipeline, node

from augury.settings import CATEGORY_COLS
from ..nodes import common
from .. import player, match

PIPELINE_CATEGORY_COLS = CATEGORY_COLS + [
    "prev_match_oppo_team",
    "oppo_prev_match_oppo_team",
]


def create_pipeline(
    start_date: str,
    end_date: str,
):
    """Create a pipeline that joins all data-source-specific pipelines."""
    return Pipeline(
        [
            match.create_pipeline(start_date, end_date),
            player.create_pipeline(start_date, end_date),
            node(
                common.clean_full_data,
                ["final_match_data", "final_player_data"],
                [
                    "clean_match_data_df",
                    "clean_player_data_df",
                ],
            ),
            node(
                common.combine_data(axis=1),
                [
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
                common.sort_data_frame_columns(category_cols=PIPELINE_CATEGORY_COLS),
                "filtered_data",
                "sorted_data",
            ),
            node(common.finalize_data, "sorted_data", "full_data"),
        ],
    )
