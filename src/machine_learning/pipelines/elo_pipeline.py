"""Pipeline for match data for Elo model"""

from kedro.pipeline import Pipeline, node

from machine_learning.nodes import common, match


def create_elo_pipeline(match_pipeline: Pipeline = Pipeline([]), **_kwargs):
    """
    Kedro pipeline for loading and transforming match data with Elo features included.
    Only relevant for generating predictions from older models.

    Args:
        start_date (str, YYYY-MM-DD format): Earliest date for included data.
        end_date (str, YYYY-MM-DD format): Latest date for included data.
        match_pipeline (kedro.pipeline.Pipeline): Pipeline for match data. Only needed
            if the Elo pipeline is run in isolation.
    """

    return Pipeline(
        [
            match_pipeline,
            node(
                match.add_shifted_team_features(shift_columns=["oppo_team", "at_home"]),
                "shifted_match_data",
                "shifted_elo_data",
            ),
            # add_elo_rating depends on DF still being organized per-match
            # with home_team/away_team columns
            node(
                match.convert_teammatch_rows_to_match_rows,
                "shifted_elo_data",
                "elo_match_row_data",
            ),
            node(common.convert_to_json, "elo_match_row_data", "elo_data"),
        ]
    )
