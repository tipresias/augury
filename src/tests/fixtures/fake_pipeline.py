from kedro.pipeline import Pipeline, node

from machine_learning.nodes import common, match


def create_fake_pipeline(*_args, **_kwargs):
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
