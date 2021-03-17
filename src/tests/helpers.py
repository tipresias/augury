"""Functions and classes to deduplicate and simplify test code."""


class ColumnAssertionMixin:
    """Mixin class for making columns assertions in tests for Kedro nodes."""

    def _assert_column_added(
        self, column_names=[], valid_data_frame=None, feature_function=None, col_diff=1
    ):

        for column_name in column_names:
            with self.subTest(data_frame=valid_data_frame):
                data_frame = valid_data_frame
                transformed_data_frame = feature_function(data_frame)

                self.assertEqual(
                    len(data_frame.columns) + col_diff,
                    len(transformed_data_frame.columns),
                )
                self.assertIn(column_name, transformed_data_frame.columns)

    def _assert_required_columns(
        self, req_cols=[], valid_data_frame=None, feature_function=None
    ):
        for req_col in req_cols:
            with self.subTest(data_frame=valid_data_frame.drop(req_col, axis=1)):
                data_frame = valid_data_frame.drop(req_col, axis=1)
                with self.assertRaises(AssertionError):
                    feature_function(data_frame)

    def _make_column_assertions(
        self,
        column_names=[],
        req_cols=[],
        valid_data_frame=None,
        feature_function=None,
        col_diff=1,
    ):
        self._assert_column_added(
            column_names=column_names,
            valid_data_frame=valid_data_frame,
            feature_function=feature_function,
            col_diff=col_diff,
        )

        self._assert_required_columns(
            req_cols=req_cols,
            valid_data_frame=valid_data_frame,
            feature_function=feature_function,
        )
