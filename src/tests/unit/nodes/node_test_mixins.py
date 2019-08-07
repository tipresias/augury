"""Class mixins for shared functionality among TestCases"""


class ColumnAssertionMixin:
    @staticmethod
    def _assert_column_added(
        test_case,
        column_names=[],
        valid_data_frame=None,
        feature_function=None,
        col_diff=1,
    ):

        for column_name in column_names:
            with test_case.subTest(data_frame=valid_data_frame):
                data_frame = valid_data_frame
                transformed_data_frame = feature_function(data_frame)

                test_case.assertEqual(
                    len(data_frame.columns) + col_diff,
                    len(transformed_data_frame.columns),
                )
                test_case.assertIn(column_name, transformed_data_frame.columns)

    @staticmethod
    def _assert_required_columns(
        test_case, req_cols=[], valid_data_frame=None, feature_function=None
    ):
        for req_col in req_cols:
            with test_case.subTest(data_frame=valid_data_frame.drop(req_col, axis=1)):
                data_frame = valid_data_frame.drop(req_col, axis=1)
                with test_case.assertRaises(AssertionError):
                    feature_function(data_frame)

    def _make_column_assertions(
        self,
        test_case,
        column_names=[],
        req_cols=[],
        valid_data_frame=None,
        feature_function=None,
        col_diff=1,
    ):
        self._assert_column_added(
            test_case,
            column_names=column_names,
            valid_data_frame=valid_data_frame,
            feature_function=feature_function,
            col_diff=col_diff,
        )

        self._assert_required_columns(
            test_case,
            req_cols=req_cols,
            valid_data_frame=valid_data_frame,
            feature_function=feature_function,
        )
