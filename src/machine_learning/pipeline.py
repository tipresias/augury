# Copyright 2018-2019 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
#     or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pipeline construction."""

from kedro.pipeline import Pipeline, node

from machine_learning.data_processors.feature_calculation import (
    feature_calculator,
    calculate_rolling_rate,
)
from machine_learning.data_processors import feature_functions
from .nodes.example import predict, report_accuracy, split_data, train_model
from .nodes import betting

# Here you can define your data-driven pipeline by importing your functions
# and adding them to the pipeline as follows:
#
# from nodes.data_wrangling import clean_data, compute_features
#
# pipeline = Pipeline([
#     node(clean_data, 'customers', 'prepared_customers'),
#     node(compute_features, 'prepared_customers', ['X_train', 'Y_train'])
# ])
#
# Once you have your pipeline defined, you can run it from the root of your
# project by calling:
#
# $ kedro run
#


def betting_pipeline(**_kwargs):
    return Pipeline(
        [
            node(betting.clean_data, "betting_data", "clean_betting_data"),
            node(
                betting.convert_match_rows_to_teammatch_rows,
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
                feature_functions.add_oppo_features(
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


def create_pipeline(**_kwargs):
    """Create the project's pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        Pipeline: The resulting pipeline.

    """

    ###########################################################################
    # Here you can find an example pipeline with 4 nodes.
    #
    # PLEASE DELETE THIS PIPELINE ONCE YOU START WORKING ON YOUR OWN PROJECT AS
    # WELL AS THE FILE nodes/example.py
    # -------------------------------------------------------------------------

    pipeline = Pipeline(
        [
            node(
                split_data,
                ["example_iris_data", "parameters"],
                dict(
                    train_x="example_train_x",
                    train_y="example_train_y",
                    test_x="example_test_x",
                    test_y="example_test_y",
                ),
            ),
            node(
                train_model,
                ["example_train_x", "example_train_y", "parameters"],
                "example_model",
            ),
            node(
                predict,
                dict(model="example_model", test_x="example_test_x"),
                "example_predictions",
            ),
            node(report_accuracy, ["example_predictions", "example_test_y"], None),
        ]
    )
    ###########################################################################

    return pipeline
