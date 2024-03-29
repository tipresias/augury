# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Project hooks."""
from typing import Any, Dict, Iterable, Optional

from kedro.config import ConfigLoader
from kedro.framework.hooks import hook_impl
from kedro.framework.session import get_current_session
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from kedro.versioning import Journal


class ProjectHooks:
    """Lifecycle hooks for running Kedro pipelines."""

    @hook_impl
    def register_pipelines(self) -> Dict[str, Pipeline]:
        """Register the project's pipeline.

        Returns:
            A mapping from a pipeline name to a ``Pipeline`` object.
        """
        # We need to import inside the method, because Kedro's bogarting
        # of the 'settings' module creates circular dependencies, so we either have to
        # do this or create a separate settings module to import around the app.
        from augury.pipelines import (  # pylint: disable=import-outside-toplevel
            betting,
            full,
            match,
            player,
        )

        session = get_current_session()
        assert session is not None

        context = session.load_context()
        start_date: str = context.start_date  # type: ignore
        end_date: str = context.end_date  # type: ignore

        return {
            "__default__": Pipeline([]),
            "betting": betting.create_pipeline(start_date, end_date),
            "match": match.create_pipeline(start_date, end_date),
            "player": player.create_pipeline(
                start_date,
                end_date,
                past_match_pipeline=match.create_past_match_pipeline(),
            ),
            "full": full.create_pipeline(start_date, end_date),
        }

    @hook_impl
    def register_config_loader(self, conf_paths: Iterable[str]) -> ConfigLoader:
        """Register the project's config loader."""
        return ConfigLoader(conf_paths)

    @hook_impl
    def register_catalog(
        self,
        catalog: Optional[Dict[str, Dict[str, Any]]],
        credentials: Dict[str, Dict[str, Any]],
        load_versions: Dict[str, str],
        save_version: str,
        journal: Journal,
    ) -> DataCatalog:
        """Register the project's data catalog."""
        # We need to import inside the method, because Kedro's bogarting
        # of the 'settings' module creates circular dependencies, so we either have to
        # do this or create a separate settings module to import around the app.
        from augury.io import (  # pylint: disable=import-outside-toplevel
            JSONRemoteDataSet,
        )

        data_catalog = DataCatalog.from_config(
            catalog, credentials, load_versions, save_version, journal
        )
        session = get_current_session()
        assert session is not None

        context = session.load_context()
        round_number: Optional[int] = context.round_number  # type: ignore

        data_catalog.add(
            "roster_data",
            JSONRemoteDataSet(
                data_source="augury.data_import.player_data.fetch_roster_data",
                round_number=round_number,
            ),
        )

        return data_catalog


project_hooks = ProjectHooks()
