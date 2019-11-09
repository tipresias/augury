"""Application entry point."""

from pathlib import Path
from typing import Iterable, Dict, Optional
from datetime import date
import os

from kedro.context import KedroContext, load_context
from kedro.runner import AbstractRunner
from kedro.pipeline import Pipeline
from kedro.io import DataCatalog

from machine_learning.pipelines import create_pipelines, create_full_pipeline
from machine_learning.io import JSONRemoteDataSet


class ProjectContext(KedroContext):
    """Users can override the remaining methods from the parent class here, or create new ones
    (e.g. as required by plugins)

    """

    project_name = "augury"
    project_version = "0.15.2"

    def __init__(
        self,
        project_path,
        env=os.getenv("PYTHON_ENV"),
        round_number: Optional[int] = None,
        start_date: str = "1897-01-01",
        end_date: str = f"{date.today().year}-12-31",
    ):
        super().__init__(project_path, env=env)
        self.round_number = round_number
        self.start_date = start_date
        self.end_date = end_date

    @property
    def pipeline(self):
        return create_full_pipeline(self.start_date, self.end_date)

    def _get_pipelines(self) -> Dict[str, Pipeline]:
        return create_pipelines(self.start_date, self.end_date)

    def _get_catalog(
        self, save_version=None, journal=None, load_versions=None
    ) -> DataCatalog:
        catalog = super()._get_catalog(
            save_version=save_version, journal=journal, load_versions=load_versions
        )
        catalog.add(
            "roster_data",
            JSONRemoteDataSet(
                data_source="machine_learning.data_import.player_data.fetch_roster_data",
                date_range_type="round_number",
                load_kwargs={"round_number": self.round_number},
            ),
        )

        return catalog


def main(
    tags: Iterable[str] = None,
    env: str = None,
    runner: AbstractRunner = None,
    node_names: Iterable[str] = None,
    from_nodes: Iterable[str] = None,
    to_nodes: Iterable[str] = None,
    from_inputs: Iterable[str] = None,
    round_number: Optional[int] = None,
    start_date: str = "1897-01-01",
    end_date: str = f"{date.today().year}-12-31",
):
    """Application main entry point.

    Args:
        tags: An optional list of node tags which should be used to
            filter the nodes of the ``Pipeline``. If specified, only the nodes
            containing *any* of these tags will be run.
        env: An optional parameter specifying the environment in which
            the ``Pipeline`` should be run.
        runner: An optional parameter specifying the runner that you want to run
            the pipeline with.
        node_names: An optional list of node names which should be used to filter
            the nodes of the ``Pipeline``. If specified, only the nodes with these
            names will be run.
        from_nodes: An optional list of node names which should be used as a
            starting point of the new ``Pipeline``.
        to_nodes: An optional list of node names which should be used as an
            end point of the new ``Pipeline``.
        from_inputs: An optional list of input datasets which should be used as a
            starting point of the new ``Pipeline``.

    """
    project_context = load_context(
        Path.cwd(),
        env=env,
        round_number=round_number,
        start_date=start_date,
        end_date=end_date,
    )
    project_context.run(
        tags=tags,
        runner=runner,
        node_names=node_names,
        from_nodes=from_nodes,
        to_nodes=to_nodes,
        from_inputs=from_inputs,
    )


if __name__ == "__main__":
    main()
