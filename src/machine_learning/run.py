"""Application entry point."""

import logging.config
from pathlib import Path
from typing import Iterable
from warnings import warn
import os
from datetime import date

from kedro.cli.utils import KedroCliError
from kedro.config import ConfigLoader, MissingConfigException
from kedro.io import DataCatalog
from kedro.runner import SequentialRunner
from kedro.utils import load_obj

import pandas as pd

from machine_learning.pipeline import (
    betting_pipeline,
    match_pipeline,
    fake_estimator_pipeline,
)
from machine_learning.settings import BASE_DIR


# Name of root directory containing project configuration.
CONF_ROOT = "conf"

# Default configuration environment to be used for running the pipeline.
# Change this constant value if you want to load configuration
# from a different location.
DEFAULT_RUN_ENV = "development"


def __kedro_context__():
    """Provide this project's context to ``kedro`` CLI and plugins.
    Please do not rename or remove, as this will break the CLI tool.

    Plugins may request additional objects from this method.
    """
    return {
        "get_config": get_config,
        "create_catalog": create_catalog,
        "create_pipeline": betting_pipeline,
        "template_version": "0.14.3",
        "project_name": "Augury",
        "project_path": BASE_DIR,
    }


def get_config(project_path: str, env: str = None, **_kwargs) -> ConfigLoader:
    """Loads Kedro's configuration at the root of the project.

    Args:
        project_path: The root directory of the Kedro project.
        env: The environment used for loading configuration.
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        ConfigLoader which can be queried to access the project config.

    """
    project_path_obj = Path(project_path)
    env = env or DEFAULT_RUN_ENV
    conf_paths = [
        str(project_path_obj / CONF_ROOT / "base"),
        str(project_path_obj / CONF_ROOT / env),
    ]
    return ConfigLoader(conf_paths)


def create_catalog(config: ConfigLoader, **_kwargs) -> DataCatalog:
    """Loads Kedro's ``DataCatalog``.

    Args:
        config: ConfigLoader which can be queried to access the project config.
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        DataCatalog defined in `catalog.yml`.

    """
    conf_logging = config.get("logging*", "logging*/**")
    logging.config.dictConfig(conf_logging)
    conf_catalog = config.get("catalog*", "catalog*/**")

    try:
        conf_creds = config.get("credentials*", "credentials*/**")
    except MissingConfigException:
        warn("Your Kedro project is missing a credentials file!")
        conf_creds = {}

    conf_params = config.get("parameters*", "parameters*/**")
    logging.config.dictConfig(conf_logging)
    catalog = DataCatalog.from_config(conf_catalog, conf_creds)
    catalog.add_feed_dict({"parameters": conf_params})
    return catalog


def run_betting_pipeline(
    start_date: str, end_date: str, runner: str = None
) -> pd.DataFrame:
    # Load Catalog
    conf = get_config(project_path=BASE_DIR, env=os.getenv("PYTHON_ENV"))
    catalog = create_catalog(config=conf)

    # Load the runner
    # When either --parallel or --runner is used, class_obj is assigned to runner
    runner_func = load_obj(runner, "kedro.runner") if runner else SequentialRunner

    # Run the runner
    return runner_func().run(betting_pipeline(start_date, end_date), catalog)


def run_match_pipeline(
    start_date: str, end_date: str, runner: str = None
) -> pd.DataFrame:
    # Load Catalog
    conf = get_config(project_path=str(Path.cwd()), env=os.getenv("PYTHON_ENV"))
    catalog = create_catalog(config=conf)

    # Load the runner
    # When either --parallel or --runner is used, class_obj is assigned to runner
    runner_func = load_obj(runner, "kedro.runner") if runner else SequentialRunner

    # Run the runner
    return runner_func().run(match_pipeline(start_date, end_date), catalog)


def run_fake_estimator_pipeline(runner: str = None) -> pd.DataFrame:
    # Load Catalog
    conf = get_config(project_path=str(Path.cwd()), env=None)
    catalog = create_catalog(config=conf)

    # Load the runner
    # When either --parallel or --runner is used, class_obj is assigned to runner
    runner_func = load_obj(runner, "kedro.runner") if runner else SequentialRunner

    # Run the runner
    return runner_func().run(fake_estimator_pipeline(), catalog)


def main(tags: Iterable[str] = None, env: str = None, runner: str = None):
    """Application main entry point.

    Args:
        tags: An optional list of node tags which should be used to
            filter the nodes of the ``Pipeline``. If specified, only the nodes
            containing *any* of these tags will be added to the ``Pipeline``.
        env: An optional parameter specifying the environment in which
            the ``Pipeline`` should be run. If not specified defaults to "development".
        runner: An optional parameter specifying the runner that you want to run
            the pipeline with.

    Raises:
        KedroCliError: If the resulting ``Pipeline`` is empty.

    """
    # Report project name
    logging.info("** Kedro project {}".format(BASE_DIR))

    # Load Catalog
    conf = get_config(project_path=BASE_DIR, env=env)
    catalog = create_catalog(config=conf)

    # Load the pipeline
    pipeline = betting_pipeline("2010-01-01", str(date.today()))
    pipeline = pipeline.only_nodes_with_tags(*tags) if tags else pipeline
    if not pipeline.nodes:
        if tags:
            raise KedroCliError("Pipeline contains no nodes with tags: " + str(tags))
        raise KedroCliError("Pipeline contains no nodes")

    # Load the runner
    # When either --parallel or --runner is used, class_obj is assigned to runner
    runner_func = load_obj(runner, "kedro.runner") if runner else SequentialRunner

    # Run the runner
    runner_func().run(pipeline, catalog)


if __name__ == "__main__":
    main()
