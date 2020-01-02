"""
``PickleGCStorageDataSet`` loads and saves data to a file in Google Cloud Storage.

Current assumption is that this only runs in the context of a Google Cloud Function,
meaning that credentials are unnecessary (use local data files when running in
dev environment).
"""

from typing import Any, Dict, List
import os

import pandas as pd
import joblib
from kedro.io.core import AbstractDataSet
from google.cloud import storage

from augury.settings import BASE_DIR


class PickleGCStorageDataSet(AbstractDataSet):
    """Loads and saves pickled objects to a file in Google Cloud Storage."""

    def __init__(
        self, filepath: str, bucket_name: str, project_dir: str = BASE_DIR
    ) -> None:
        """Instantiate a PickleGCStorageDataSet.

        Params
        ------
        filepath: Path to a pickle file.
        bucket_name: GC Storage bucket name.
        project_dir: Root directory for the project.
        """
        self._filepath = filepath
        self._bucket_name = bucket_name
        self._project_dir = project_dir

    def _load(self) -> List[Dict[str, Any]]:
        client = storage.Client()

        bucket = client.get_bucket(self._bucket_name)
        blob = bucket.get_blob(self._filepath)

        local_file_dir = os.path.join(self._project_dir, "tmp")

        if not os.path.exists(local_file_dir):
            os.mkdir(local_file_dir)

        local_filepath = os.path.join(local_file_dir, self._filepath)

        with open(local_filepath, "wb") as file_obj:
            blob.download_to_file(file_obj)

        return joblib.load(local_filepath)

    def _save(self, data: pd.DataFrame) -> None:
        pass

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            bucket_name=self._bucket_name,
            project_dir=self._project_dir,
        )
