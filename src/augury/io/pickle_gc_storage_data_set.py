"""
``PickleGCStorageDataSet`` loads and saves data to a file in Google Cloud Storage.

Current assumption is that this only runs in the context of a Google Cloud Function,
meaning that credentials are unnecessary (use local data files when running in
dev environment).
"""

from typing import Any, Dict, List
import os

import joblib
from kedro.io.core import AbstractDataSet
from google.cloud import storage
from google.api_core import timeout

from augury.settings import BASE_DIR


# The bigger model files take a really long time to upload and run against
# the default timeout limit
UPLOAD_TIMEOUT = 60


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
        self.filepath = filepath
        self.bucket_name = bucket_name
        self.project_dir = project_dir

    def _load(self) -> List[Dict[str, Any]]:
        client = storage.Client()
        bucket = client.get_bucket(self.bucket_name)
        blob = bucket.get_blob(self.filepath)

        local_file_dir = os.path.join(self.project_dir, "tmp")

        if not os.path.exists(local_file_dir):
            os.mkdir(local_file_dir)

        local_filepath = os.path.join(local_file_dir, self.filepath)

        with open(local_filepath, "wb") as file_obj:
            blob.download_to_file(file_obj)

        return joblib.load(local_filepath)

    @timeout.ConstantTimeout(timeout=UPLOAD_TIMEOUT)
    def _save(self, data: Any) -> None:
        """Save a Python object as a pickle file in Google Cloud Storage.

        Params
        ------
        data: Any Python object that can be pickled.
        """
        client = storage.Client()
        bucket = client.get_bucket(self.bucket_name)

        local_file_dir = os.path.join(self.project_dir, "tmp")

        if not os.path.exists(local_file_dir):
            os.mkdir(local_file_dir)

        local_filepath = os.path.join(local_file_dir, self.filepath)
        joblib.dump(data, local_filepath)

        blob = bucket.blob(self.filepath)

        # TODO: As far as I can tell, we can't make the timeout longer for this API call,
        # and for now, it's just tipresias_2020 that takes too long,
        # so I'm just rescuing with a warning until I can figure out a better solution.
        try:
            blob.upload_from_filename(local_filepath)
        except requests.exceptions.ConnectionError:
            logging.warning(
                "Couldn't upload %s due to the connection timing out. "
                "Upload it manually.",
                self.filepath,
            )

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self.filepath,
            bucket_name=self.bucket_name,
            project_dir=self.project_dir,
        )
