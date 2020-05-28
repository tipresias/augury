"""
``JSONGCStorageDataSet`` loads and saves data to a file in Google Cloud Storage.

Current assumption is that this only runs in the context of a Google Cloud Function,
meaning that credentials are unnecessary (use local data files when running in
dev environment).
"""

from typing import Any, Dict, List
import json

from kedro.io.core import AbstractDataSet
from google.cloud import storage


class JSONGCStorageDataSet(AbstractDataSet):
    """
    ``JSONGCStorageDataSet`` loads and saves data to a file in Google Cloud Storage.

    Current assumption is that this only runs in the context of a Google Cloud Function,
    meaning that credentials are unnecessary (use local data files when running in
    dev environment).
    """

    def __init__(self, filepath: str, bucket_name: str) -> None:
        """Instantiate a JSONGCStorageDataSet object.

        Params
        ------
        filepath: Path to a json file.
        bucket_name: GC Storage bucket name.
        """
        self._filepath = filepath
        self._bucket_name = bucket_name

    def _load(self) -> List[Dict[str, Any]]:
        client = storage.Client()

        bucket = client.get_bucket(self._bucket_name)
        blob = bucket.get_blob(self._filepath)

        return json.loads(blob.download_as_string())

    def _save(self, data: List[Dict[str, Any]]) -> None:
        client = storage.Client()

        bucket = client.get_bucket(self._bucket_name)
        blob = storage.Blob(self._filepath, bucket)

        blob.upload_from_string(
            json.dumps(data, indent=2), content_type="application/json"
        )

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath, bucket_name=self._bucket_name)

    def _exists(self) -> bool:
        client = storage.Client()
        bucket = client.get_bucket(self._bucket_name)

        return storage.Blob(bucket=bucket, name=self._filepath).exists(client)
