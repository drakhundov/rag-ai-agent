"""
Basic memory manager to store JSON files.
SessionStore and CacheStore have a composition relationship with it.

— Dict --serialize--> JSON --stream--> filestore.json
— filestore.json --stream--> JSON --deserialize--> Dict

Structure:
    <base_dir>/
        <service_id>/           # e.g. semantic-text-splitter/ or
            <data_id>.json      # e.g. <doc_hash>.json ({"emb": [...], "splits": [...]})
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

logger: logging.Logger = logging.getLogger(__name__)


class JSONDiskStore:
    def __init__(self, base_dir: str | Path, svc_id: str):
        """
        Initialized by each service individually.

        * A service is a single component of the RAG pipeline (e.g., retriever, chunker).

        svc_id: determines subfolder name reserved for the service.
        """
        self.svc_id = svc_id
        self._dir = Path(base_dir) / self.svc_id
        self._dir.mkdir(parents=True, exist_ok=True)

    def construct_path(self, data_id: str) -> Path:
        return self._dir / f"{data_id}.json"

    def dump(self, data: Dict[str, Any], data_id: str | None = None):
        """
        Dumps the session data to a JSON file in the service's session directory.

        Args:
            data: the data to dump
            data_id: name of the JSON file to store data (if not provided, will be generated automatically)
        """
        if data_id is None:
            # * If data_id is not provided, it generates a unique ID using the current
            # * timestamp down to the microsecond to prevent collision overwrites.
            data_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        data_fpath = self.construct_path(data_id)
        with open(data_fpath, "x", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=4)

    def load(self, data_id: str) -> Dict[str, Any]:
        """
        Loads data from a JSON file in the service's dedicated directory.

        Args:
            data_id: name of the JSON file to load
        Returns:
            Data in the Dict format
        """
        data_fpath = self.construct_path(data_id)
        with open(data_fpath, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data
