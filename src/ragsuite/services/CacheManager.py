"""
A centralized unit for caching data used across various components.
Might save time and resources by avoiding redundant computations.

Structure:
    <cache_dir>/
        documents/
            <cache_key>/  # document hash
                splits/
                    <split_hash>.json
"""

import json
import os
from pathlib import Path
from typing import Text, Optional, Dict, Any, Union

from ragsuite.core.config import load_conf
from ragsuite.core.types import CacheAttr


# All the components that might rely on this:
# - ChromaDocumentRetriever
# - QueryTranslators
# - TextSplitters


class CacheManager:
    def __init__(self, dir_path: str):
        self._dir = os.path.join(self._get_cache_base_dir(), dir_path)
        os.makedirs(self._dir, exist_ok=True)

    def get(self, cache_id: str, attr: CacheAttr) -> Optional[Union[Text, Any]]:
        path = self._path(cache_id)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        path = os.path.join(path, attr.value + '.json')
        with open(path, 'r') as f:
            return json.load(f)

    def set(self, cache_id: str, data: Dict[CacheAttr, Any]):
        """Set a cache entry.
        data: Dictionary with one key-value pair. key must be from CacheAttr (SPLITTER, EMBEDDINGS).
              Value could be any serializable object.
        """
        if len(data) != 1:
            raise ValueError(f"CacheManager.set must be provided with a dictionary of one item, got {len(data)} items.")
        cache_t = list(data.keys())[0]
        if not isinstance(cache_t, CacheAttr):
            raise ValueError("CacheManager.set key must be an instance of CacheAttr Enum.")
        os.makedirs(os.path.join(self._path(cache_id)), exist_ok=True)
        cache_val = data[cache_t]
        path = os.path.join(self._path(cache_id), cache_t.value + '.json')
        with open(path, 'w') as f:
            json.dump(cache_val, f)

    def _path(self, cache_id: str) -> str:
        return os.path.join(self._dir, cache_id)

    @staticmethod
    def _get_cache_base_dir() -> Path:
        """Helper to safely read configuration path."""
        with load_conf() as conf:
            cache_base_dir = Path(conf.paths.cache_dir)
        cache_base_dir.mkdir(exist_ok=True)
        return cache_base_dir
