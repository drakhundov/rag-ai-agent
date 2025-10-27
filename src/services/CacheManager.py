"""
A centralized unit for caching data used across various components.
Might save time and resources by avoiding redundant computations.

Structure:
    <cache_dir>/
        documents/
            <cache_key>/  # document hash
                splits/
                    <split_hash>.pkl
"""

import os
from typing import Text, Optional, Dict, Any, Union
import pickle

from core.config import load_conf
from core.types import CacheAttr

with load_conf() as conf:
    CACHE_DIR = conf.paths.cache_dir
    os.makedirs(CACHE_DIR, exist_ok=True)

# All the components that might rely on this:
# - ChromaDocumentRetriever
# - QueryTranslators
# - TextSplitters


class CacheManager:
    def __init__(self, dir_path: str):
        self._dir = os.path.join(CACHE_DIR, dir_path)
        os.makedirs(self._dir, exist_ok=True)

    def get(self, cache_id: str, attr: CacheAttr, read_as_binary: bool = False) -> Optional[Union[Text, bytes]]:
        path = self._path(cache_id)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        path = os.path.join(path, attr.value + ('.pkl' if read_as_binary else '.txt'))
        with open(path, 'rb' if read_as_binary else 'r') as f:
            if read_as_binary:
                return pickle.load(f)
            else:
                return f.read()

    def set(self, cache_id: str, data: Dict[CacheAttr, Any], write_as_binary: bool = False):
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
        path = os.path.join(self._path(cache_id), cache_t.value + ('.pkl' if write_as_binary else '.txt'))
        with open(path, 'wb' if write_as_binary else 'w') as f:
            if write_as_binary:
                pickle.dump(cache_val, f)
            else:
                f.write(str(cache_val))

    def _path(self, cache_id: str) -> str:
        return os.path.join(self._dir, cache_id)
