"""
A centralized unit for caching data used across components.
Might save time and resources by avoiding redundant computation.
! This class has a composition relationship with JSONDiskStore.
! They are stored strictly in the cache directory defined in the configuration files.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Tuple, Any

from ragsuite.core.config import load_conf
from .JSONDiskStore import JSONDiskStore

logger: logging.Logger = logging.getLogger(__name__)


# All the components that might rely on this:
# - ChromaDocumentRetriever
# - QueryTranslators
# - TextSplitters

class CacheStore:
    """
    Used to store processed data for future use (e.g. document splits)
    !Critical points:
    — collisions
    """

    # Cache expires within Time-To-Live seconds.
    TTL_SECONDS = 88600

    def __init__(self, svc_id: str):
        """Initialized by each service individually."""
        self.local_disk_store = JSONDiskStore(
            self._get_base_cache_dir(),
            svc_id
        )
        self.svc_id = svc_id

    @staticmethod
    def _get_base_cache_dir() -> Path:
        """Helper to safely read configuration path."""
        with load_conf() as conf:
            cache_base_dir = Path(conf.paths.cache_dir)
        cache_base_dir.mkdir(exist_ok=True, parents=True)
        return cache_base_dir

    def set(self, cache_key: str, data: Dict[str, Any]) -> bool:
        try:
            self.local_disk_store.dump(data, cache_key)
            return True

        except FileExistsError as e:
            # ! For cache managing, collision is a serious error.
            logger.exception("Cache key collision detected: %s", e.filename)
        except PermissionError as e:
            logger.exception("Don't have permission for: %s %s", e.filename, e.strerror)
        except OSError as e:
            logger.exception("OS level failure writing to %s: %s", e.filename, e.strerror)
        except TypeError:
            logger.warning("Format mismatch when saving cache: %s", cache_key)
        except Exception as e:
            logger.exception("Unexpected error when saving cache: %s", cache_key)

        return False

    def get(self, cache_key: str) -> Tuple[bool, Dict[str, Any] | None]:
        try:
            filepath = self.local_disk_store.construct_path(cache_key)
            if os.path.exists(filepath):
                t = time.time() - os.path.getmtime(filepath)
                if t > CacheStore.TTL_SECONDS:
                    logger.debug("Cache cache_key=%s expired. File age: %s", cache_key, t)
                    os.remove(filepath)
                    return False, None

            loaded_cache = self.local_disk_store.load(cache_key)
            return True, loaded_cache

        except FileNotFoundError as e:
            logger.warning("Cache not found: %s", e.filename)
        except PermissionError as e:
            logger.exception("Don't have permission for: %s %s", e.filename, e.strerror)
        except OSError as e:
            logger.exception("OS level failure reading from %s: %s", e.filename, e.strerror)
        except json.JSONDecodeError:
            logger.warning("Cache file is corrupted or contains invalid JSON: %s", cache_key)
        except Exception as e:
            logger.exception("Unexpected error when loading cache: %s", cache_key)

        return False, None

    def retrieve_all_cache(self) -> Tuple[bool, Dict[str, Any] | None]:
        all_caches = {}
        for cache_file in os.listdir(os.path.join(self._get_base_cache_dir(), self.svc_id)):
            cache_key = os.path.splitext(cache_file)[0]
            is_success, cache_data = self.get(cache_key)
            if is_success:
                all_caches[cache_key] = cache_data
        return True, all_caches
