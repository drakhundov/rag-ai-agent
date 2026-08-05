"""
Sessions store decisions made by pipeline services in JSON format. In a way, it's extensive logging.
! This class has a composition relationship with JSONDiskStore.
! They are stored strictly in the sessions directory defined in the configuration files.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

from ragsuite.core.config import load_conf
from .JSONDiskStore import JSONDiskStore

logger: logging.Logger = logging.getLogger(__name__)


class SessionStore:
    def __init__(self, svc_id: str):
        """
        Initialized by each service individually.

        * A service is a single component of the RAG pipeline (e.g., retriever, re-ranker).
        * A session is a single run for a user request.
        """
        self.local_disk_store: JSONDiskStore = JSONDiskStore(
            self._get_sessions_base_dir(),
            svc_id
        )

    @staticmethod
    def _get_sessions_base_dir() -> Path:
        """Helper to safely read configuration path."""
        with load_conf() as conf:
            sessions_base_dir = Path(conf.paths.sessions_dir)
        sessions_base_dir.mkdir(exist_ok=True, parents=True)
        return sessions_base_dir

    def record(self, session_data: Dict[str, Any], session_id: str) -> bool:
        """
        Saves session data to disk.

        Args:
            session_data: session data to save in Dict format
            session_id: unique session id
        Returns:
            True if session was successfully saved, else False
        """
        try:
            self.local_disk_store.dump(session_data, session_id)
            return True

        except FileExistsError as e:
            logger.warning("File collision detected: %s", e.filename)
        except PermissionError as e:
            logger.exception("Don't have permission for: %s %s", e.filename, e.strerror)
        except OSError as e:
            logger.exception("OS level failure writing to %s: %s", e.filename, e.strerror)
        except TypeError:
            logger.warning("Format mismatch when recording session: %s", session_id)
        except Exception as e:
            logger.exception("Unexpected error when saving session: %s", session_id)

        return False

    def load(self, session_id: str) -> Tuple[bool, Dict[str, Any] | None]:
        """
        Loads session data from disk.

        Args:
            session_id: unique session id
        Returns:
            A tuple of success status and session data in Dict format
        """
        try:
            session_data = self.local_disk_store.load(session_id)
            return True, session_data

        except FileNotFoundError as e:
            logger.warning("File not found: %s", e.filename)
        except PermissionError as e:
            logger.exception("Don't have permission for: %s %s", e.filename, e.strerror)
        except OSError as e:
            logger.exception("OS level failure reading from %s: %s", e.filename, e.strerror)
        except json.JSONDecodeError:
            logger.warning("Session file is corrupted or contains invalid JSON: %s", session_id)
        except Exception as e:
            logger.exception("Unexpected error when loading session: %s", session_id)

        return False, None
