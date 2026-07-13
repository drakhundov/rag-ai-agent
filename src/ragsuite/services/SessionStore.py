"""
Sessions store decisions made by pipeline services in JSON format. In a way, it's extensive logging.
! They are stored strictly in the sessions directory defined in the configuration files.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from ragsuite.core.config import load_conf

logger: logging.Logger = logging.getLogger(__name__)


class SessionStore:
    def __init__(self, svc_id: str):
        """
        Initialized by each service individually.

        * A service is a single component of the RAG pipeline (e.g., retriever, re-ranker).
        * A session is a single run for a user request.
        """
        self.svc_id = svc_id
        self.session_dir = Path(self._get_sessions_base_dir()) / self.svc_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_sessions_base_dir() -> Path:
        """Helper to safely read configuration path."""
        with load_conf() as conf:
            sessions_base_dir = Path(conf.paths.sessions_dir)
        sessions_base_dir.mkdir(exist_ok=True)

    def dump(self, session_data: Dict[str, Any], session_id: str | None = None) -> None:
        """
        Dumps the session data to a JSON file in the service's session directory.

        If session_id is not provided, it generates a unique ID using the current
        timestamp down to the microsecond to prevent collision overwrites.
        """
        if session_id is None:
            # Added microsecond (%f) to prevent concurrent request collisions
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Appending .json extension enforces standard structural formatting
        sess_fpath = self.session_dir / f"{session_id}.json"

        try:
            with open(sess_fpath, "w", encoding="utf-8") as handle:
                json.dump(session_data, handle, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(
                f"Failed to dump session {session_id} for service {self.svc_id}: {e}"
            )
            raise
