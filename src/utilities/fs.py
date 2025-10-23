import logging
import os
from typing import Dict

logger: logging.Logger = logging.getLogger()

def save_session(session_data: Dict, path: str, session_id: str, filemode="w"):
    session_file = os.path.join(
        path,
        f"{session_id}"
    )
    with open(session_file, filemode) as f:
        for k, v in session_data.items():
            if isinstance(v, dict):
                save_session(v, k, filemode="a")
            elif isinstance(v, list):
                f.write(f"{k}:\n")
                for i, item in enumerate(v):
                    f.write(f"\t{i}: {item}\n")
            else:
                f.write(f"{k}: {v}\n")
    logger.debug(f"Session saved to {session_file}")
