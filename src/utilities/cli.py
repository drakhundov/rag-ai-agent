import argparse
import sys
from functools import lru_cache
from typing import List


@lru_cache(maxsize=1)
def parse_args(sys_args: List[str] = None) -> argparse.Namespace:
    if sys_args is None:
        sys_args = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Run the RAG Assistant in different modes."
    )
    parser.add_argument(
        "--cl", action="store_true", help="Run the assistant in Chainlit web mode."
    )
    parser.add_argument(
        "--shell", action="store_true", help="Run the assistant in shell mode."
    )
    flags = [arg for arg in sys_args if arg.startswith("-")]
    return parser.parse_args(flags)
