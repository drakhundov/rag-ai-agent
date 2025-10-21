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


def with_temp_message(message: str):
    """Decorator to show a temporary message while executing a function."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"\r{message}", end="", flush=True)
            result = func(*args, **kwargs)
            print("\r\033[K", end="", flush=True)
            return result
        return wrapper
    return decorator
