import os
import logging
import subprocess

from ragsuite.app_composition import build_rag_engine, setup_langsmith, init_logs
from ragsuite.core.types import QueryStr
from ragsuite.services.RAGEngine import RAGEngine
from ragsuite.utilities import cli
from ragsuite.utilities import string
from ragsuite.core.config import load_conf

setup_langsmith()

rag_svc: RAGEngine
logger: logging.Logger


def run_web_mode():
    env = os.environ.copy()
    try:
        with load_conf() as conf:
            subprocess.run(
                ["chainlit", "run", os.path.join(conf.paths.proj_dir, "src/ragsuite/app_cl.py")],
                env=env
            )
    except Exception as e:
        logger.debug("Exited with failure: ", e.with_traceback)
        raise e


def run_terminal_mode(rag_svc: RAGEngine):
    logger.info("Running terminal mode")
    try:
        while True:
            user_input: QueryStr = QueryStr(input(">> "))
            if not user_input:
                continue
            response = rag_svc.generate_answer(user_input)
            print(f"\033[96m{string.format_response(response)}\033[0m")
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    logger = init_logs()
    logger.debug("Logging is configured")
    logger.debug("Starting RAG Assistant Application")
    args = cli.parse_args()
    if args.cl:
        os.environ["RAG_FILES"] = os.pathsep.join(args.files)
        run_web_mode()
    else:
        print("Running in terminal mode. Press Ctrl+C to exit.")
        rag_svc = build_rag_engine(args.files)
        if rag_svc is None:
            raise RuntimeError("RAG Engine has not been initialized.")
        run_terminal_mode(rag_svc)
