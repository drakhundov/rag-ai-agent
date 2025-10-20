import logging
import os
import sys
from datetime import datetime

import anyio

from app_composition import build_rag_engine, setup_langsmith
from core.config import load_conf
from services.RAGEngine import RAGEngine
from utilities.string import format_response
from utilities.cli import parse_args

setup_langsmith()

rag_svc: RAGEngine
logger: logging.Logger


def run_web_mode(rag_svc: RAGEngine):
    logger.info("Running web mode")
    try:
        import chainlit as cl
    except ImportError:
        raise ImportError(
            "chainlit is not installed. Please install it to use ChainlitWebAssistant."
        )

    @cl.on_chat_start
    def start():
        # ! Might want to reconsider for production environment.
        rag_svc = build_rag_engine()
        cl.user_session.set("rag_engine", rag_svc)
        cl.user_session.set("chat_history", [])

    @cl.on_message
    async def on_message(msg: str | cl.Message):
        if rag_svc is None:
            await cl.Message(
                "Error: RAG engine not initialized. Please start a new chat session."
            ).send()
            return
        text = msg if isinstance(msg, str) else msg.content
        response = await anyio.to_thread.run_sync(rag_svc.generate_answer, text)
        await cl.Message(response).send()

    return


def run_terminal_mode(rag_svc: RAGEngine):
    logger.info("Running terminal mode")
    try:
        while True:
            user_input = input(">> ")
            if not user_input:
                continue
            response = rag_svc.generate_answer(user_input)
            print(f"\033[96m{format_response(response)}\033[0m")
    except KeyboardInterrupt:
        print("\nExiting...")


def main():
    global rag_svc
    if rag_svc is None:
        raise RuntimeError("RAG Engine has not been initialized.")
    args = parse_args()
    if args.cl:
        run_web_mode(rag_svc)
    else:
        print("Running in terminal mode. Press Ctrl+C to exit.")
        run_terminal_mode(rag_svc)


if __name__ == "__main__":
    # Initialize logging.
    with load_conf() as conf:
        proj_dir = str(conf.paths.proj_dir)
    log_dir = os.path.join(proj_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(
                os.path.join(
                    log_dir, datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S.log")
                ),
                mode="a",
                encoding="utf-8",
            ),
        ],
        force=True,
    )
    logger = logging.getLogger()
    logger.debug("Logging is configured")
    logger.debug("Starting RAG Assistant Application")
    # Takes file paths as positional argument.
    files = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    rag_svc = build_rag_engine(files)
    main()
