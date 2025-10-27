import logging
import sys

import anyio

from app_composition import build_rag_engine, setup_langsmith, init_logs
from core.types import QueryStr
from services.RAGEngine import RAGEngine
from utilities import string, cli

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
    # Takes file paths as positional argument.
    files = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    rag_svc = build_rag_engine(files)
    if rag_svc is None:
        raise RuntimeError("RAG Engine has not been initialized.")
    args = cli.parse_args()
    if args.cl:
        run_web_mode(rag_svc)
    else:
        print("Running in terminal mode. Press Ctrl+C to exit.")
        run_terminal_mode(rag_svc)
