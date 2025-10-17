import argparse
import logging
import sys

import anyio

from app_composition import build_rag_engine, setup_langsmith
from services.RAGEngine import RAGEngine

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
        setup_langsmith()
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
        answer = await anyio.to_thread.run_sync(rag_svc.generate_answer, text)
        await cl.Message(answer).send()

    return


def run_terminal_mode(rag_svc: RAGEngine):
    logger.info("Running terminal mode")
    try:
        while True:
            user_input = input("[User] >> ")
            if not user_input:
                continue
            answer = rag_svc.generate_answer(user_input)
            print(f"Assistant: {answer}")
    except KeyboardInterrupt:
        print("\nExiting...")


def main():
    parser = argparse.ArgumentParser(
        description="Run the RAG Assistant in different modes."
    )
    parser.add_argument(
        "--cl", action="store_true", help="Run the assistant in Chainlit web mode."
    )
    parser.add_argument(
        "--shell", action="store_true", help="Run the assistant in shell mode."
    )
    flags = [arg for arg in sys.argv[1:] if arg.startswith("-")]
    args = parser.parse_args(flags)

    if args.cl:
        run_web_mode(rag_svc)
    else:
        print("Running in terminal mode. Press Ctrl+C to exit.")
        run_terminal_mode(rag_svc)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(".log", mode="a", encoding="utf-8"),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging is configured")
    logger.info("Starting RAG Assistant Application")
    # Takes file paths as positional argument.
    files = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    rag_svc = build_rag_engine(files)
    main()
