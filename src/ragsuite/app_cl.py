import logging
import os

import anyio

from ragsuite.app_composition import build_rag_engine
from ragsuite.services import RAGEngine

rag_svc: RAGEngine
logger: logging.Logger = logging.getLogger(__name__)

logger.info("Running web mode")
try:
    import chainlit as cl
except ImportError:
    raise ImportError(
        "chainlit is not installed. Please install it to use ChainlitWebAssistant."
    )


@cl.on_chat_start
def start():
    global rag_svc
    # ! Might want to reconsider for production environment.
    files = os.environ["RAG_FILES"].split(":")
    rag_svc = build_rag_engine(files)
    cl.user_session.set("rag_engine", rag_svc)
    cl.user_session.set("chat_history", [])


@cl.on_message
async def on_message(msg: str | cl.Message):
    global rag_svc
    if rag_svc is None:
        await cl.Message(
            "Error: RAG engine not initialized. Please start a new chat session."
        ).send()
        return
    text = msg if isinstance(msg, str) else msg.content
    response = await anyio.to_thread.run_sync(rag_svc.generate_answer, text)
    await cl.Message(response).send()
