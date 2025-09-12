from typing import Optional

import anyio
import chainlit as cl

from app_composition import build_rag_engine, setup_langsmith
from services.rag_engine import RAGEngine

rag_svc: Optional[RAGEngine] = None

@cl.on_chat_start
def start():
    global rag_svc
    # ! Might want to reconsider for production environment.
    setup_langsmith()
    rag_svc = build_rag_engine()
    cl.user_session.set("rag_engine", rag_svc)
    cl.user_session.set("chat_history", [])


@cl.on_message
async def on_message(msg: str | cl.Message):
    global rag_svc
    text = msg if isinstance(msg, str) else msg.content
    answer = await anyio.to_thread.run_sync(rag_svc.generate_answer, text)
    await cl.Message(answer).send()
