import chainlit as cl
import anyio

from app_composition import build_app
from services.rag_engine import RAGEngine

svc: RAGEngine = build_app()

@cl.on_message
async def on_message(msg: str | cl.Message):
    text = msg if isinstance(msg, str) else msg.content
    answer = await anyio.to_thread.run_sync(svc.generate_answer, text)
    await cl.Message(answer).send()
