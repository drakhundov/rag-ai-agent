import chainlit as cl
import anyio

from app_composition import build_app
from services.retriever_service import RetrieverService

svc: RetrieverService = build_app()


@cl.on_message
async def on_message(msg: str):
    text = msg if isinstance(msg, str) else msg.content
    answer = await anyio.to_thread.run_sync(svc.answer, text)
    await cl.Message(answer).send()
