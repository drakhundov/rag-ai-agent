import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from core.config import load_conf
from core.types import QueryStr, ResponseStr

logger: logging.Logger = logging.getLogger()


# Interface: ports/ChatModel
class OpenAIChatModel(Runnable):
    def __init__(self, model_name: str = None, api_key: SecretStr = None):
        logger.debug("Starting OpenAIChatModel initialization")
        with load_conf() as conf:
            if model_name is None:
                model_name = conf.models.chat_model_name
            if api_key is None:
                api_key = conf.openai_api_key
            self._llm_model = ChatOpenAI(
                model=model_name, base_url=conf.paths.hf_router_url, api_key=api_key
            )
        logger.debug("OpenAIChatModel initialized")

    def generate(
        self, prompt_templ: PromptTemplate, query: QueryStr, context: List[Document]
    ) -> ResponseStr:
        logger.debug(f"Generating the answer for query: {query}")
        chain = prompt_templ | self._llm_model
        return ResponseStr(
            chain.invoke({"query": query, "context": context}).content
        )

    # ! Used to make sure compatibility with LangChain pipelines.
    def invoke(self, input, *args, **kwargs):
        return self._llm_model.invoke(input, *args, **kwargs)
