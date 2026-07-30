import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from ragsuite.core.config import load_conf
from ragsuite.core.types import QueryStr, ResponseStr

logger: logging.Logger = logging.getLogger(__name__)


# Interface: ports/LLMClient
class OpenAIClient(Runnable):
    def __init__(self, model_name: str, api_key: SecretStr | None = None):
        logger.debug("Starting OpenAIClient initialization")
        if api_key is None:
            with load_conf() as conf:
                api_key = conf.llm_api_key
        self._llm_model = ChatOpenAI(
            model=model_name, base_url=conf.paths.router_url, api_key=api_key
        )
        logger.debug("OpenAIClient initialized")

    def generate(
        self, prompt_templ: PromptTemplate, query: QueryStr, context: List[Document]
    ) -> ResponseStr:
        logger.debug(f"Generating the answer for query: {query}")
        chain = prompt_templ | self._llm_model
        response = chain.invoke({"query": query, "context": context}).content
        if isinstance(response, str):
            return ResponseStr(response)
        elif isinstance(response, list):
            response_lst = []
            for item in response:
                if isinstance(item, str):
                    response_lst.append(ResponseStr(item))
                elif isinstance(item, dict):
                    response_lst.append(ResponseStr("\n".join([f"{k}: {v}" for k, v in item.items()])))
            return ResponseStr("\n".join(response_lst))
        raise TypeError(f"Unexpected response type: {type(response)}")

    # ! Used to ensure compatibility with LangChain pipelines.
    def invoke(self, _input: LanguageModelInput, *args, **kwargs):
        return self._llm_model.invoke(_input, *args, **kwargs)
