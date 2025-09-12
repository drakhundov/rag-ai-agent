from typing import List

from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from core.config import load_conf


# Interface: ports/ChatModel
class OpenAIChatModel:
    def __init__(self, model_name: str = None, api_key: str = None):
        self.conf = load_conf()
        if model_name is None:
            model_name = self.conf.models.chat_model_name
        if api_key is None:
            api_key = self.conf.openai_api_key
        self._llm_model = ChatOpenAI(
            model=model_name,
            base_url=self.conf.paths.router_url,
            api_key=api_key
        )

    def generate(
        self, prompt_templ: PromptTemplate, query: str, context: List[Document]
    ) -> str:
        chain = prompt_templ | self._llm_model
        return chain.invoke({"question": query, "context": context}).content

    def __ror__(self, other):
        return other | self._llm_model
