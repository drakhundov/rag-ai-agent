from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate

from config import load_conf


# Interface: ports/LLMService
class ChatOpenAILLMService:
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
        self, prompt_templ: PromptTemplate, question: str, context: List[Document]
    ) -> str:
        chain = prompt_templ | self._llm_model
        return chain.invoke({"question": question, "context": context}).content
