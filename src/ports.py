"""
Provides a common interface for Chainlit, so the background processes
could be modified, different techniques could be used without the need
to refactor.
"""

from typing import Protocol, List
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import BaseRetriever


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 4) -> List[Document]: ...

    # Access to the original LangChain retriever.
    # * Supposed to be used for piping.
    def as_langchain(self) -> BaseRetriever:
        ...


class LLMService(Protocol):
    def generate(
        self, prompt_templ: PromptTemplate, question: str, context: List[Document]
    ) -> str: ...

    # Access to the original LangChain LLM model.
    # * Supposed to be used for piping.
    def as_langchain(self) -> BaseChatModel:
        ...


class TextSplitter(Protocol):
    def split(self, docs: List[Document]) -> List[Document]: ...
