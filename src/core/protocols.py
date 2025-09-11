"""
Provides a common interface for Chainlit, so the background processes could be
modified, different techniques could be used without the need to refactor.
"""

from typing import Protocol, Optional, List, Dict

from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever


class DocumentRetriever(Protocol):
    def retrieve(self, query: str, k: int = 4) -> List[Document]: ...

    # Access to the original LangChain retriever.
    # * Supposed to be used for piping.
    def __call__(self) -> BaseRetriever:
        ...


class ChatModel(Protocol):
    def generate(
        self, prompt_templ: PromptTemplate, question: str, context: List[Document]
    ) -> str: ...

    # Access to the original LangChain LLM model.
    # * Supposed to be used for piping.
    def __call__(self) -> BaseChatModel:
        ...


class TextSplitter(Protocol):
    def split(self, docs: List[Document]) -> List[Document]:
        ...


class QueryTranslator(Protocol):
    def translate(self, query: str, ctx: Optional[Dict]) -> List[str]:
        ...


class TranslatorRouter(Protocol):
    def route(self, query: str, ctx: Optional[Dict]) -> List[QueryTranslator]:
        ...
