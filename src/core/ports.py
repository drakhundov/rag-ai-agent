"""
Provides a common interface for the application, so the background processes could
be modified, different techniques could be used without the need to refactor.
"""

from typing import Protocol, List

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from core.types import (
    QueryStr,
    ResponseStr,
    QueryList,
    TranslationContext,
    TranslationRouter,
)


class ChatModel(Protocol):
    def generate(
        self, prompt_templ: PromptTemplate, query: QueryStr, context: List[Document]
    ) -> ResponseStr: ...


class DocumentRetriever(Protocol):
    def retrieve(self, query: QueryStr, top_k: int = 4) -> List[Document]: ...


class TextSplitter(Protocol):
    def split(self, docs: List[Document]) -> List[Document]: ...


class QueryTranslator(Protocol):
    def translate(
        self, ctx: TranslationContext, router: TranslationRouter
    ) -> QueryList: ...
