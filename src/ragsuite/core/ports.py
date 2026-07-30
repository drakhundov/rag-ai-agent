"""
Provides a common interface for the application, so the background processes could
be modified, different techniques could be used without the need to refactor.
"""

from typing import Protocol, List

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from ragsuite.core.types import (
    QueryStr,
    ResponseStr,
    QueryList,
    TranslationContext,
    TranslationRoute
)


class LLMClient(Protocol):
    def generate(
        self, prompt_templ: PromptTemplate, query: QueryStr, context_docs: List[Document]
    ) -> ResponseStr: ...


class Retriever(Protocol):
    def retrieve(self, query: QueryStr, top_k: int = 5) -> List[Document]: ...


class Splitter(Protocol):
    def split(self, docs: List[Document]) -> List[Document]: ...


class QueryTranslator(Protocol):
    def translate(
        self, ctx: TranslationContext
    ) -> QueryList: ...


class TranslationRouter(Protocol):
    def route(self, config: TranslationContext) -> TranslationRoute: ...
