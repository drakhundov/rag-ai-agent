"""
Provides a common interface for Chainlit, so the background processes
could be modified, different techniques could be used without the need
to refactor.
"""

from typing import Protocol, List
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 4) -> List[Document]: ...


class LLMService(Protocol):
    def generate(
        self, prompt_templ: PromptTemplate, question: str, context: List[Document]
    ) -> str: ...


class TextSplitter(Protocol):
    def split(self, docs: List[Document]) -> List[Document]: ...
