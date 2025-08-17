from typing import List

from langchain_core.documents import Document
from langchain.prompts import PromptTemplate

from ports import Retriever, LLMService


class RetrieverService:
    def __init__(
        self, retriever: Retriever, llm_model: LLMService, prompt_template: PromptTemplate
    ):
        self.retriever = retriever
        self.llm_model = llm_model
        self.prompt_template = prompt_template

    def answer(self, question: str, k: int = 4) -> str:
        ctx: List[Document] = self.retriever.retrieve(question, k=k)
        return self.llm_model.generate(self.prompt_template, question, ctx)
