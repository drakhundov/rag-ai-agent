from typing import List

from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

from core.ports import DocumentRetriever, ChatModel
from utilities.fusion import perform_reciprocal_rank_fusion

# TODO: incorporate different query translation strategies.

class RAGEngine:
    def __init__(
        self, doc_retriever: DocumentRetriever, chat_model: ChatModel, sys_prompt_template: PromptTemplate
    ):
        self.doc_retriever = doc_retriever
        self.chat_model = chat_model
        self.sys_prompt_template = sys_prompt_template

    def generate_answer(self, query: str, top_k: int = 4) -> str:
        # TODO: incorporate heuristic routing, query translation, and fusion.
        ctx: List[Document] = self.doc_retriever.retrieve(query, top_k=top_k)
        return self.chat_model.generate(self.sys_prompt_template, query, ctx)
