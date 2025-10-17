import logging
from typing import List

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from core.ports import DocumentRetriever, ChatModel
from core.types import QueryStr, QueryList, TranslationContext
from routing.HeuristicRouter import HeuristicRouter
from utilities.fusion import perform_reciprocal_rank_fusion

logger: logging.Logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(
        self, doc_retriever: DocumentRetriever, chat_model: ChatModel, sys_prompt_template: PromptTemplate
    ):
        self.doc_retriever = doc_retriever
        self.chat_model = chat_model
        self.sys_prompt_template = sys_prompt_template

    def generate_answer(self, query: QueryStr, top_k: int = 4) -> str:
        logger.debug(f"Generating answer for query: {query}")
        router = HeuristicRouter(
            ctx=TranslationContext(query=query, quantity=top_k, max_tokens=256),
            chat_model=self.chat_model
        )
        router.route()
        qlist: QueryList = router.run_route()
        docs: List[List[Document]] = []
        for q in qlist:
            docs.append(self.doc_retriever.retrieve(q, top_k=top_k))
        # Weed out the most relevant documents using Reciprocal Rank Fusion.
        ranked_docs: List[Document] = perform_reciprocal_rank_fusion(docs)
        return self.chat_model.generate(self.sys_prompt_template, query, ranked_docs)
