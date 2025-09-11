from typing import List

from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from core.config import load_conf, QueryStr
from core.protocols import DocumentRetriever, ChatModel


# TODO: incorporate different query translation strategies.

class RAGEngine:
    def __init__(
        self, doc_retriever: DocumentRetriever, chat_model: ChatModel, sys_prompt_template: PromptTemplate
    ):
        self.doc_retriever = doc_retriever
        self.chat_model = chat_model
        self.sys_prompt_template = sys_prompt_template
        with load_conf() as conf:
            self.multi_query_prompt = PromptTemplate(
                input_variables=conf.prompt_templs.multi_query_rag_prompt.input_variables,
                template=conf.prompt_templs.multi_query_rag_prompt.template
            )

    # Multi-query retrieval.
    def gen_multiple_queries(self, q: QueryStr, n: int) -> List[QueryStr]:
        chain = (
            self.multi_query_prompt |
            self.chat_model() |
            StrOutputParser() |
            (lambda x: x.split("\n"))
        )
        return chain.invoke({"question": q, "quantity": n})

    def generate_answer(self, question: str, k: int = 4) -> str:
        ctx: List[Document] = self.doc_retriever.retrieve(question, k=k)
        return self.chat_model.generate(self.sys_prompt_template, question, ctx)
