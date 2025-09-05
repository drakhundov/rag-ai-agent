from typing import List

from langchain_core.documents import Document
from langchain.prompts import PromptTemplate

from ports import Retriever, LLMService
from config import load_conf


# TODO: incorporate different query translation strategies.

class RetrieverService:
    def __init__(
        self, retriever: Retriever, llm_model: LLMService, sys_prompt_template: PromptTemplate
    ):
        self.retriever = retriever
        self.llm_model = llm_model
        self.sys_prompt_template = sys_prompt_template
        conf = load_conf()
        self.multi_query_prompt = PromptTemplate(
            input_variables=conf.prompt_templs.multi_query_rag_prompt.input_variables,
            template=conf.prompt_templs.multi_query_rag_prompt.template
        )

    def multi_query(self, q: str, n: int):
        chain = self.multi_query_prompt | self.llm_model.as_langchain()
        return chain.invoke({"question": q, "quantity": n}).content

    def answer(self, question: str, k: int = 4) -> str:
        ctx: List[Document] = self.retriever.retrieve(question, k=k)
        return self.llm_model.generate(self.sys_prompt_template, question, ctx)
