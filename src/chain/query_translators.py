from typing import Dict, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from core.config import load_conf
from core.ports import ChatModel
from core.types import QueryStr, QueryList, TranslationContext


# Since all query translation methods share the same steps, I've implemented
# a general solution, which only needs a prompt. It is used via composition
# in the specific translators.
class _QueryTranslatorImpl:
    def __init__(
        self,
        chat_model: ChatModel,
        prompt_templ: PromptTemplate,
    ):
        self.chat_model = chat_model
        self.prompt_templ = prompt_templ

    def run(self, query: QueryStr | None, ctx_dict: Dict | None) -> QueryList:
        ctx_dict = ctx_dict or {}
        chain = (
            self.prompt_templ |
            self.chat_model.as_langchain() |
            StrOutputParser() |
            (lambda x: x.split("\n"))
        )
        if query is not None:
            ctx_dict.update({"query": query})
        elif ctx_dict.get("query") is None:
            raise ValueError("Either query or ctx['query'] must be provided.")
        llm_response = chain.invoke(ctx_dict)
        # Remove dupliates while preserving order.
        queries = []
        seen = set()
        for s in [query, *llm_response]:
            s_norm = s.strip()
            if s_norm and s_norm not in seen:
                seen.add(s_norm)
                queries.append(s_norm)
        return QueryList(
            original_query=ctx_dict["query"],
            queries=queries
        )


class MultiQueryTranslator:
    def __init__(self, chat_model: ChatModel):
        with load_conf() as conf:
            prompt_templ = PromptTemplate(
                input_variables=conf.prompt_templs.multi_query_rag_prompt.input_variables,
                template=conf.prompt_templs.multi_query_rag_prompt.template
            )
        self._impl = _QueryTranslatorImpl(
            chat_model=chat_model,
            prompt_templ=prompt_templ
        )

    def translate(self, query: QueryStr, ctx: TranslationContext) -> QueryList:
        return self._impl.run(query, ctx.to_dict())


class HyDETranslator:
    def __init__(self, chat_model: ChatModel):
        with load_conf() as conf:
            prompt_templ = PromptTemplate(
                input_variables=conf.prompt_templs.hyde_rag_prompt.input_variables,
                template=conf.prompt_templs.hyde_rag_prompt.template
            )
        self._impl = _QueryTranslatorImpl(
            chat_model=chat_model,
            prompt_templ=prompt_templ
        )

    def translate(self, query: QueryStr, ctx: TranslationContext) -> QueryList:
        return self._impl.run(query, ctx.to_dict())


class DecompositionTranslator:
    def __init__(self, chat_model: ChatModel):
        with load_conf() as conf:
            prompt_templ = PromptTemplate(
                input_variables=conf.prompt_templs.decomposition_rag_prompt.input_variables,
                template=conf.prompt_templs.decomposition_rag_prompt.template
            )
        self._impl = _QueryTranslatorImpl(
            chat_model=chat_model,
            prompt_templ=prompt_templ
        )

    def translate(self, query: QueryStr, ctx: TranslationContext) -> QueryList:
        return self._impl.run(query, ctx.to_dict())


class StepUpTranslator:
    def __init__(self, chat_model: ChatModel):
        with load_conf() as conf:
            prompt_templ = PromptTemplate(
                input_variables=conf.prompt_templs.step_up_rag_prompt.input_variables,
                template=conf.prompt_templs.step_up_rag_prompt.template
            )
        self._impl = _QueryTranslatorImpl(
            chat_model=chat_model,
            prompt_templ=prompt_templ
        )

    def translate(self, query: QueryStr, ctx: TranslationContext) -> QueryList:
        return self._impl.run(query, ctx.to_dict())


class IdentityTranslator:
    def translate(self, query: QueryStr, ctx: TranslationContext) -> QueryList:
        return QueryList(
            original_query=query,
            queries=[query]
        )
