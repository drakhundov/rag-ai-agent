import logging
from typing import Dict, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from core.config import load_conf
from core.ports import ChatModel
from core.types import QueryList, TranslationContext, TranslationRouter

logger: logging.Logger = logging.getLogger()


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

    def run(self, ctx_dict: Dict) -> QueryList:
        logger.debug("Running _QueryTranslatorImpl")
        ctx_dict = ctx_dict or {}
        chain = (
            self.prompt_templ
            | self.chat_model
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )
        llm_response = chain.invoke(ctx_dict)
        # Remove dupliates while preserving order.
        queries = []
        seen = set()
        for s in llm_response:
            s_norm = s.strip()
            if s_norm and s_norm not in seen:
                seen.add(s_norm)
                queries.append(s_norm)
        return QueryList(
            original_query=ctx_dict.get("query"),
            queries=queries,
            translation_router=ctx_dict.get("translation_router", None),
        )


class BaseTranslator:
    def __init__(self, chat_model: ChatModel, prompt_templ: PromptTemplate):
        self.chat_model = chat_model
        self.prompt_templ = prompt_templ
        self._impl: Optional[_QueryTranslatorImpl] = None

    def initialize_impl(self):
        if self._impl is not None:
            return
        self._impl = _QueryTranslatorImpl(
            chat_model=self.chat_model, prompt_templ=self.prompt_templ
        )

    def translate(
        self,
        ctx: TranslationContext,
        router: TranslationRouter = TranslationRouter.HEURISTIC,
    ) -> QueryList:
        self.initialize_impl()
        ctx_dict = ctx.to_dict()
        ctx_dict["translation_router"] = router
        return self._impl.run(ctx_dict)


class MultiQueryTranslator(BaseTranslator):
    def __init__(self, chat_model: ChatModel):
        logger.debug("Starting MultiQueryTranslator initialization")
        with load_conf() as conf:
            prompt_templ = PromptTemplate(
                input_variables=conf.prompt_templs.multi_query_rag_prompt.input_variables,
                template=conf.prompt_templs.multi_query_rag_prompt.template,
            )
        super().__init__(chat_model, prompt_templ)


class HyDETranslator(BaseTranslator):
    def __init__(self, chat_model: ChatModel):
        logger.debug("Starting HyDETranslator initialization")
        with load_conf() as conf:
            prompt_templ = PromptTemplate(
                input_variables=conf.prompt_templs.hyde_rag_prompt.input_variables,
                template=conf.prompt_templs.hyde_rag_prompt.template,
            )
        super().__init__(chat_model, prompt_templ)


class DecompositionTranslator(BaseTranslator):
    def __init__(self, chat_model: ChatModel):
        logger.debug("Starting DecompositionTranslator initialization")
        with load_conf() as conf:
            prompt_templ = PromptTemplate(
                input_variables=conf.prompt_templs.decomposition_rag_prompt.input_variables,
                template=conf.prompt_templs.decomposition_rag_prompt.template,
            )
        super().__init__(chat_model, prompt_templ)


class StepBackTranslator(BaseTranslator):
    def __init__(self, chat_model: ChatModel):
        logger.debug("Starting StepBackTranslator initialization")
        with load_conf() as conf:
            prompt_templ = PromptTemplate(
                input_variables=conf.prompt_templs.step_back_rag_prompt.input_variables,
                template=conf.prompt_templs.step_back_rag_prompt.template,
            )
        super().__init__(chat_model, prompt_templ)


class IdentityTranslator(BaseTranslator):
    def __init__(self, chat_model: Optional[ChatModel] = None):
        logger.debug("Starting IdentityTranslator initialization")
        # Identity translator doesn't need a chat model.

    def translate(
        self,
        ctx: TranslationContext,
        router: TranslationRouter = TranslationRouter.HEURISTIC,
    ) -> QueryList:
        return QueryList(
            original_query=ctx.query, queries=[ctx.query], translation_router=router
        )
