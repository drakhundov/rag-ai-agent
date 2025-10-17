import logging
from typing import List

from chain.query_translators import MultiQueryTranslator, HyDETranslator, IdentityTranslator, StepBackTranslator, \
    DecompositionTranslator
from core.ports import ChatModel
from core.types import QueryList, TranslationMethod, TranslationContext, TranslationRouter
from routing.HeuristicAnalyzer import HeuristicAnalyzer

logger: logging.Logger = logging.getLogger(__name__)


# Interface: ports/TranslationRouter
class HeuristicRouter:
    def __init__(self, ctx: TranslationContext, chat_model: ChatModel):
        self.ctx = ctx
        self.chat_model = chat_model
        self.route = []
        self.qlist: QueryList = QueryList(
            original_query=self.ctx.query,
            queries=[self.ctx.query],
            translation_router=TranslationRouter.HEURISTIC
        )
        self.route_constructed: bool = False
        self.add_translation_step(TranslationMethod.IDENTITY)
        self.translator_map = {
            TranslationMethod.IDENTITY: IdentityTranslator(),
            TranslationMethod.MULTI_QUERY: MultiQueryTranslator(chat_model=self.chat_model),
            TranslationMethod.HYDE: HyDETranslator(chat_model=self.chat_model),
            TranslationMethod.STEPBACK: StepBackTranslator(chat_model=self.chat_model),
            TranslationMethod.DECOMPOSITION: DecompositionTranslator(chat_model=self.chat_model),
        }
        logger.debug(f"HeuristicRouter initialized (query='{self.ctx.query}')")

    def route(self):
        logger.debug("Routing the query...")
        # Figure out the pipeline.
        q = self.ctx.query
        q_len = len(q.split())
        text = q.lower()
        analyzer = HeuristicAnalyzer(q)
        self.analysis = analyzer.analyze()

        if self.analysis["has_logical_operators"] or self.analysis["is_comparative"]:
            self.add_translation_step(TranslationMethod.DECOMPOSITION)
        if not self.analysis['is_short']:
            self.add_translation_step(TranslationMethod.MULTI_QUERY)
        if self.analysis['is_ambiguous']:
            self.add_translation_step(TranslationMethod.STEPBACK)
            self.add_translation_step(TranslationMethod.HYDE)

        self.route_constructed = True
        logger.debug(f"HeuristicRouter route: {self.route}")

    def add_translation_step(self, method: TranslationMethod):
        """Use to construct the route."""
        logger.debug(f"Adding translation step: {method.name}")
        if self.qlist is None:
            raise ValueError("`qlist` hasn't been initialized yet.")
        elif not isinstance(method, TranslationMethod):
            raise ValueError("`method` must be an instance of TranslationMethod Enum.")
        self.qlist.route.append(method)

    def run_route(self) -> QueryList:
        """Execute the route."""
        logger.debug("Running the constructed route...")
        if not self.route_constructed:
            raise RuntimeError("Route hasn't been constructed yet. Call `route()` first.")
        self.translators: List = [self.translator_map[method] for method in self.qlist.route]
        for translator in self.translators:
            self.qlist.extend(translator.translate(self.ctx))
        return self.qlist
