from dataclasses import dataclass
from enum import Enum
from typing import NewType, List, Optional, Dict

QueryStr = NewType("QueryStr", str)
ResponseStr = NewType("ResponseStr", str)


class TranslationMethod(Enum):
    MULTI_QUERY = "multi-query"
    HYDE = "hyde"
    IDENTITY = "identity"
    STEPBACK = "stepback"
    DECOMPOSITION = "decomposition"


TranslationRoute = NewType("TranslationRoute", List[TranslationMethod])


class TranslationRouter(Enum):
    HEURISTIC = "heuristic"


@dataclass(frozen=True)
class TranslationContext:
    query: QueryStr
    quantity: Optional[int] = None  # for MultiQuery
    max_tokens: Optional[int] = None  # for HyDE

    def to_dict(self):
        return {
            "query": self.query,
            "quantity": self.quantity,
            "max_tokens": self.max_tokens
        }


@dataclass(frozen=True)
class HeuristicAnalysisParameters:
    short_len_le: int = 12  # queries with length <= this are considered short


HeuristicAnalysis = NewType("HeuristicAnalysis", Dict[str, bool])


@dataclass
class QueryList:
    original_query: QueryStr
    queries: List[QueryStr]
    translation_router: Optional[TranslationRouter] = None
    route: TranslationRoute = None

    def __iter__(self):
        return iter(self.queries)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        return self.queries[index]

    def __eq__(self, other: 'QueryList'):
        if not isinstance(other, QueryList):
            return False
        return (self.original_query == other.original_query and
                self.translation_router == other.translation_router)

    def extend(self, querylst: 'QueryList'):
        if self != querylst:
            raise ValueError("Cannot extend QueryList with a different original_query or translation_router")
        self.queries.extend(querylst.queries)

    def to_dict(self):
        return {
            "original_query": self.original_query,
            "queries": self.queries,
            "translation_router": self.translation_router,
            "route": self.route
        }

    def add_step(self, method: TranslationMethod):
        if not method in self.route:
            self.route.append(method)
        else:
            raise ValueError(f"Method {method} already in the route: {self.route}")
