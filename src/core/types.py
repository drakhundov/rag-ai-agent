from dataclasses import dataclass
from enum import Enum
from typing import NewType, List, Optional

QueryStr = NewType("QueryStr", str)
ResponseStr = NewType("ResponseStr", str)


class TranslationMethod(Enum):
    MULTI_QUERY = "multi-query"
    HYDE = "hyde"
    IDENTITY = "identity"


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


@dataclass
class QueryList:
    original_query: QueryStr
    queries: List[QueryStr]
    route: Optional[List[TranslationMethod]] = None
    translation_router: Optional[TranslationRouter] = None

    def __iter__(self):
        return iter(self.queries)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        return self.queries[index]

    def add_step(self, method: TranslationMethod):
        if not method in self.route:
            self.route.append(method)
        else:
            raise ValueError(f"Method {method} already in the route: {self.route}")
