from json import dumps, loads
from typing import List

from langchain_core.documents import Document


def get_unique_union(docs: List[Document]) -> List[Document]:
    flattened = [dumps(doc) for doc in docs]
    unique_docs = list(set(flattened))
    return [loads(doc) for doc in unique_docs]
