from json import dumps, loads
from typing import List

from langchain_core.documents import Document

from utilities import hashing


def get_unique_union(docs: List[Document]) -> List[Document]:
    flattened = [dumps(doc) for doc in docs]
    unique_docs = list(set(flattened))
    return [loads(doc) for doc in unique_docs]


def compute_doc_hash(doc: Document):
    payload = {
        "page_content": getattr(doc, "page_content", "") or "",
        "metadata": getattr(doc, "metadata", {}) or {},
    }
    return hashing.compute_hash(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def hash_documents(docs: List[Document]):
    return [compute_doc_hash(doc) for doc in docs]
