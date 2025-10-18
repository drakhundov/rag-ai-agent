import logging
from typing import List, Dict, Optional

from langchain_core.documents import Document

logger: logging.Logger = logging.getLogger()


def perform_reciprocal_rank_fusion(docs: List[List[Document]], top_k: Optional[int] = None, k_rrf: int = 60) -> List[
    Document]:
    logger.debug("Performing reciprocal rank fusion")
    scores: Dict[str, float] = {}
    first_seen: Dict[str, Document] = {}
    for ranking in docs:
        for r, doc in enumerate(ranking, start=1):
            doc_id = doc.metadata.get("id") or doc.metadata.get("chunk_id") or doc.page_content[:80]
            if doc_id not in first_seen:
                first_seen[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + r)
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [first_seen[doc_id] for doc_id, _ in fused][:top_k]
