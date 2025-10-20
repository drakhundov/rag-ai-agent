from langchain_core.documents import Document

from utilities import fusion


def create_dummy__doc(id: str = None, content: str = "") -> Document:
    meta = {"id": id} if id is not None else {}
    return Document(page_content=content or id or "content", metadata=meta)


def extract_doc_id(doc: Document) -> str:
    return doc.metadata.get("id") or doc.metadata.get("chunk_id") or doc.page_content[:80]


def test_basic_fusion_order():
    """
    Two rankings:
      - ranking_a: doc1 (rank1), doc2 (rank2), doc3 (rank3)
      - ranking_b: doc2 (rank1), doc1 (rank2), doc4 (rank3)
    Using a small k_rrf to make score differences obvious and deterministic.
    Expect stable tie-breaking according to first-seen insertion order.
    """
    ranking_a = [
        create_dummy__doc("doc1"),
        create_dummy__doc("doc2"),
        create_dummy__doc("doc3"),
    ]
    ranking_b = [
        create_dummy__doc("doc2"),
        create_dummy__doc("doc1"),
        create_dummy__doc("doc4"),
    ]

    fused = fusion.perform_rrf([ranking_a, ranking_b], k_rrf=1)
    ids = [extract_doc_id(d) for d in fused]

    # Expected order.
    assert ids == ["doc1", "doc2", "doc3", "doc4"]


def test_top_k_truncation():
    """
    Ensure top_k limits the returned fused list.
    """
    ranking_a = [create_dummy__doc("a1"), create_dummy__doc("a2"), create_dummy__doc("a3")]
    ranking_b = [create_dummy__doc("b1"), create_dummy__doc("a1"), create_dummy__doc("b2")]

    fused_top2 = perform_reciprocal_rank_fusion([ranking_a, ranking_b], top_k=2, k_rrf=1)
    ids = [extract_doc_id(d) for d in fused_top2]
    assert len(ids) == 2
    assert ids[0] in {"a1", "b1", "a2"}


def test_top_k_none_returns_all():
    """
    top_k=None should return all unique fused documents.
    (Tests that implementation does not accidentally drop the last item.)
    """
    ranking_a = [create_dummy__doc("x1"), create_dummy__doc("x2")]
    ranking_b = [create_dummy__doc("x2"), create_dummy__doc("x3")]

    fused = perform_reciprocal_rank_fusion([ranking_a, ranking_b], top_k=None, k_rrf=1)
    ids = [extract_doc_id(d) for d in fused]

    # Unique docs are x1, x2, x3 => expect all 3 returned.
    assert set(ids) == {"x1", "x2", "x3"}
    assert len(ids) == 3
