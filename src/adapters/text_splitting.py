from typing import List, Dict

import numpy as np
from langchain_core.documents import Document

from utils.vector import embed_texts, calc_pairwise_semantic_distances
from utils.string import split_into_sentences, windowed_concat

from config import load_conf


# Interface: ports/TextSplitter
class SemanticTextSplitter:
    def __init__(
        self,
        bufsz: int = None,
        breakpoint_percentile_threshold: int = None,
    ):
        conf = load_conf()
        if bufsz is None:
            bufsz = conf.concat_bufsz
        if breakpoint_percentile_threshold is None:
            breakpoint_percentile_threshold = conf.breakpoint_percentile_threshold
        self.bufsz = bufsz
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold

    def split(self, docs: List[Document]) -> List[Document]:
        """
        Chunk each Document into semantically-cohesive pieces using sentence-level
        embedding distances. Works with LangChain Documents.
        """
        all_chunks: List[Document] = []
        for doc in docs:
            sentences = split_into_sentences(doc.page_content)
            if len(sentences) <= 1:
                # Nothing to chunk; keep as-is.
                all_chunks.append(doc)
                continue

            # Build local context windows per sentence and embed.
            windowed = windowed_concat(sentences, self.bufsz)
            embs_lst = embed_texts(windowed)

            # Calculate distances between adjacent windows.
            distances = calc_pairwise_semantic_distances(embs_lst)
            if not distances:
                all_chunks.append(doc)
                continue

            # Determine breakpoints by percentile threshold.
            # Basically, if the distance lies at the top N %
            # of the list, we consider it to be too large and
            # seperate the sentences at that particular point.
            threshold = float(
                np.percentile(distances, self.breakpoint_percentile_threshold)
            )
            breakpoints = [i for i, d in enumerate(distances) if d > threshold]

            # Create chunk strings and wrap as Documents, inheriting metadata
            chunk_texts = SemanticTextSplitter.break_sentences_at_breakpoints(
                sentences, breakpoints
            )
            for idx, chunk_text in enumerate(chunk_texts):
                all_chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata=SemanticTextSplitter.inherit_metadata(doc, idx),
                    )
                )

        return all_chunks

    @staticmethod
    def inherit_metadata(parent: Document, chunk_index: int) -> Dict:
        essential_metadata_keys = ("id", "source", "page", "doc_id")
        meta = dict(parent.metadata or {})
        if "parent_id" not in meta:
            for k in essential_metadata_keys:
                if k in meta:
                    meta["parent_id"] = meta[k]
                    break
            else:
                meta["parent_id"] = id(parent)
        meta["chunk_index"] = chunk_index
        return meta

    @staticmethod
    def break_sentences_at_breakpoints(
        sentences: List[str], breakpoint_indices: List[int]
    ) -> List[str]:
        """
        Create chunk strings by splitting sentences at given breakpoint indices.
        Breakpoints refer to the boundary AFTER sentence i, i.e., between i and i+1.
        """
        if not sentences:
            return []
        chunks: List[str] = []
        start = 0
        for bp in breakpoint_indices:
            end = bp + 1  # include sentence at index bp
            chunks.append(" ".join(sentences[start:end]))
            start = end
        # last chunk
        if start < len(sentences):
            chunks.append(" ".join(sentences[start:]))
        return [c.strip() for c in chunks if c and c.strip()]
