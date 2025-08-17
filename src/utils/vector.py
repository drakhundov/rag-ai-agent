from typing import List

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from config import load_conf

conf = load_conf()


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def calc_pairwise_semantic_distances(embs: np.ndarray) -> List[float]:
    """
    Given a list of embeddings, returns a list of
    corresponding pairwise distances. Calculates
    distances by utilizing cosine similarity.
    dist[0] - distance btw embs[0] and embs[1].
    """
    # One element or empty.
    if embs.shape[0] <= 1:
        return []
    dists: List[float] = []
    for i in range(embs.shape[0] - 1):
        d = 1.0 - cosine_similarity(embs[i], embs[i + 1])
        dists.append(float(d))
    return dists


def embed_texts(texts: List[str], model_name: str = None) -> np.ndarray:
    """
    Returns a list of embedded strings (vectors) for
    each corresponding element in the original list.
    """
    if not texts:
        return np.empty((0,))
    if model_name is None:
        # Use the default embedding model if not specified.
        model_name = conf.models.emb_model_name
    encoder = HuggingFaceEmbeddings(model_name=model_name)
    embs = encoder.embed_documents(texts)
    return np.array(embs, dtype=np.float32)
