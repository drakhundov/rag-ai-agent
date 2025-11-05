from typing import List
import hashlib

import numpy as np
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from core.config import load_conf
from core.types import CacheAttr
from services.CacheManager import CacheManager

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
    embs = []
    cache_hit_idx = set()
    cmng = CacheManager("texts")
    for i, text in enumerate(texts):
        hashval = hashlib.md5(text.encode()).hexdigest()
        hit, cached_emb = cmng.get(
            cache_id=hashval,
            attr=CacheAttr.EMBEDDINGS,
            read_as_binary=True
        )
        if hit:
            cache_hit_idx.add(i)
            embs[i] = cached_emb
    # If all texts have been retrieved from cache, return cached data.
    if len(cache_hit_idx) == len(texts):
        return np.array(embs, dtype=np.float32)
    # Otherwise, run the encoder for some texts.
    if model_name is None:
        # Use the default embedding model if not specified.
        model_name = conf.models.emb_model_name
    encoder = HuggingFaceEndpointEmbeddings(
        model=model_name,
        huggingfacehub_api_token=conf.hf_token.get_secret_value()
    )
    for i, text in enumerate(texts):
        if i in cache_hit_idx:
            continue
        embs[i] = encoder.embed_query(text)
    embs = encoder.embed_documents(texts)
    return np.array(embs, dtype=np.float32)
