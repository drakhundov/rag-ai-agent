import hashlib
from typing import List, Optional

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
    embs: List[np.ndarray] = [None] * len(texts)
    hashed: List[Optional[str]] = [None] * len(texts)
    cache_hit_idx = set()
    cmng = CacheManager("texts")
    for i, text in enumerate(texts):
        hashed[i] = hashlib.md5(text.encode()).hexdigest()
        try:
            hit, cached_emb = cmng.get(
                cache_id=hashed[i],
                attr=CacheAttr.EMBEDDINGS,
                read_as_binary=True
            )
            cache_hit_idx.add(i)
            if isinstance(cached_emb, (bytes, bytearray)):
                arr = np.frombuffer(cached_emb, dtype=np.float32).copy()
            else:
                arr = np.array(cached_emb, dtype=np.float32)
            embs[i] = arr
        except FileNotFoundError:
            continue
    # If all texts have been retrieved from cache, return cached data.
    if len(cache_hit_idx) == len(texts):
        return np.vstack(embs).astype(np.float32)
    # Otherwise, run the encoder for some texts.
    if model_name is None:
        # Use the default embedding model if not specified.
        model_name = conf.models.emb_model_name
    encoder = HuggingFaceEndpointEmbeddings(
        model=model_name,
        huggingfacehub_api_token=conf.hf_token.get_secret_value()
    )
    missing_indices = [i for i in range(len(texts)) if i not in cache_hit_idx]
    missing_texts = [texts[i] for i in missing_indices]
    if missing_texts:
        new_embs = encoder.embed_documents(missing_texts)
        for idx, new_emb in zip(missing_indices, new_embs):
            arr = np.array(new_emb, dtype=np.float32)
            embs[idx] = arr
    return np.vstack(embs).astype(np.float32)
