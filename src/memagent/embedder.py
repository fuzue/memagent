from __future__ import annotations

import numpy as np
from typing import List

_model = None


def _get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _model


def embed(texts: List[str]) -> np.ndarray:
    model = _get_model()
    return np.array(list(model.embed(texts)), dtype=np.float32)


def embed_batch(texts: List[str]) -> List[np.ndarray]:
    """Batched embedding for many texts at once. Much faster than embed_one in a loop."""
    if not texts:
        return []
    matrix = embed(texts)
    return [matrix[i] for i in range(len(texts))]


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]


def to_bytes(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def from_bytes(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
