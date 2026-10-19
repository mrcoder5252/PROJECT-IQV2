"""
Project-IQ: shared embedding model loader and vector math.

One cached SentenceTransformer instance is shared by the vector store and the
team matcher. The model is loaded from the local HuggingFace cache when present
and downloaded otherwise (so a fresh machine works without manual steps).
"""

from __future__ import annotations

import threading
from typing import Dict, Sequence

import numpy as np

from src.config import config
from src.logger import get_logger

logger = get_logger("project_iq.embeddings")

_MODELS: Dict[str, object] = {}
_LOCK = threading.Lock()


def get_embedder(model_name: str | None = None):
    """Return a cached SentenceTransformer for ``model_name``."""
    name = model_name or config.embedding_model_name
    with _LOCK:
        if name not in _MODELS:
            from sentence_transformers import SentenceTransformer

            try:
                _MODELS[name] = SentenceTransformer(name, local_files_only=True)
            except Exception:
                logger.info("Embedding model '%s' not cached locally; downloading once.", name)
                _MODELS[name] = SentenceTransformer(name)
        return _MODELS[name]


def encode(texts: Sequence[str], model_name: str | None = None) -> np.ndarray:
    """Encode texts to L2-normalised vectors (so dot product == cosine similarity)."""
    model = get_embedder(model_name)
    return np.asarray(model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two 1-D vectors (safe for zero vectors)."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0
