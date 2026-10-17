"""
services/embedding_service.py
------------------------------
Generates dense vector embeddings for text using Sentence Transformers.

The model is loaded once (lazy singleton) to avoid repeated disk I/O.
"""

import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config import Config

logger = logging.getLogger(__name__)

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    """Return the singleton embedding model, loading it on first call."""
    global _model
    if _model is None:
        logger.info("Loading embedding model '%s' …", Config.EMBEDDING_MODEL)
        _model = SentenceTransformer(Config.EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Encode a list of strings into an (N, D) float32 numpy array.

    Parameters
    ----------
    texts : list of str
        The text strings to encode.

    Returns
    -------
    np.ndarray  shape (len(texts), embedding_dim), dtype float32
    """
    if not texts:
        return np.empty((0,), dtype=np.float32)

    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2 normalisation — enables cosine sim via dot product
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Encode a single query string and return shape (1, D) float32 array.
    """
    return embed_texts([query])


def get_embedding_dimension() -> int:
    """Return the output dimensionality of the current model."""
    model = _get_model()
    return model.get_sentence_embedding_dimension()
