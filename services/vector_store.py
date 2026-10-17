"""
services/vector_store.py
-------------------------
In-memory FAISS vector index, scoped per user session.

Why per-session?
  Each browser session gets its own isolated FAISS index built from that
  session's uploaded documents. This avoids one user seeing another user's
  documents in search results.

Architecture
  VectorStore  — manages a dict of {session_id → SessionIndex}
  SessionIndex — wraps a faiss.IndexFlatIP (Inner Product) index plus
                 the metadata list needed to map hit positions back to
                 the original chunk records.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import faiss
import numpy as np

from services.embedding_service import embed_query, embed_texts, get_embedding_dimension
from config import Config

logger = logging.getLogger(__name__)


@dataclass
class SessionIndex:
    """Holds the FAISS index and associated chunk metadata for one session."""
    index: faiss.Index
    metadata: list[dict] = field(default_factory=list)

    @property
    def size(self) -> int:
        return self.index.ntotal


class VectorStore:
    """
    Thread-safe in-memory vector store keyed by session_id.

    Note: In-memory storage means the index is lost on server restart.
    For production use, persist the index to disk with faiss.write_index().
    This is acceptable for an academic prototype.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionIndex] = {}

    # ── Building / updating the index ───────────────────────────────────────

    def build_index(self, session_id: str, chunks: list[dict]) -> int:
        """
        Create (or replace) the FAISS index for *session_id* from *chunks*.

        Parameters
        ----------
        session_id : str
        chunks     : list of dicts with keys:
                     id, content, page_number, document_id, original_name

        Returns the number of vectors indexed.
        """
        if not chunks:
            logger.warning("build_index called with empty chunk list for session %s", session_id)
            self._sessions[session_id] = SessionIndex(
                index=faiss.IndexFlatIP(get_embedding_dimension()), metadata=[]
            )
            return 0

        texts = [c["content"] for c in chunks]
        embeddings = embed_texts(texts)  # (N, D) float32

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)   # Inner Product → cosine sim (vectors are L2-normalised)
        index.add(embeddings)

        self._sessions[session_id] = SessionIndex(index=index, metadata=list(chunks))
        logger.info("Built FAISS index for session %s: %d vectors, dim=%d", session_id, index.ntotal, dim)
        return index.ntotal

    def add_chunks(self, session_id: str, chunks: list[dict]) -> int:
        """
        Add new chunks to an existing session index (incremental update).
        Creates the index if it doesn't exist yet.
        """
        if not chunks:
            return 0

        texts = [c["content"] for c in chunks]
        embeddings = embed_texts(texts).astype(np.float32)

        if session_id not in self._sessions:
            dim = embeddings.shape[1]
            self._sessions[session_id] = SessionIndex(
                index=faiss.IndexFlatIP(dim), metadata=[]
            )

        session = self._sessions[session_id]
        session.index.add(embeddings)
        session.metadata.extend(chunks)
        return session.size

    # ── Retrieval ────────────────────────────────────────────────────────────

    def search(
        self,
        session_id: str,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Retrieve the top-k most relevant chunks for *query* from the
        session index.

        Returns a list of dicts (chunk metadata + similarity score),
        sorted by descending similarity.
        Returns an empty list if the session has no documents.
        """
        top_k = top_k or Config.TOP_K_RESULTS

        if session_id not in self._sessions:
            return []

        session = self._sessions[session_id]
        if session.size == 0:
            return []

        q_emb = embed_query(query)  # (1, D) float32

        actual_k = min(top_k, session.size)
        scores, indices = session.index.search(q_emb, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(session.metadata[idx])
            chunk["similarity_score"] = float(score)
            results.append(chunk)

        # Already sorted descending by FAISS
        return results

    # ── Session management ───────────────────────────────────────────────────

    def remove_session(self, session_id: str) -> None:
        """Drop the entire index for a session (e.g. on logout / clear)."""
        self._sessions.pop(session_id, None)

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions and self._sessions[session_id].size > 0

    def get_index_size(self, session_id: str) -> int:
        if session_id not in self._sessions:
            return 0
        return self._sessions[session_id].size


# Singleton instance used by Flask routes
vector_store = VectorStore()
