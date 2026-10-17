"""
tests/test_vector_store.py
---------------------------
Tests for the FAISS vector store: build, add, search, and session isolation.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from services.vector_store import VectorStore


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def store():
    """Fresh VectorStore for each test."""
    return VectorStore()


def _make_chunk(doc_id, chunk_idx, content, page=1, session="sess_test"):
    return {
        "id": chunk_idx,
        "document_id": doc_id,
        "chunk_index": chunk_idx,
        "content": content,
        "page_number": page,
        "session_id": session,
        "original_name": f"doc{doc_id}.pdf",
    }


# ── build_index ───────────────────────────────────────────────────────────────

class TestBuildIndex:
    def test_builds_with_valid_chunks(self, store):
        chunks = [_make_chunk(1, i, f"This is document content sentence number {i}") for i in range(5)]
        count = store.build_index("session1", chunks)
        assert count == 5

    def test_empty_chunks_creates_empty_index(self, store):
        count = store.build_index("session1", [])
        assert count == 0
        assert store.session_exists("session1") is False

    def test_replaces_existing_index(self, store):
        chunks_a = [_make_chunk(1, 0, "Machine learning is a subset of AI")]
        chunks_b = [_make_chunk(2, i, f"Deep learning content sentence {i}") for i in range(3)]
        store.build_index("session1", chunks_a)
        store.build_index("session1", chunks_b)
        assert store.get_index_size("session1") == 3


# ── add_chunks ────────────────────────────────────────────────────────────────

class TestAddChunks:
    def test_add_to_existing_index(self, store):
        initial = [_make_chunk(1, 0, "Natural language processing is fascinating")]
        store.build_index("s1", initial)
        new_chunks = [_make_chunk(2, 1, "Python is a versatile programming language")]
        store.add_chunks("s1", new_chunks)
        assert store.get_index_size("s1") == 2

    def test_add_creates_index_if_absent(self, store):
        chunks = [_make_chunk(1, 0, "First chunk of content")]
        store.add_chunks("new_session", chunks)
        assert store.get_index_size("new_session") == 1

    def test_add_empty_chunks_noop(self, store):
        chunks = [_make_chunk(1, 0, "Content")]
        store.build_index("s1", chunks)
        store.add_chunks("s1", [])
        assert store.get_index_size("s1") == 1


# ── search ────────────────────────────────────────────────────────────────────

class TestSearch:
    def _build(self, store, session):
        chunks = [
            _make_chunk(1, 0, "The Eiffel Tower is located in Paris, France", session=session),
            _make_chunk(1, 1, "Python is a high-level programming language", session=session),
            _make_chunk(1, 2, "Machine learning uses algorithms to learn from data", session=session),
            _make_chunk(1, 3, "The Louvre museum is also in Paris", session=session),
            _make_chunk(1, 4, "Deep neural networks power modern AI", session=session),
        ]
        store.build_index(session, chunks)
        return chunks

    def test_returns_results_for_valid_query(self, store):
        self._build(store, "s1")
        results = store.search("s1", "Paris tourist attractions", top_k=2)
        assert len(results) == 2

    def test_results_have_required_keys(self, store):
        self._build(store, "s1")
        results = store.search("s1", "programming language", top_k=1)
        assert len(results) == 1
        r = results[0]
        assert "content" in r
        assert "page_number" in r
        assert "similarity_score" in r

    def test_top_k_respected(self, store):
        self._build(store, "s1")
        results = store.search("s1", "AI and machine learning", top_k=3)
        assert len(results) <= 3

    def test_relevant_result_ranked_first(self, store):
        self._build(store, "s1")
        results = store.search("s1", "Python programming", top_k=5)
        contents = [r["content"] for r in results]
        assert any("Python" in c for c in contents[:2])

    def test_empty_session_returns_empty(self, store):
        results = store.search("nonexistent", "any question", top_k=5)
        assert results == []

    def test_session_isolation(self, store):
        """Chunks from session A must NOT appear in session B results."""
        self._build(store, "session_a")
        results = store.search("session_b", "Paris Eiffel Tower", top_k=5)
        assert results == []


# ── session management ────────────────────────────────────────────────────────

class TestSessionManagement:
    def test_remove_session(self, store):
        chunks = [_make_chunk(1, 0, "Test content for session removal")]
        store.build_index("s1", chunks)
        store.remove_session("s1")
        assert not store.session_exists("s1")

    def test_remove_nonexistent_session_no_error(self, store):
        store.remove_session("ghost_session")  # should not raise

    def test_get_index_size_missing_session(self, store):
        assert store.get_index_size("missing") == 0
