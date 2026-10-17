"""
tests/test_app.py
-----------------
Flask API route tests using pytest-flask.
"""

import json
import io
import pytest
from unittest.mock import patch, MagicMock

from app import app as flask_app


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def app():
    flask_app.config.update({
        "TESTING": True,
        "SECRET_KEY": "test-secret",
        "WTF_CSRF_ENABLED": False,
    })
    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def session_client(client):
    """Client with an active session."""
    with client.session_transaction() as sess:
        sess["session_id"] = "test-session-123"
    return client


# ── Health check ──────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        res = client.get("/health")
        assert res.status_code == 200

    def test_health_json_shape(self, client):
        data = client.get("/health").get_json()
        assert "status" in data
        assert data["status"] == "ok"
        assert "llm_provider" in data
        assert "embedding_model" in data


# ── Index route ───────────────────────────────────────────────────────────────

class TestIndex:
    def test_index_returns_200(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert b"DocuMind" in res.data or b"<!DOCTYPE html>" in res.data


# ── Upload route ──────────────────────────────────────────────────────────────

class TestUpload:
    def _pdf_file(self, content=b"%PDF-1.4 test"):
        return (io.BytesIO(content), "test.pdf")

    def test_no_file_returns_400(self, session_client):
        res = session_client.post("/api/upload", data={})
        assert res.status_code == 400
        data = res.get_json()
        assert data["success"] is False

    def test_wrong_extension_rejected(self, session_client):
        data = {"file": (io.BytesIO(b"content"), "malware.exe")}
        res = session_client.post("/api/upload", data=data,
                                  content_type="multipart/form-data")
        assert res.status_code == 400
        assert res.get_json()["success"] is False

    def test_empty_filename_rejected(self, session_client):
        data = {"file": (io.BytesIO(b""), "")}
        res = session_client.post("/api/upload", data=data,
                                  content_type="multipart/form-data")
        assert res.status_code == 400

    @patch("app.process_pdf")
    @patch("app.insert_document", return_value=42)
    def test_corrupted_pdf_rejected(self, mock_insert, mock_process, session_client):
        from services.pdf_processor import PDFValidationError
        mock_process.side_effect = PDFValidationError("Corrupted PDF")

        data = {"file": (io.BytesIO(b"%PDF-1.4"), "bad.pdf")}
        res = session_client.post("/api/upload", data=data,
                                  content_type="multipart/form-data")
        assert res.status_code == 400
        assert "Corrupted" in res.get_json()["error"]


# ── Documents route ───────────────────────────────────────────────────────────

class TestDocuments:
    def test_list_documents_returns_json(self, session_client):
        with patch("app.get_documents_by_session", return_value=[]):
            res = session_client.get("/api/documents")
            assert res.status_code == 200
            data = res.get_json()
            assert data["success"] is True
            assert "documents" in data

    def test_delete_nonexistent_document_404(self, session_client):
        with patch("app.get_document_by_id", return_value=None):
            res = session_client.delete("/api/documents/9999")
            assert res.status_code == 404


# ── Chat route ────────────────────────────────────────────────────────────────

class TestChat:
    def test_missing_question_returns_400(self, session_client):
        res = session_client.post("/api/chat",
                                  data=json.dumps({}),
                                  content_type="application/json")
        assert res.status_code == 400
        assert session_client.post("/api/chat",
                                   data=json.dumps({}),
                                   content_type="application/json").get_json()["success"] is False

    def test_question_too_long_rejected(self, session_client):
        long_q = "A" * 2001
        res = session_client.post("/api/chat",
                                  data=json.dumps({"question": long_q}),
                                  content_type="application/json")
        assert res.status_code == 400

    @patch("app.answer_question")
    @patch("app._ensure_index")
    @patch("app.insert_message")
    def test_valid_question_returns_answer(self, mock_insert, mock_ensure, mock_rag,
                                           session_client):
        mock_rag.return_value = MagicMock(
            answer="Test answer",
            sources=[{"document_name": "test.pdf", "pages": [1]}],
            chunks_used=3,
            error=None,
            to_dict=lambda: {
                "answer": "Test answer",
                "sources": [{"document_name": "test.pdf", "pages": [1]}],
                "chunks_used": 3,
                "error": None,
            },
        )
        res = session_client.post(
            "/api/chat",
            data=json.dumps({"question": "What is this document about?"}),
            content_type="application/json",
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert "answer" in data


# ── Chat history routes ───────────────────────────────────────────────────────

class TestChatHistory:
    def test_get_history_returns_list(self, session_client):
        with patch("app.get_chat_history", return_value=[]):
            res = session_client.get("/api/chat/history")
            assert res.status_code == 200
            data = res.get_json()
            assert "history" in data

    def test_clear_history(self, session_client):
        with patch("app.clear_chat_history"):
            res = session_client.delete("/api/chat/history")
            assert res.status_code == 200
            assert res.get_json()["success"] is True


# ── Error handlers ────────────────────────────────────────────────────────────

class TestErrorHandlers:
    def test_404_handler(self, client):
        res = client.get("/api/nonexistent-endpoint")
        assert res.status_code == 404
        data = res.get_json()
        assert data["success"] is False
