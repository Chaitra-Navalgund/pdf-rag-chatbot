"""
database/db.py
--------------
SQLite database layer.

Tables:
  documents  — metadata about each uploaded PDF
  chunks     — individual text chunks extracted from PDFs
  chat_history — per-session question/answer pairs
"""

import sqlite3
import logging
from datetime import datetime
from typing import Optional

from config import Config

logger = logging.getLogger(__name__)


def get_connection() -> sqlite3.Connection:
    """Return a new SQLite connection with row_factory set."""
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row          # enables dict-style access
    conn.execute("PRAGMA journal_mode=WAL")  # better concurrent write safety
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create all tables if they do not already exist."""
    conn = get_connection()
    try:
        with conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename    TEXT    NOT NULL,
                    original_name TEXT  NOT NULL,
                    file_size   INTEGER NOT NULL,
                    page_count  INTEGER DEFAULT 0,
                    chunk_count INTEGER DEFAULT 0,
                    status      TEXT    DEFAULT 'processing',
                    error_msg   TEXT,
                    session_id  TEXT    NOT NULL,
                    uploaded_at TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content     TEXT    NOT NULL,
                    page_number INTEGER NOT NULL,
                    session_id  TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT    NOT NULL,
                    role        TEXT    NOT NULL CHECK(role IN ('user', 'assistant')),
                    content     TEXT    NOT NULL,
                    sources     TEXT,           -- JSON-encoded source list
                    created_at  TEXT    NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_docs_session   ON documents(session_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_doc     ON chunks(document_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id);
                CREATE INDEX IF NOT EXISTS idx_chat_session   ON chat_history(session_id);
                """
            )
        logger.info("Database initialised at %s", Config.DATABASE_PATH)
    finally:
        conn.close()


# ── Document operations ──────────────────────────────────────────────────────

def insert_document(
    filename: str,
    original_name: str,
    file_size: int,
    session_id: str,
) -> int:
    """Insert a new document record and return its id."""
    conn = get_connection()
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO documents (filename, original_name, file_size,
                                       status, session_id, uploaded_at)
                VALUES (?, ?, ?, 'processing', ?, ?)
                """,
                (filename, original_name, file_size, session_id,
                 datetime.utcnow().isoformat()),
            )
            return cursor.lastrowid
    finally:
        conn.close()


def update_document_status(
    doc_id: int,
    status: str,
    page_count: int = 0,
    chunk_count: int = 0,
    error_msg: Optional[str] = None,
) -> None:
    """Update processing status and stats for a document."""
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                """
                UPDATE documents
                SET status = ?, page_count = ?, chunk_count = ?, error_msg = ?
                WHERE id = ?
                """,
                (status, page_count, chunk_count, error_msg, doc_id),
            )
    finally:
        conn.close()


def get_documents_by_session(session_id: str) -> list[dict]:
    """Return all documents belonging to a session."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM documents WHERE session_id = ? ORDER BY uploaded_at DESC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_document_by_id(doc_id: int, session_id: str) -> Optional[dict]:
    """Return a single document; enforces session ownership."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ? AND session_id = ?",
            (doc_id, session_id),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def delete_document(doc_id: int, session_id: str) -> bool:
    """Delete a document (and its chunks via CASCADE). Returns True on success."""
    conn = get_connection()
    try:
        with conn:
            result = conn.execute(
                "DELETE FROM documents WHERE id = ? AND session_id = ?",
                (doc_id, session_id),
            )
            return result.rowcount > 0
    finally:
        conn.close()


# ── Chunk operations ─────────────────────────────────────────────────────────

def insert_chunks(chunks: list[dict]) -> None:
    """Bulk-insert chunk records.

    Each dict must have: document_id, chunk_index, content, page_number, session_id
    """
    conn = get_connection()
    try:
        with conn:
            conn.executemany(
                """
                INSERT INTO chunks (document_id, chunk_index, content,
                                    page_number, session_id)
                VALUES (:document_id, :chunk_index, :content,
                        :page_number, :session_id)
                """,
                chunks,
            )
    finally:
        conn.close()


def get_chunks_by_session(session_id: str) -> list[dict]:
    """Return all chunks for a session (used for rebuilding FAISS index)."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT c.*, d.original_name
            FROM   chunks c
            JOIN   documents d ON d.id = c.document_id
            WHERE  c.session_id = ?
              AND  d.status = 'ready'
            ORDER  BY c.document_id, c.chunk_index
            """,
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── Chat history operations ──────────────────────────────────────────────────

def insert_message(
    session_id: str,
    role: str,
    content: str,
    sources: Optional[str] = None,
) -> int:
    """Persist a single chat message. Returns the new row id."""
    conn = get_connection()
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO chat_history (session_id, role, content, sources, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, role, content, sources, datetime.utcnow().isoformat()),
            )
            return cursor.lastrowid
    finally:
        conn.close()


def get_chat_history(session_id: str, limit: int = 50) -> list[dict]:
    """Return the last `limit` messages for a session, oldest first."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT * FROM chat_history
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        return list(reversed([dict(r) for r in rows]))
    finally:
        conn.close()


def clear_chat_history(session_id: str) -> None:
    """Delete all messages for a session."""
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "DELETE FROM chat_history WHERE session_id = ?",
                (session_id,),
            )
    finally:
        conn.close()
