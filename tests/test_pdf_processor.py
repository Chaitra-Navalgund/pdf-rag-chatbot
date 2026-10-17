"""
tests/test_pdf_processor.py
----------------------------
Tests for PDF upload validation, text extraction, and chunking.
"""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from services.pdf_processor import (
    allowed_file,
    generate_safe_filename,
    validate_pdf_file,
    chunk_pages,
    is_scanned_pdf,
    PDFValidationError,
    _clean_text,
)


# ── allowed_file ─────────────────────────────────────────────────────────────

class TestAllowedFile:
    def test_pdf_allowed(self):
        assert allowed_file("report.pdf") is True

    def test_uppercase_pdf_allowed(self):
        assert allowed_file("REPORT.PDF") is True

    def test_txt_rejected(self):
        assert allowed_file("doc.txt") is False

    def test_docx_rejected(self):
        assert allowed_file("doc.docx") is False

    def test_no_extension_rejected(self):
        assert allowed_file("nodotfile") is False

    def test_double_extension_pdf_last(self):
        # security: only last extension matters
        assert allowed_file("evil.exe.pdf") is True

    def test_empty_filename(self):
        assert allowed_file("") is False


# ── generate_safe_filename ────────────────────────────────────────────────────

class TestGenerateSafeFilename:
    def test_produces_pdf_extension(self):
        name = generate_safe_filename("my report.pdf")
        assert name.endswith(".pdf")

    def test_no_path_components(self):
        name = generate_safe_filename("../../etc/passwd.pdf")
        assert "/" not in name
        assert "\\" not in name

    def test_unique_each_call(self):
        a = generate_safe_filename("file.pdf")
        b = generate_safe_filename("file.pdf")
        assert a != b

    def test_only_safe_characters(self):
        import re
        name = generate_safe_filename("file with spaces & <script>.pdf")
        stem = name[:-4]  # strip .pdf
        assert re.fullmatch(r"[a-z0-9_\-]+", stem, re.IGNORECASE)


# ── validate_pdf_file ─────────────────────────────────────────────────────────

class TestValidatePdfFile:
    def test_missing_file_raises(self):
        with pytest.raises(PDFValidationError, match="not found"):
            validate_pdf_file("/nonexistent/path/file.pdf")

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / "empty.pdf"
        f.write_bytes(b"")
        with pytest.raises(PDFValidationError, match="empty"):
            validate_pdf_file(str(f))

    def test_corrupted_file_raises(self, tmp_path):
        f = tmp_path / "bad.pdf"
        f.write_bytes(b"this is not a valid pdf")
        with pytest.raises(PDFValidationError):
            validate_pdf_file(str(f))


# ── chunk_pages ───────────────────────────────────────────────────────────────

class TestChunkPages:
    def _make_pages(self, text, page=1):
        return [{"page_number": page, "text": text}]

    def test_short_text_single_chunk(self):
        pages = self._make_pages("Hello world", page=1)
        chunks = chunk_pages(pages, document_id=1, session_id="s1",
                             chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0]["content"] == "Hello world"
        assert chunks[0]["page_number"] == 1
        assert chunks[0]["document_id"] == 1
        assert chunks[0]["session_id"] == "s1"

    def test_long_text_multiple_chunks(self):
        text = "A" * 2500
        pages = self._make_pages(text)
        chunks = chunk_pages(pages, document_id=1, session_id="s1",
                             chunk_size=1000, chunk_overlap=200)
        # With overlap: starts at 0, 800, 1600, 2400 → 4 chunks
        assert len(chunks) >= 3

    def test_overlap_preserved(self):
        text = "X" * 1000 + "Y" * 1000
        pages = self._make_pages(text)
        chunks = chunk_pages(pages, document_id=1, session_id="s1",
                             chunk_size=800, chunk_overlap=200)
        # Second chunk should overlap with first
        assert len(chunks) >= 2
        # Overlap content from first chunk should appear in second
        assert chunks[1]["content"][:200] == chunks[0]["content"][-200:]

    def test_empty_page_skipped(self):
        pages = [{"page_number": 1, "text": ""}, {"page_number": 2, "text": "Real content"}]
        chunks = chunk_pages(pages, document_id=1, session_id="s1")
        assert all(c["content"].strip() for c in chunks)
        assert len(chunks) == 1

    def test_chunk_index_sequential(self):
        pages = [{"page_number": i, "text": "Word " * 300} for i in range(1, 4)]
        chunks = chunk_pages(pages, document_id=1, session_id="s1",
                             chunk_size=500, chunk_overlap=100)
        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))


# ── is_scanned_pdf ────────────────────────────────────────────────────────────

class TestIsScannedPdf:
    def test_text_rich_pdf_not_scanned(self):
        pages = [{"page_number": i, "text": "A" * 500} for i in range(1, 6)]
        assert is_scanned_pdf(pages) is False

    def test_sparse_text_detected_as_scanned(self):
        pages = [{"page_number": i, "text": "A" * 10} for i in range(1, 6)]
        assert is_scanned_pdf(pages) is True

    def test_empty_pages_scanned(self):
        pages = [{"page_number": 1, "text": ""}]
        assert is_scanned_pdf(pages) is True


# ── _clean_text ───────────────────────────────────────────────────────────────

class TestCleanText:
    def test_removes_extra_blank_lines(self):
        result = _clean_text("Line 1\n\n\n\n\nLine 2")
        assert "\n\n\n" not in result

    def test_strips_leading_trailing_whitespace(self):
        result = _clean_text("  hello world  ")
        assert result == "hello world"

    def test_normalises_tabs(self):
        result = _clean_text("col1\t\tcol2")
        assert "\t" not in result
