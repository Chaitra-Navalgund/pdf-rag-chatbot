"""
services/pdf_processor.py
--------------------------
Handles PDF upload validation, text extraction (via PyMuPDF),
and text chunking with overlap.

Design decisions
----------------
- PyMuPDF (fitz) is used instead of pdfplumber because it is faster and
  handles a wider range of PDF encodings.
- Overlapping chunks preserve context across chunk boundaries, which
  improves retrieval quality.
- Scanned PDFs produce empty or near-empty text; the module detects this
  and advises the user to run OCR instead.
"""

import os
import re
import uuid
import logging
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from config import Config

logger = logging.getLogger(__name__)

# Characters below this per page threshold likely indicate a scanned PDF
_SCANNED_CHARS_PER_PAGE = 50


# ── Validation ───────────────────────────────────────────────────────────────

class PDFValidationError(Exception):
    """Raised when an uploaded file fails validation."""


def allowed_file(filename: str) -> bool:
    """Return True if the file extension is in the allowed set."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    )


def validate_pdf_file(filepath: str) -> None:
    """
    Check that the file at *filepath* is a valid, non-encrypted PDF.
    Raises PDFValidationError with a user-friendly message on failure.
    """
    if not os.path.exists(filepath):
        raise PDFValidationError("Uploaded file not found on server.")

    if os.path.getsize(filepath) == 0:
        raise PDFValidationError("The uploaded PDF file is empty (0 bytes).")

    try:
        doc = fitz.open(filepath)
    except fitz.FileDataError:
        raise PDFValidationError(
            "The file appears to be corrupted and cannot be opened."
        )
    except Exception as exc:
        raise PDFValidationError(f"Could not open PDF: {exc}")

    if doc.is_encrypted:
        doc.close()
        raise PDFValidationError(
            "The PDF is password-protected. Please remove the password before uploading."
        )

    if doc.page_count == 0:
        doc.close()
        raise PDFValidationError("The PDF contains no pages.")

    doc.close()


# ── Safe filename generation ─────────────────────────────────────────────────

def generate_safe_filename(original_name: str) -> str:
    """
    Generate a server-side filename that:
    - Has no path components
    - Contains only alphanumeric characters, hyphens, and underscores
    - Is unique (UUID prefix)
    """
    stem = Path(original_name).stem
    # Keep only safe characters
    safe_stem = re.sub(r"[^\w\-]", "_", stem)[:50]
    return f"{uuid.uuid4().hex}_{safe_stem}.pdf"


# ── Text Extraction ──────────────────────────────────────────────────────────

def extract_text_from_pdf(filepath: str) -> list[dict]:
    """
    Extract text from every page of a PDF.

    Returns a list of dicts, one per page:
        {"page_number": int, "text": str}

    If the extracted text is very sparse (likely a scanned PDF), the
    function still returns the data but logs a warning suggesting OCR.
    """
    doc = fitz.open(filepath)
    pages = []
    total_chars = 0

    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text("text")  # plain text extraction
        text = _clean_text(text)
        total_chars += len(text)
        pages.append({"page_number": page_num + 1, "text": text})

    doc.close()

    avg_chars = total_chars / max(len(pages), 1)
    if avg_chars < _SCANNED_CHARS_PER_PAGE:
        logger.warning(
            "PDF '%s' has very little extractable text (avg %d chars/page). "
            "This is likely a scanned document. Consider OCR with tools like "
            "Tesseract or Adobe Acrobat before uploading.",
            filepath,
            avg_chars,
        )

    return pages


def _clean_text(text: str) -> str:
    """Normalise whitespace in extracted text."""
    # Replace form feeds and multiple blank lines
    text = re.sub(r"\f", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def is_scanned_pdf(pages: list[dict]) -> bool:
    """Return True if the extracted pages look like a scanned document."""
    total = sum(len(p["text"]) for p in pages)
    avg = total / max(len(pages), 1)
    return avg < _SCANNED_CHARS_PER_PAGE


# ── Text Chunking ────────────────────────────────────────────────────────────

def chunk_pages(
    pages: list[dict],
    document_id: int,
    session_id: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[dict]:
    """
    Split page text into overlapping chunks.

    Each chunk dict contains:
        document_id, chunk_index, content, page_number, session_id

    Strategy
    --------
    We process text page by page. If a page's text exceeds chunk_size we
    slide a window over it with the given overlap. Chunks spanning multiple
    pages carry the starting page's number.
    """
    chunk_size = chunk_size or Config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP

    chunks = []
    chunk_index = 0

    for page in pages:
        text = page["text"]
        page_number = page["page_number"]

        if not text:
            continue

        if len(text) <= chunk_size:
            chunks.append(
                _make_chunk(chunk_index, text, page_number, document_id, session_id)
            )
            chunk_index += 1
        else:
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end]
                if chunk_text.strip():
                    chunks.append(
                        _make_chunk(
                            chunk_index, chunk_text, page_number, document_id, session_id
                        )
                    )
                    chunk_index += 1
                start += chunk_size - chunk_overlap

    return chunks


def _make_chunk(
    index: int,
    content: str,
    page_number: int,
    document_id: int,
    session_id: str,
) -> dict:
    return {
        "chunk_index": index,
        "content": content.strip(),
        "page_number": page_number,
        "document_id": document_id,
        "session_id": session_id,
    }


# ── Orchestration helper ─────────────────────────────────────────────────────

def process_pdf(
    filepath: str,
    document_id: int,
    session_id: str,
) -> tuple[list[dict], int, bool]:
    """
    Full pipeline: validate → extract → chunk.

    Returns (chunks, page_count, is_scanned).
    Raises PDFValidationError if the file is invalid.
    """
    validate_pdf_file(filepath)
    pages = extract_text_from_pdf(filepath)
    scanned = is_scanned_pdf(pages)
    chunks = chunk_pages(pages, document_id, session_id)
    return chunks, len(pages), scanned
