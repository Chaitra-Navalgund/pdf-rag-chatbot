"""
app.py
------
Flask application entry point.

Routes
------
GET  /                     → Serve the single-page chat UI
GET  /health               → Health-check endpoint
POST /api/upload           → Upload and process a PDF
GET  /api/documents        → List uploaded documents for this session
DELETE /api/documents/<id> → Remove a document
POST /api/chat             → Ask a question (RAG pipeline)
GET  /api/chat/history     → Retrieve chat history
DELETE /api/chat/history   → Clear chat history
"""

import json
import logging
import os
import uuid
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    session,
)
from werkzeug.utils import secure_filename

from config import Config
from database.db import (
    clear_chat_history,
    delete_document,
    get_chat_history,
    get_chunks_by_session,
    get_document_by_id,
    get_documents_by_session,
    init_db,
    insert_chunks,
    insert_document,
    insert_message,
    update_document_status,
)
from services.pdf_processor import (
    PDFValidationError,
    allowed_file,
    generate_safe_filename,
    process_pdf,
    is_scanned_pdf,
    extract_text_from_pdf,
)
from services.rag_service import answer_question
from services.vector_store import vector_store

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── App setup ────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = Config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = Config.UPLOAD_FOLDER

# Ensure upload directory exists
Path(Config.UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

# Print config warnings at startup
for warning in Config.validate():
    logger.warning("CONFIG: %s", warning)

# Initialise database
init_db()


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_session_id() -> str:
    """Return (or create) a stable session identifier for the current user."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]


def _rebuild_index_for_session(session_id: str) -> int:
    """Load all ready chunks from DB and rebuild the FAISS index."""
    chunks = get_chunks_by_session(session_id)
    if chunks:
        return vector_store.build_index(session_id, chunks)
    return 0


def _ensure_index(session_id: str) -> None:
    """Build the FAISS index if it is not already in memory."""
    if not vector_store.session_exists(session_id):
        _rebuild_index_for_session(session_id)


def error_response(message: str, status_code: int = 400) -> tuple:
    return jsonify({"success": False, "error": message}), status_code


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main SPA."""
    get_session_id()  # initialise session
    return render_template("index.html")


@app.route("/health")
def health():
    """Health-check endpoint — returns 200 if the app is running."""
    return jsonify({
        "status": "ok",
        "llm_provider": Config.LLM_PROVIDER,
        "llm_model": Config.LLM_MODEL,
        "gemini_key_configured": bool(Config.GEMINI_API_KEY),
        "openai_key_configured": bool(Config.OPENAI_API_KEY),
        "embedding_model": Config.EMBEDDING_MODEL,
    })


# ── PDF Upload ────────────────────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload_pdf():
    """
    Upload a PDF, extract text, chunk it, generate embeddings,
    and add to the FAISS index.
    """
    session_id = get_session_id()

    if "file" not in request.files:
        return error_response("No file part in the request.")

    file = request.files["file"]

    if file.filename == "":
        return error_response("No file selected.")

    if not allowed_file(file.filename):
        return error_response(
            "Invalid file type. Only PDF files are accepted."
        )

    # Generate safe server-side filename
    safe_name = generate_safe_filename(file.filename)
    original_name = secure_filename(file.filename)  # sanitise for storage only
    filepath = os.path.join(Config.UPLOAD_FOLDER, safe_name)

    # Save file to disk
    try:
        file.save(filepath)
    except OSError as exc:
        logger.error("Failed to save uploaded file: %s", exc)
        return error_response("Failed to save the uploaded file.", 500)

    file_size = os.path.getsize(filepath)

    # Insert document record (status = 'processing')
    doc_id = insert_document(safe_name, original_name, file_size, session_id)

    # Process PDF
    try:
        chunks, page_count, scanned = process_pdf(filepath, doc_id, session_id)
    except PDFValidationError as exc:
        # Mark document as failed and return error
        update_document_status(doc_id, "failed", error_msg=str(exc))
        os.remove(filepath)
        return error_response(str(exc))
    except Exception as exc:
        logger.error("Unexpected error processing PDF %s: %s", safe_name, exc)
        update_document_status(doc_id, "failed", error_msg="Internal processing error.")
        os.remove(filepath)
        return error_response("An unexpected error occurred while processing the PDF.", 500)

    if not chunks:
        msg = (
            "No text could be extracted from this PDF. "
            "It may be a scanned document. Please use OCR software first."
        ) if scanned else "No text content found in the PDF."
        update_document_status(doc_id, "failed", page_count=page_count, error_msg=msg)
        return error_response(msg)

    # Attach original document name to chunks for source attribution
    for chunk in chunks:
        chunk["original_name"] = original_name

    # Persist chunks
    insert_chunks(chunks)

    # Update document status
    update_document_status(doc_id, "ready", page_count=page_count, chunk_count=len(chunks))

    # Add to FAISS index
    vector_store.add_chunks(session_id, chunks)

    logger.info(
        "Uploaded '%s': %d pages, %d chunks (session=%s)",
        original_name, page_count, len(chunks), session_id[:8],
    )

    response_data = {
        "success": True,
        "document": {
            "id": doc_id,
            "original_name": original_name,
            "page_count": page_count,
            "chunk_count": len(chunks),
            "file_size": file_size,
            "status": "ready",
        },
    }

    if scanned:
        response_data["warning"] = (
            "This PDF appears to be scanned. Text extraction may be incomplete. "
            "For best results, use an OCR-processed PDF."
        )

    return jsonify(response_data), 201


# ── Document management ───────────────────────────────────────────────────────

@app.route("/api/documents", methods=["GET"])
def list_documents():
    """Return all documents for the current session."""
    session_id = get_session_id()
    docs = get_documents_by_session(session_id)
    return jsonify({"success": True, "documents": docs})


@app.route("/api/documents/<int:doc_id>", methods=["DELETE"])
def remove_document(doc_id: int):
    """Remove a document and rebuild the FAISS index."""
    session_id = get_session_id()

    doc = get_document_by_id(doc_id, session_id)
    if not doc:
        return error_response("Document not found.", 404)

    # Delete from DB (chunks cascade)
    deleted = delete_document(doc_id, session_id)
    if not deleted:
        return error_response("Failed to delete document.", 500)

    # Remove file from disk (best-effort)
    filepath = os.path.join(Config.UPLOAD_FOLDER, doc["filename"])
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except OSError as exc:
        logger.warning("Could not remove file %s: %s", filepath, exc)

    # Rebuild FAISS index without the deleted document's chunks
    _rebuild_index_for_session(session_id)

    return jsonify({"success": True, "message": "Document removed."})


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Accept a question, run the RAG pipeline, and return the answer.

    Request body (JSON):
        {"question": "What is the main topic of the document?"}
    """
    session_id = get_session_id()

    data = request.get_json(silent=True)
    if not data or "question" not in data:
        return error_response("Request body must contain a 'question' field.")

    question = str(data["question"]).strip()
    if len(question) > 2000:
        return error_response("Question is too long (max 2000 characters).")

    # Ensure FAISS index is loaded for this session
    _ensure_index(session_id)

    # Persist user message
    insert_message(session_id, "user", question)

    # Run RAG pipeline
    rag_response = answer_question(session_id, question)

    # Persist assistant message
    sources_json = json.dumps(rag_response.sources) if rag_response.sources else None
    insert_message(session_id, "assistant", rag_response.answer, sources=sources_json)

    return jsonify({
        "success": True,
        **rag_response.to_dict(),
    })


@app.route("/api/chat/history", methods=["GET"])
def chat_history():
    """Return the chat history for the current session."""
    session_id = get_session_id()
    history = get_chat_history(session_id)

    # Deserialise sources JSON for each assistant message
    for msg in history:
        if msg.get("sources"):
            try:
                msg["sources"] = json.loads(msg["sources"])
            except (json.JSONDecodeError, TypeError):
                msg["sources"] = []

    return jsonify({"success": True, "history": history})


@app.route("/api/chat/history", methods=["DELETE"])
def clear_history():
    """Clear conversation history for the current session."""
    session_id = get_session_id()
    clear_chat_history(session_id)
    return jsonify({"success": True, "message": "Chat history cleared."})


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(413)
def request_entity_too_large(error):
    max_mb = Config.MAX_CONTENT_LENGTH // (1024 * 1024)
    return error_response(f"File too large. Maximum allowed size is {max_mb} MB.", 413)


@app.errorhandler(404)
def not_found(error):
    return error_response("The requested resource was not found.", 404)


@app.errorhandler(500)
def internal_error(error):
    logger.error("Internal server error: %s", error)
    return error_response("An internal server error occurred.", 500)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=Config.DEBUG,
    )
