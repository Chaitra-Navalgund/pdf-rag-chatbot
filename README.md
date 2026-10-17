# DocuMind AI — PDF RAG Chatbot

> **MCA Academic Project** | AI-Powered PDF Question Answering using Retrieval-Augmented Generation (RAG)

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask)](https://flask.palletsprojects.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0078D7)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Abstract

DocuMind AI is a web application that enables users to upload PDF documents and query their content using natural language. The system implements a **Retrieval-Augmented Generation (RAG)** pipeline: it extracts and chunks document text, generates semantic embeddings, stores them in a FAISS vector index, and passes the most relevant excerpts as context to a Large Language Model (Gemini / OpenAI GPT) to produce accurate, grounded answers with page-level source citations.

---

## 🚨 Problem Statement

Knowledge workers, students, and researchers frequently need to extract specific information from lengthy PDF documents. Traditional keyword search is brittle and fails to understand semantic intent. Reading entire documents to find one answer is inefficient. Existing general-purpose chatbots hallucinate facts because they have no access to the user's actual documents.

---

## 🎯 Project Objectives

1. Provide a drag-and-drop PDF upload interface with validation and progress feedback.
2. Automatically extract and index document text for semantic search.
3. Answer user questions in natural language using only information present in the uploaded documents.
4. Display page-level source citations alongside every answer.
5. Support multiple documents within a single session.
6. Maintain conversation history for context continuity.
7. Deliver a professional, mobile-responsive web interface.

---

## 🔄 Existing vs Proposed System

| Aspect | Existing Systems | DocuMind AI (Proposed) |
|--------|-----------------|------------------------|
| Search method | Keyword matching | Dense semantic vector search |
| Answer quality | Retrieves passages verbatim | Synthesises a coherent answer |
| Hallucination risk | N/A (no generation) | Mitigated by strict context grounding |
| Source tracing | Limited | Page-level citations |
| Multi-document | Rarely | Yes |
| Cost | Often paid enterprise tools | Free tier (Gemini API) available |

---

## ⚙️ Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-1 | Users shall upload PDF files up to 50 MB via drag-and-drop or file picker. |
| FR-2 | System shall validate file type, size, encryption, and emptiness before processing. |
| FR-3 | System shall extract text from all PDF pages using PyMuPDF. |
| FR-4 | Extracted text shall be split into overlapping chunks (configurable size/overlap). |
| FR-5 | Each chunk shall be embedded using Sentence Transformers and indexed in FAISS. |
| FR-6 | Users shall submit natural-language questions via a chat interface. |
| FR-7 | System shall retrieve the top-k most similar chunks for each question. |
| FR-8 | Retrieved chunks plus the question shall be passed to an LLM for answer generation. |
| FR-9 | Answers shall include document name and page numbers as source citations. |
| FR-10 | Users shall be able to delete uploaded documents and clear chat history. |

## 🔒 Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-1 | API keys shall never appear in frontend code or logs. |
| NFR-2 | Uploaded files shall be assigned server-generated safe filenames. |
| NFR-3 | Each user session shall be isolated; users cannot access each other's documents. |
| NFR-4 | The system shall respond to a chat query within 15 seconds under normal load. |
| NFR-5 | The UI shall be responsive across desktop, tablet, and mobile viewports. |
| NFR-6 | The system shall handle corrupted, encrypted, and scanned PDFs gracefully. |

---

## 🛠️ Technology Stack

### Software Requirements

| Layer | Technology | Version |
|-------|-----------|---------|
| Backend framework | Flask | 3.0.3 |
| PDF extraction | PyMuPDF (fitz) | 1.24.5 |
| Embeddings | Sentence Transformers | 3.0.1 |
| Vector search | FAISS-CPU | 1.8.0 |
| LLM (primary) | Google Gemini | 1.5 Flash |
| LLM (optional) | OpenAI GPT | 3.5 Turbo |
| Database | SQLite 3 | built-in |
| Runtime | Python | 3.12 |

### Hardware Requirements (Minimum)

| Component | Minimum |
|-----------|---------|
| RAM | 4 GB (8 GB recommended) |
| CPU | Dual-core 2 GHz |
| Storage | 2 GB free space |
| Internet | Required for LLM API calls |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Browser (SPA)                           │
│   Sidebar        Chat Area        Input Area       Toasts       │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP/REST
┌────────────────────────▼────────────────────────────────────────┐
│                      Flask Application (app.py)                 │
│                                                                  │
│   /api/upload   /api/documents   /api/chat   /api/chat/history  │
└───┬──────────────────────┬──────────────┬────────────────────────┘
    │                      │              │
    ▼                      ▼              ▼
┌──────────┐        ┌──────────────┐  ┌──────────────────────────┐
│   PDF    │        │   Database   │  │      RAG Service          │
│ Processor│        │   (SQLite)   │  │                          │
│          │        │              │  │  1. Query Embedding       │
│ Validate │        │  documents   │  │  2. FAISS Search          │
│ Extract  │        │  chunks      │  │  3. Build Prompt          │
│ Chunk    │        │  chat_hist   │  │  4. LLM Call              │
└────┬─────┘        └──────────────┘  │  5. Return Answer         │
     │                                └──────────┬───────────────┘
     ▼                                            │
┌────────────────┐                    ┌───────────▼──────────────┐
│ Embedding      │◄───────────────────│   Vector Store (FAISS)   │
│ Service        │                    │   Per-session index       │
│ (SentenceXfmr) │                    └──────────────────────────┘
└────────────────┘                              │
                                    ┌───────────▼──────────────┐
                                    │   LLM Provider           │
                                    │   Gemini 1.5 Flash       │
                                    │   (or OpenAI GPT)        │
                                    └──────────────────────────┘
```

---

## 📦 Module Descriptions

| Module | Purpose |
|--------|---------|
| `app.py` | Flask routes, session management, orchestration |
| `config.py` | Centralised environment-variable config |
| `database/db.py` | SQLite CRUD: documents, chunks, chat history |
| `services/pdf_processor.py` | PDF validation, text extraction, chunking |
| `services/embedding_service.py` | Sentence-Transformer embeddings (singleton) |
| `services/vector_store.py` | Per-session FAISS index management |
| `services/rag_service.py` | Full RAG pipeline: retrieve → prompt → LLM |
| `templates/index.html` | Single-page application HTML |
| `static/css/style.css` | Premium dark-mode design system |
| `static/js/app.js` | Frontend logic: upload, chat, UI state |

---

## 🔁 RAG Workflow

```
User uploads PDF
      │
      ▼
[pdf_processor] Validate → Extract pages → Chunk with overlap
      │
      ▼
[embedding_service] embed_texts(chunks) → float32 numpy array
      │
      ▼
[vector_store] faiss.IndexFlatIP.add(embeddings)  ← stored per session
      │
User types question
      │
      ▼
[embedding_service] embed_query(question) → (1, D) array
      │
      ▼
[vector_store] index.search(query_emb, top_k) → ranked chunks
      │
      ▼
[rag_service] Build grounded prompt (system rules + excerpts + question)
      │
      ▼
[LLM API] Generate answer (Gemini / OpenAI)
      │
      ▼
Return answer + source citations → displayed in chat UI
```

---

## 📁 Project Structure

```
pdf-rag-chatbot/
├── app.py                  # Flask application entry point
├── config.py               # Centralised configuration
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .gitignore
├── README.md
│
├── services/
│   ├── __init__.py
│   ├── pdf_processor.py    # Upload validation, extraction, chunking
│   ├── embedding_service.py# Sentence Transformers singleton
│   ├── vector_store.py     # FAISS per-session index
│   └── rag_service.py      # RAG orchestration + LLM call
│
├── database/
│   ├── __init__.py
│   └── db.py               # SQLite operations
│
├── templates/
│   └── index.html          # SPA template
│
├── static/
│   ├── css/style.css       # Design system
│   ├── js/app.js           # Frontend logic
│   └── images/             # (reserved for assets)
│
├── tests/
│   ├── __init__.py
│   ├── test_pdf_processor.py
│   ├── test_vector_store.py
│   └── test_app.py
│
└── uploads/                # Uploaded PDFs (git-ignored)
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- Git

### Step 1 — Clone the repository

```bash
git clone https://github.com/Chaitra-Navalgund/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

### Step 2 — Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ The first run will download the Sentence Transformer model (~90 MB). Ensure you have an internet connection.

### Step 4 — Configure environment variables

```bash
copy .env.example .env      # Windows
# or
cp .env.example .env        # macOS/Linux
```

Open `.env` and set your API key:

```env
# Get a FREE Gemini key at: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your-actual-key-here
```

### Step 5 — Run the application

```bash
python app.py
```

Open your browser at **http://localhost:5000**

---

## 🔑 LLM API Setup

### Option A: Google Gemini (Recommended — Free Tier Available)

1. Visit https://aistudio.google.com/app/apikey
2. Sign in with a Google account
3. Click **Create API key**
4. Copy the key into `.env` → `GEMINI_API_KEY`
5. Set `LLM_PROVIDER=gemini` and `LLM_MODEL=gemini-1.5-flash`

### Option B: OpenAI GPT (Paid)

1. Visit https://platform.openai.com/api-keys
2. Create and copy your key
3. Set `OPENAI_API_KEY` in `.env`
4. Set `LLM_PROVIDER=openai` and `LLM_MODEL=gpt-3.5-turbo`

### No API Key

The application works without a key. Instead of an LLM-generated answer, it returns the raw retrieved document excerpts.

---

## 🧪 Running Tests

```bash
# Activate virtual environment first
pytest tests/ -v
```

Expected test count: ~35 tests across three modules.

> **Note**: Tests that require the Sentence Transformer model (`test_vector_store.py`) will download it on first run. Tests use mocks wherever possible to avoid real API calls.

---

## 🩺 Health Check

```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "ok",
  "llm_provider": "gemini",
  "llm_model": "gemini-1.5-flash",
  "gemini_key_configured": true,
  "openai_key_configured": false,
  "embedding_model": "all-MiniLM-L6-v2"
}
```

---

## ⚠️ Limitations (Academic Prototype)

1. **In-memory FAISS index** — lost on server restart. Documents must be re-uploaded or the index rebuilt from the database.
2. **Single-user deployment** — no authentication; all sessions share one server process.
3. **No OCR** — scanned PDFs without embedded text cannot be processed.
4. **LLM accuracy** — answers are grounded in retrieved context but may still contain errors. The disclaimer is shown in every answer.
5. **SQLite** — suitable for development; not recommended for high-concurrency production.

---

## 🔮 Future Enhancements

- [ ] OCR integration (Tesseract/EasyOCR) for scanned PDFs
- [ ] Persistent FAISS index (disk storage with faiss.write_index)
- [ ] User authentication and multi-user support
- [ ] Chat memory for multi-turn contextual answers
- [ ] Support for DOCX, TXT, and web URLs
- [ ] Re-ranking step (cross-encoder) for improved retrieval precision
- [ ] Docker containerisation for easy deployment
- [ ] Citation highlighting — scroll directly to the cited PDF page

---

## 📊 Database Schema

```sql
-- PDF documents uploaded by users
CREATE TABLE documents (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    filename      TEXT NOT NULL,       -- safe server-side name
    original_name TEXT NOT NULL,       -- user-provided name
    file_size     INTEGER NOT NULL,
    page_count    INTEGER DEFAULT 0,
    chunk_count   INTEGER DEFAULT 0,
    status        TEXT DEFAULT 'processing',
    error_msg     TEXT,
    session_id    TEXT NOT NULL,
    uploaded_at   TEXT NOT NULL        -- ISO 8601 UTC
);

-- Text chunks extracted from documents
CREATE TABLE chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,
    content       TEXT NOT NULL,
    page_number   INTEGER NOT NULL,
    session_id    TEXT NOT NULL
);

-- Conversation history
CREATE TABLE chat_history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    role          TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content       TEXT NOT NULL,
    sources       TEXT,               -- JSON-encoded source list
    created_at    TEXT NOT NULL
);
```

---

## 🎓 Viva Questions & Answers

**Q1: What is RAG and why is it used?**
RAG (Retrieval-Augmented Generation) combines information retrieval with LLM generation. Instead of relying solely on the model's training data, RAG retrieves relevant passages from the user's documents and injects them into the prompt, enabling accurate, grounded answers and reducing hallucination.

**Q2: What is FAISS and why is it chosen?**
FAISS (Facebook AI Similarity Search) is an open-source library for efficient vector similarity search. It is chosen for its high speed on CPU (IndexFlatIP), ease of integration with NumPy, and zero licensing cost — ideal for academic projects.

**Q3: How does cosine similarity work in this system?**
Sentence Transformer embeddings are L2-normalised before being stored in FAISS's IndexFlatIP (Inner Product) index. For unit-normalised vectors, the inner product equals the cosine similarity. Higher scores indicate greater semantic similarity between the query and document chunk.

**Q4: How does chunking with overlap work?**
The text of each PDF page is split into fixed-size windows. Consecutive windows overlap by a configurable number of characters. This prevents context loss at chunk boundaries — a sentence split across chunks can still be retrieved as part of either chunk.

**Q5: How is security implemented?**
- API keys stored in environment variables, never in code
- Uploaded files assigned UUID-based server-side names
- File type validated both by extension and MIME content
- Session-scoped ownership checks prevent cross-user document access
- Input length limits on questions (2000 characters)
- HTML escaping in the frontend to prevent XSS

**Q6: What happens if the PDF is scanned?**
PyMuPDF extracts minimal text from scanned PDFs. The system detects this (average < 50 chars/page) and returns a warning message advising the user to run OCR (e.g., Tesseract) before uploading.

**Q7: What is the embedding model used?**
`all-MiniLM-L6-v2` from the Sentence Transformers library. It produces 384-dimensional embeddings, balancing quality and inference speed, and is free to use.

**Q8: How is hallucination prevented?**
The system prompt explicitly instructs the LLM to answer only from the provided document excerpts. If the relevant information is absent, the model must respond: "I could not find an answer in the uploaded documents."

**Q9: What database is used and why?**
SQLite is used via Python's built-in `sqlite3` module. It requires no separate server process, making setup trivial for an academic prototype. WAL mode provides better concurrent write safety.

**Q10: How would you scale this for production?**
Replace SQLite with PostgreSQL, persist FAISS indexes to disk, add Redis caching, containerise with Docker, add authentication (JWT/OAuth), and deploy behind a reverse proxy (Nginx) with multiple worker processes.

---

## 👩‍💻 Author

**Chaitra Navalgund**  
MCA Student | GitHub: [Chaitra-Navalgund](https://github.com/Chaitra-Navalgund)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
