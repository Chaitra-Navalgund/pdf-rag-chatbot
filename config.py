"""
config.py
---------
Centralised configuration for the PDF RAG Chatbot.
All settings are loaded from environment variables (see .env.example).
"""

import os
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


class Config:
    # ── Flask ──────────────────────────────────────────────
    SECRET_KEY: str = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")
    DEBUG: bool = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    ENV: str = os.getenv("FLASK_ENV", "development")

    # ── File Upload ────────────────────────────────────────
    UPLOAD_FOLDER: str = os.getenv("UPLOAD_FOLDER", "uploads")
    MAX_CONTENT_LENGTH: int = int(os.getenv("MAX_CONTENT_LENGTH", 52_428_800))  # 50 MB
    ALLOWED_EXTENSIONS: set = set(
        ext.strip() for ext in os.getenv("ALLOWED_EXTENSIONS", "pdf").split(",")
    )

    # ── Database ───────────────────────────────────────────
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "database/chatbot.db")

    # ── LLM Provider ──────────────────────────────────────
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

    # ── Embedding Model ────────────────────────────────────
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ── RAG Pipeline ──────────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", 5))

    @classmethod
    def validate(cls) -> list[str]:
        """Return a list of configuration warnings (not errors)."""
        warnings = []
        if cls.LLM_PROVIDER == "gemini" and not cls.GEMINI_API_KEY:
            warnings.append(
                "GEMINI_API_KEY is not set. LLM responses will be unavailable. "
                "Get a free key at https://aistudio.google.com/app/apikey"
            )
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            warnings.append("OPENAI_API_KEY is not set. LLM responses will be unavailable.")
        return warnings
