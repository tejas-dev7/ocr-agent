"""Configuration for OCR Agent.

All settings are loaded from environment variables. Use a .env file in the
project root - copy .env.example to .env and fill in your values.
"""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Load .env from current directory or project root
load_dotenv(Path.cwd() / ".env")
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OCRConfig(BaseSettings):
    """OCR Agent configuration from environment variables (.env file)."""

    model_config = SettingsConfigDict(
        env_prefix="OCR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OCR provider: ollama, tesseract, auto (LLM selects best)
    ocr_provider: Literal["ollama", "tesseract", "auto"] = Field(
        default="ollama",
        description="OCR engine: ollama (minicpm-v4.5), tesseract, auto",
    )

    # Vision model for OCR routing (when ocr_provider=auto)
    llm_router_model: str = Field(
        default="openai/gpt-4o-mini",
        description="Vision LLM for document analysis and OCR model selection",
    )

    # Ollama settings (when ocr_provider=ollama)
    # Accepts OLLAMA_BASE_URL or OCR_OLLAMA_BASE_URL from .env
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL",
        validation_alias=AliasChoices("OLLAMA_BASE_URL", "OCR_OLLAMA_BASE_URL"),
    )
    ollama_vision_model: str = Field(
        default="openbmb/minicpm-v4.5",
        description="Ollama vision model for OCR (e.g. openbmb/minicpm-v4.5, openbmb/minicpm-o4.5, minicpm-v)",
    )
    ollama_timeout: int = Field(
        default=1800,
        description="Ollama API timeout in seconds per page (default 600 for large/complex docs)",
    )

    # LLM settings (LiteLLM) - OPENAI_API_KEY, ANTHROPIC_API_KEY read by LiteLLM from env
    llm_model: str = Field(
        default="openai/gpt-4o-mini",
        description="LLM model (e.g. openai/gpt-4o, anthropic/claude-3-5-sonnet, ollama/llama3)",
    )

    # Storage backend: postgres, qdrant, json
    storage_backend: Literal["postgres", "qdrant", "json"] = Field(
        default="json",
        description="Storage backend for document output",
    )

    # Database (for postgres) - accepts DATABASE_URL or OCR_DATABASE_URL from .env
    database_url: str = Field(
        default="postgresql://localhost/ocr_agent",
        description="PostgreSQL connection URL",
        validation_alias=AliasChoices("DATABASE_URL", "OCR_DATABASE_URL"),
    )

    # Qdrant
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL",
    )

    # Chunking
    chunk_size: int = Field(default=512, description="Tokens per chunk")
    chunk_overlap: int = Field(default=51, description="Overlap between chunks (10% of 512)")
    chunk_strategy: Literal["section", "recursive", "page"] = Field(
        default="recursive",
        description="Chunking strategy",
    )
    pages_per_chunk: int = Field(default=5, description="Pages per chunk (page strategy)")

    # Output
    output_dir: str = Field(
        default="./ocr_output",
        description="Directory for JSON output when storage_backend=json",
    )

    # LLM audit trail - logs all API calls for compliance/debugging
    llm_audit_log_enabled: bool = Field(
        default=True,
        description="Enable audit logging of all LLM API calls",
    )
    llm_audit_log_path: str = Field(
        default="./llm_audit.jsonl",
        description="Path to JSONL audit log file (one JSON object per line)",
    )

    # Logs persistence - store all logs (LLM + OCR) in database
    logs_store_in_db: bool = Field(
        default=True,
        description="Store logs in database (PostgreSQL). Falls back to JSONL file if DB unavailable.",
    )
    logs_database_url: str | None = Field(
        default=None,
        description="Database URL for logs. Uses database_url if not set.",
        validation_alias="OCR_LOGS_DATABASE_URL",
    )


@lru_cache
def get_config() -> OCRConfig:
    """Get cached config instance."""
    return OCRConfig()
