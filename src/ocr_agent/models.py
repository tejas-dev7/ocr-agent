"""Data models for OCR Agent."""

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID


@dataclass
class Page:
    """Single page content from a document."""

    page_num: int
    text: str
    tables: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Chunk:
    """Text chunk for storage and retrieval."""

    chunk_id: str
    content: str
    page_range: tuple[int, int]
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """Processed document with pages and chunks."""

    document_id: str
    metadata: dict[str, Any]
    pages: list[Page]
    chunks: list[Chunk]
    full_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "document_id": self.document_id,
            "metadata": self.metadata,
            "pages": [
                {
                    "page_num": p.page_num,
                    "text": p.text,
                    "tables": p.tables,
                }
                for p in self.pages
            ],
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "content": c.content,
                    "page_range": list(c.page_range),
                    "embedding": c.embedding,
                    "metadata": c.metadata,
                }
                for c in self.chunks
            ],
            "full_text": self.full_text,
        }
