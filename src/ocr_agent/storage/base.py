"""Storage provider protocol and factory."""

from typing import Protocol

from ocr_agent.config import OCRConfig, get_config
from ocr_agent.models import Document


class StorageProvider(Protocol):
    """Protocol for document storage backends."""

    def store(self, document: Document) -> None:
        """Store document."""
        ...

    def get(self, document_id: str) -> Document | None:
        """Retrieve document by ID."""
        ...

    def list_documents(self) -> list[str]:
        """List all document IDs. Used to restore document list after page refresh."""
        ...

    def list_documents_with_metadata(self) -> list[dict]:
        """List documents with uploaded_at. Returns [{"document_id": str, "uploaded_at": str|None}]."""
        ...

    def search(
        self,
        document_id: str,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> list[dict]:
        """Search within document. Returns list of {chunk_id, content, score, page_range}."""
        ...

    def delete(self, document_id: str) -> bool:
        """Delete document. Returns True if found and deleted."""
        ...


def get_storage_provider(config: OCRConfig | None = None) -> StorageProvider:
    """Get storage provider based on config."""
    config = config or get_config()
    backend = config.storage_backend

    if backend == "json":
        from ocr_agent.storage.json_file import JSONFileStorage
        return JSONFileStorage(output_dir=config.output_dir)
    if backend == "postgres":
        from ocr_agent.storage.postgres import PostgresStorage
        return PostgresStorage(database_url=config.database_url)
    if backend == "qdrant":
        from ocr_agent.storage.qdrant import QdrantStorage
        return QdrantStorage(url=config.qdrant_url)

    raise ValueError(f"Unknown storage backend: {backend}")
