"""Qdrant vector storage (optional)."""

import re

from ocr_agent.models import Document, Chunk, Page


class QdrantStorage:
    """Qdrant storage for vector search."""

    def __init__(self, url: str = "http://localhost:6333"):
        self.url = url

    def store(self, document: Document) -> None:
        """Store document in Qdrant."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import PointStruct, VectorParams, Distance
        except ImportError:
            raise ImportError("Qdrant requires: pip install ocr-agent[qdrant]") from None

        client = QdrantClient(url=self.url)
        collection = f"doc_{document.document_id}"
        # Use 384-dim placeholder if no embeddings
        dim = 384
        if document.chunks and document.chunks[0].embedding:
            dim = len(document.chunks[0].embedding)
        client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        points = []
        for i, c in enumerate(document.chunks):
            vec = c.embedding or [0.0] * dim
            points.append(
                PointStruct(
                    id=i,
                    vector=vec,
                    payload={
                        "chunk_id": c.chunk_id,
                        "content": c.content,
                        "page_range": list(c.page_range),
                        "document_id": document.document_id,
                        "metadata": document.metadata,
                        "full_text": document.full_text,
                    },
                )
            )
        # Metadata point (id -1 stored as max uint to avoid conflict)
        points.append(
            PointStruct(
                id=2**63 - 1,
                vector=[0.0] * dim,
                payload={
                    "document_id": document.document_id,
                    "metadata": document.metadata,
                    "full_text": document.full_text,
                    "pages": [
                        {"page_num": p.page_num, "text": p.text, "tables": p.tables}
                        for p in document.pages
                    ],
                    "is_metadata": True,
                },
            )
        )
        if points:
            client.upsert(collection_name=collection, points=points)

    def get(self, document_id: str) -> Document | None:
        """Retrieve document from Qdrant (metadata only; chunks from scroll)."""
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            return None

        client = QdrantClient(url=self.url)
        collection = f"doc_{document_id}"
        try:
            result, _ = client.scroll(collection_name=collection, limit=1000)
        except Exception:
            return None
        if not result:
            return None
        metadata: dict = {}
        full_text = ""
        stored_pages: list[dict] = []
        chunks: list = []
        for p in result:
            payload = p.payload or {}
            if payload.get("is_metadata"):
                metadata = payload.get("metadata", {})
                full_text = payload.get("full_text", "")
                stored_pages = payload.get("pages", [])
            elif payload.get("chunk_id"):
                chunks.append(
                    Chunk(
                        chunk_id=payload["chunk_id"],
                        content=payload["content"],
                        page_range=tuple(payload.get("page_range", [1, 1])),
                        metadata=payload,
                    )
                )
                if not metadata:
                    metadata = payload.get("metadata", {})
                    full_text = payload.get("full_text", "")

        if stored_pages:
            pages = [
                Page(page_num=p["page_num"], text=p.get("text", ""), tables=p.get("tables", []))
                for p in sorted(stored_pages, key=lambda x: x["page_num"])
            ]
        else:
            # Fallback: parse full_text (--- Page N ---\n...)
            max_page = max((c.page_range[1] for c in chunks), default=1) if chunks else 1
            pages = [Page(page_num=i, text="", tables=[]) for i in range(1, max_page + 1)]
            if full_text:
                for m in re.finditer(r"--- Page (\d+) ---\s*\n([\s\S]*?)(?=--- Page \d+ ---|$)", full_text):
                    page_num = int(m.group(1))
                    text = m.group(2).strip()
                    for p in pages:
                        if p.page_num == page_num:
                            p.text = text
                            break
        return Document(
            document_id=document_id,
            metadata=metadata,
            pages=pages,
            chunks=chunks,
            full_text=full_text,
        )

    def search(
        self,
        document_id: str,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> list[dict]:
        """Vector search (requires query embedding - simplified keyword fallback)."""
        doc = self.get(document_id)
        if not doc:
            return []
        query_lower = query.lower()
        scored = []
        for c in doc.chunks:
            if query_lower in c.content.lower():
                scored.append({
                    "chunk_id": c.chunk_id,
                    "content": c.content,
                    "score": 1.0,
                    "page_range": list(c.page_range),
                })
        return scored[:top_k]

    def delete(self, document_id: str) -> bool:
        """Delete collection."""
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            return False
        client = QdrantClient(url=self.url)
        collection = f"doc_{document_id}"
        try:
            client.delete_collection(collection_name=collection)
            return True
        except Exception:
            return False

    def list_documents(self) -> list[str]:
        """List all document IDs from Qdrant collections."""
        return [d["document_id"] for d in self.list_documents_with_metadata()]

    def list_documents_with_metadata(self) -> list[dict]:
        """List documents; uploaded_at is None (Qdrant has no created_at)."""
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            return []
        client = QdrantClient(url=self.url)
        try:
            collections = client.get_collections().collections
        except Exception:
            return []
        prefix = "doc_"
        return [
            {"document_id": c.name[len(prefix):], "uploaded_at": None}
            for c in sorted(collections, key=lambda c: c.name, reverse=True)
            if c.name.startswith(prefix)
        ]
