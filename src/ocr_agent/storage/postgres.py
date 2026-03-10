"""PostgreSQL + pgvector storage (optional)."""

import re

from ocr_agent.models import Document, Page, Chunk


class PostgresStorage:
    """PostgreSQL storage with pgvector for embeddings."""

    def __init__(self, database_url: str):
        self.database_url = database_url

    def store(self, document: Document) -> None:
        """Store document in PostgreSQL."""
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError:
            raise ImportError(
                "PostgreSQL storage requires: pip install ocr-agent[postgres]"
            ) from None

        with psycopg.connect(self.database_url) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        document_id TEXT PRIMARY KEY,
                        metadata JSONB,
                        full_text TEXT,
                        pages JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS pages JSONB"
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id TEXT PRIMARY KEY,
                        document_id TEXT REFERENCES documents(document_id),
                        content TEXT,
                        page_start INT,
                        page_end INT,
                        embedding vector(1536),
                        metadata JSONB
                    )
                    """
                )
                pages_json = [{"page_num": p.page_num, "text": p.text, "tables": p.tables} for p in document.pages]
                cur.execute(
                    """
                    INSERT INTO documents (document_id, metadata, full_text, pages)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (document_id) DO UPDATE SET
                        metadata = EXCLUDED.metadata,
                        full_text = EXCLUDED.full_text,
                        pages = EXCLUDED.pages
                    """,
                    (
                        document.document_id,
                        psycopg.types.json.Jsonb(document.metadata),
                        document.full_text,
                        psycopg.types.json.Jsonb(pages_json),
                    ),
                )
                for c in document.chunks:
                    cur.execute(
                        """
                        INSERT INTO chunks (chunk_id, document_id, content, page_start, page_end, embedding, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (chunk_id) DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding
                        """,
                        (
                            c.chunk_id,
                            document.document_id,
                            c.content,
                            c.page_range[0],
                            c.page_range[1],
                            c.embedding,
                            psycopg.types.json.Jsonb(c.metadata or {}),
                        ),
                    )
            conn.commit()

    def store_partial(
        self,
        document_id: str,
        pages: list[Page],
        metadata: dict,
        total_pages: int,
    ) -> None:
        """Save partial document after each page so progress survives restarts."""
        full_text = "\n\n".join(
            f"--- Page {p.page_num} ---\n{p.text}" for p in pages if p.text.strip()
        )
        meta = {**metadata, "total_pages": total_pages, "processed_pages": len([p for p in pages if p.text.strip()])}
        doc = Document(
            document_id=document_id,
            metadata=meta,
            pages=pages,
            chunks=[],
            full_text=full_text,
        )
        self.store(doc)

    def get(self, document_id: str) -> Document | None:
        """Retrieve document from PostgreSQL."""
        try:
            import psycopg
        except ImportError:
            raise ImportError("pip install ocr-agent[postgres]") from None

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS pages JSONB")
                except Exception:
                    pass
                cur.execute(
                    "SELECT metadata, full_text, pages FROM documents WHERE document_id = %s",
                    (document_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                metadata, full_text, pages_data = row
                cur.execute(
                    "SELECT chunk_id, content, page_start, page_end, embedding, metadata FROM chunks WHERE document_id = %s ORDER BY page_start",
                    (document_id,),
                )
                rows = cur.fetchall()
            chunks = [
                Chunk(
                    chunk_id=r[0],
                    content=r[1],
                    page_range=(r[2], r[3]),
                    embedding=list(r[4]) if r[4] else None,
                    metadata=r[5] or {},
                )
                for r in rows
            ]
            if pages_data and len(pages_data) > 0:
                pages = [
                    Page(page_num=p["page_num"], text=p.get("text", ""), tables=p.get("tables", []))
                    for p in pages_data
                ]
            else:
                pages = [Page(page_num=i, text="", tables=[]) for i in range(1, max((c.page_range[1] for c in chunks), default=1) + 1)]
            return Document(
                document_id=document_id,
                metadata=metadata or {},
                pages=pages,
                chunks=chunks,
                full_text=full_text or "",
            )

    def search(
        self,
        document_id: str,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> list[dict]:
        """Keyword search (pgvector similarity if embeddings present)."""
        doc = self.get(document_id)
        if not doc:
            return []
        query_lower = query.lower()
        page_matches = re.findall(r"\bpage\s+(\d+)\b", query_lower) + re.findall(
            r"\bp\.?\s*(\d+)\b", query_lower
        )
        query_pages = {int(p) for p in page_matches}
        scored = []
        for c in doc.chunks:
            score = 0.0
            if query_lower in c.content.lower():
                score = 1.0
            if query_pages and c.page_range:
                lo, hi = c.page_range[0], c.page_range[1]
                if any(lo <= p <= hi for p in query_pages):
                    score = max(score, 0.9)
            if score > 0:
                scored.append({
                    "chunk_id": c.chunk_id,
                    "content": c.content,
                    "score": score,
                    "page_range": list(c.page_range),
                })
        return scored[:top_k]

    def delete(self, document_id: str) -> bool:
        """Delete document and its chunks."""
        try:
            import psycopg
        except ImportError:
            return False
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
                cur.execute("DELETE FROM documents WHERE document_id = %s", (document_id,))
            conn.commit()
        return True

    def list_documents(self) -> list[str]:
        """List all document IDs from PostgreSQL."""
        return [d["document_id"] for d in self.list_documents_with_metadata()]

    def list_documents_with_metadata(self) -> list[dict]:
        """List documents with uploaded_at from PostgreSQL."""
        try:
            import psycopg
        except ImportError:
            return []
        try:
            with psycopg.connect(self.database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT document_id, created_at FROM documents ORDER BY created_at DESC"
                    )
                    rows = cur.fetchall()
            return [
                {
                    "document_id": r[0],
                    "uploaded_at": r[1].isoformat() if r[1] else None,
                }
                for r in rows
            ]
        except Exception:
            return []
