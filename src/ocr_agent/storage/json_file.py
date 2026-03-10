"""JSON file storage - archival, no DB required."""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from ocr_agent.models import Document, Page, Chunk


class JSONFileStorage:
    """Store documents as JSON files."""

    def __init__(self, output_dir: str = "./ocr_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, document_id: str) -> Path:
        return self.output_dir / f"{document_id}.json"

    def store(self, document: Document) -> None:
        """Save document to JSON file."""
        path = self._path(document.document_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(document.to_dict(), f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

    def store_partial(
        self,
        document_id: str,
        pages: list[Page],
        metadata: dict,
        total_pages: int,
    ) -> None:
        """Save partial document (e.g. after each page) so progress survives restarts."""
        full_text = "\n\n".join(
            f"--- Page {p.page_num} ---\n{p.text}" for p in pages if p.text.strip()
        )
        doc = Document(
            document_id=document_id,
            metadata={**metadata, "total_pages": total_pages, "processed_pages": len([p for p in pages if p.text.strip()])},
            pages=pages,
            chunks=[],
            full_text=full_text,
        )
        self.store(doc)

    def get(self, document_id: str) -> Document | None:
        """Load document from JSON file."""
        path = self._path(document_id)
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        pages = [
            Page(page_num=p["page_num"], text=p["text"], tables=p.get("tables", []))
            for p in data["pages"]
        ]
        chunks = [
            Chunk(
                chunk_id=c["chunk_id"],
                content=c["content"],
                page_range=tuple(c["page_range"]),
                embedding=c.get("embedding"),
                metadata=c.get("metadata", {}),
            )
            for c in data["chunks"]
        ]
        return Document(
            document_id=data["document_id"],
            metadata=data["metadata"],
            pages=pages,
            chunks=chunks,
            full_text=data.get("full_text", ""),
        )

    def search(
        self,
        document_id: str,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> list[dict]:
        """Simple keyword search (no embeddings in JSON-only mode)."""
        doc = self.get(document_id)
        if not doc:
            return []
        query_lower = query.lower()
        # Extract page numbers from query (e.g. "page 30" -> 30)
        page_matches = re.findall(r"\bpage\s+(\d+)\b", query_lower) + re.findall(
            r"\bp\.?\s*(\d+)\b", query_lower
        )
        query_pages = {int(p) for p in page_matches}
        scored = []
        for c in doc.chunks:
            score = 0.0
            if query_lower in c.content.lower():
                score = 1.0
            # If query mentions a page number, include chunks that span that page
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
        """Remove JSON file."""
        path = self._path(document_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_documents(self) -> list[str]:
        """List all document IDs from JSON files."""
        return [d["document_id"] for d in self.list_documents_with_metadata()]

    def list_documents_with_metadata(self) -> list[dict]:
        """List documents with uploaded_at (from file mtime)."""
        docs = []
        for p in self.output_dir.glob("*.json"):
            if p.stem == "documents_registry":
                continue
            try:
                mtime = os.path.getmtime(p)
                uploaded_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            except OSError:
                uploaded_at = None
            docs.append({"document_id": p.stem, "uploaded_at": uploaded_at})
        return sorted(docs, key=lambda d: d["uploaded_at"] or "", reverse=True)
