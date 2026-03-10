"""Recursive/semantic chunking with token-based splitting."""

import re
from uuid import uuid4

from ocr_agent.models import Chunk, Page


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


def _split_by_size(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks with overlap."""
    words = text.split()
    chunks = []
    current = []
    current_tokens = 0

    for w in words:
        current.append(w)
        current_tokens += _estimate_tokens(w) + 1
        if current_tokens >= chunk_size:
            chunks.append(" ".join(current))
            # overlap: keep last N words
            overlap_words = max(1, overlap // 5)  # rough
            current = current[-overlap_words:]
            current_tokens = sum(_estimate_tokens(x) + 1 for x in current)
    if current:
        chunks.append(" ".join(current))
    return chunks


class RecursiveChunker:
    """Token-based recursive chunking."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 51):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(self, pages: list[Page]) -> list[Chunk]:
        """Chunk pages by token size with overlap."""
        full_text = "\n\n".join(
            f"--- Page {p.page_num} ---\n{p.text}" for p in pages if p.text.strip()
        )
        if not full_text.strip():
            return []

        texts = _split_by_size(full_text, self.chunk_size, self.chunk_overlap)
        chunks = []
        for i, content in enumerate(texts):
            # Infer page range from content
            page_match = re.findall(r"--- Page (\d+) ---", content)
            if page_match:
                start_page = int(page_match[0])
                end_page = int(page_match[-1]) if len(page_match) > 1 else start_page
            else:
                start_page = end_page = 1
            chunks.append(
                Chunk(
                    chunk_id=str(uuid4()),
                    content=content,
                    page_range=(start_page, end_page),
                )
            )
        return chunks
