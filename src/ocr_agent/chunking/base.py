"""Chunker protocol and factory."""

from typing import Protocol

from ocr_agent.config import OCRConfig, get_config
from ocr_agent.models import Chunk, Page


class Chunker(Protocol):
    """Protocol for text chunking."""

    def chunk_pages(self, pages: list[Page]) -> list[Chunk]:
        """Split pages into chunks."""
        ...


def get_chunker(config: OCRConfig | None = None) -> Chunker:
    """Get chunker based on config."""
    config = config or get_config()
    strategy = config.chunk_strategy

    if strategy == "section":
        from ocr_agent.chunking.section import SectionChunker
        return SectionChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
    if strategy == "page":
        from ocr_agent.chunking.section import PageChunker
        return PageChunker(pages_per_chunk=config.pages_per_chunk)
    # default: recursive
    from ocr_agent.chunking.recursive import RecursiveChunker
    return RecursiveChunker(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
