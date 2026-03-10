"""Section-based and page-based chunking."""

import re
from uuid import uuid4

from ocr_agent.models import Chunk, Page


class SectionChunker:
    """Split at heading boundaries (##, ###, etc.)."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 51):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(self, pages: list[Page]) -> list[Chunk]:
        """Chunk by section headings, fallback to size-based."""
        full_text = "\n\n".join(
            f"--- Page {p.page_num} ---\n{p.text}" for p in pages if p.text.strip()
        )
        if not full_text.strip():
            return []

        # Split by markdown-style headings or numbered sections (1., 1.1, etc.)
        section_pattern = r"(?m)^(#{1,6}\s+.+|\d+\.(?:\d+\.)*\s+.+)$"
        parts = re.split(section_pattern, full_text)
        sections = []
        current = []
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            if re.match(r"^#{1,6}\s+|\d+\.(?:\d+\.)*\s+", part):
                if current:
                    sections.append("\n".join(current))
                current = [part]
            else:
                current.append(part)
        if current:
            sections.append("\n".join(current))

        # If no sections found, treat whole as one
        if len(sections) <= 1 and full_text:
            sections = [full_text]

        chunks = []
        for content in sections:
            page_match = re.findall(r"--- Page (\d+) ---", content)
            start_page = int(page_match[0]) if page_match else 1
            end_page = int(page_match[-1]) if page_match else start_page
            chunks.append(
                Chunk(
                    chunk_id=str(uuid4()),
                    content=content,
                    page_range=(start_page, end_page),
                )
            )
        return chunks


class PageChunker:
    """Chunk by fixed number of pages."""

    def __init__(self, pages_per_chunk: int = 5):
        self.pages_per_chunk = pages_per_chunk

    def chunk_pages(self, pages: list[Page]) -> list[Chunk]:
        """Group pages into chunks."""
        chunks = []
        for i in range(0, len(pages), self.pages_per_chunk):
            batch = pages[i : i + self.pages_per_chunk]
            content = "\n\n".join(
                f"--- Page {p.page_num} ---\n{p.text}" for p in batch if p.text.strip()
            )
            if not content.strip():
                continue
            start_page = batch[0].page_num
            end_page = batch[-1].page_num
            chunks.append(
                Chunk(
                    chunk_id=str(uuid4()),
                    content=content,
                    page_range=(start_page, end_page),
                )
            )
        return chunks
