"""OCR pipeline orchestrator."""

from pathlib import Path
from typing import Callable
from uuid import uuid4

from ocr_agent.config import OCRConfig, get_config
from ocr_agent.models import Document
from ocr_agent.pdf_processor import (
    iter_page_images,
    needs_ocr,
    process_pdf,
)
from ocr_agent.table_parser import parse_tables_in_page


class OCRPipeline:
    """Orchestrates PDF → OCR → Chunk → Store."""

    def __init__(self, config: OCRConfig | None = None):
        self.config = config or get_config()
        self._ocr = None
        self._chunker = None
        self._storage = None

    def _get_ocr_for_document(self, pdf_path: Path) -> tuple:
        """Get OCR provider - either from config or via LLM routing.
        Returns (provider, selected_model_name) - selected_model_name is set when auto mode."""
        from ocr_agent.ocr.base import get_ocr_provider

        if self.config.ocr_provider != "auto":
            return get_ocr_provider(self.config), None

        # LLM selects best OCR based on sample pages
        from ocr_agent.ocr.router import select_ocr_provider

        selected = select_ocr_provider(pdf_path, self.config)
        overrides = OCRConfig(
            **{**self.config.model_dump(), "ocr_provider": selected}
        )
        return get_ocr_provider(overrides), selected

    @property
    def ocr(self):
        """Lazy OCR provider - use _get_ocr_for_document() when processing."""
        if self._ocr is None:
            from ocr_agent.ocr.base import get_ocr_provider
            self._ocr = get_ocr_provider(self.config)
        return self._ocr

    @property
    def chunker(self):
        if self._chunker is None:
            from ocr_agent.chunking.base import get_chunker
            self._chunker = get_chunker(self.config)
        return self._chunker

    @property
    def storage(self):
        if self._storage is None:
            from ocr_agent.storage.base import get_storage_provider
            self._storage = get_storage_provider(self.config)
        return self._storage

    def process(
        self,
        pdf_path: str | Path,
        document_id: str | None = None,
        store: bool = True,
        on_progress: Callable[[dict], None] | None = None,
    ) -> Document:
        """
        Process PDF: extract/OCR text, chunk, optionally store.
        on_progress: optional callback(event_dict) for progress events.
        """
        pdf_path = Path(pdf_path)
        document_id = document_id or str(uuid4())
        filename = pdf_path.name

        def emit(event: dict) -> None:
            if on_progress:
                on_progress(event)

        # 1. Get pages (native text or placeholder for OCR)
        emit({"type": "pdf_started", "message": "Processing PDF…", "total_pages": None})
        pages = process_pdf(pdf_path)
        total_pages = len(pages)
        emit({"type": "pdf_parsed", "message": f"Converted {total_pages} pages to images", "total_pages": total_pages})

        selected_ocr = None
        metadata_base = {"filename": filename, "pages": total_pages}
        model_name = "native"

        # 2. Run OCR on pages that need it (emit each page as it completes)
        if needs_ocr(pages):
            ocr_provider, selected_ocr = self._get_ocr_for_document(pdf_path)
            if selected_ocr:
                metadata_base["ocr_model_selected"] = selected_ocr
            model_name = selected_ocr or getattr(ocr_provider, "model", None) or "tesseract"
            for page_num, image in iter_page_images(pdf_path):
                emit({"type": "page_converting", "message": f"Converting page {page_num} to image…", "page_num": page_num, "total_pages": total_pages, "model_name": model_name})
                text = ocr_provider.extract(image)
                cleaned_text, tables = parse_tables_in_page(text)
                for p in pages:
                    if p.page_num == page_num:
                        p.text = cleaned_text
                        p.tables = tables
                        break
                emit({
                    "type": "page_ocr_done",
                    "message": f"Page {page_num} OCR complete",
                    "page_num": page_num,
                    "total_pages": total_pages,
                    "page_content": {"page_num": page_num, "text": cleaned_text, "tables": tables},
                    "model_name": model_name,
                })
                # Persist partial progress so it survives server restart
                if store and hasattr(self.storage, "store_partial"):
                    self.storage.store_partial(document_id, pages, metadata_base, total_pages)
        else:
            # Native text - emit pages as "done" for consistency
            for p in pages:
                emit({
                    "type": "page_ocr_done",
                    "message": f"Page {p.page_num} extracted",
                    "page_num": p.page_num,
                    "total_pages": total_pages,
                    "page_content": {"page_num": p.page_num, "text": p.text, "tables": p.tables},
                    "model_name": "native",
                })

        # 3. Build full text
        emit({"type": "chunking", "message": "Chunking text for search…", "model_name": model_name})
        full_text = "\n\n".join(
            f"--- Page {p.page_num} ---\n{p.text}" for p in pages if p.text.strip()
        )

        # 4. Chunk
        chunks = self.chunker.chunk_pages(pages)

        metadata = {"filename": filename, "pages": len(pages)}
        if selected_ocr:
            metadata["ocr_model_selected"] = selected_ocr

        document = Document(
            document_id=document_id,
            metadata=metadata,
            pages=pages,
            chunks=chunks,
            full_text=full_text,
        )

        if store:
            self.storage.store(document)

        emit({"type": "completed", "message": "Processing complete", "model_name": model_name})
        return document
