"""OCR Agent - Extract text from PDFs with scanned images."""

from ocr_agent.pipeline import OCRPipeline
from ocr_agent.config import OCRConfig
from ocr_agent.models import Document, Page, Chunk

__all__ = [
    "OCRPipeline",
    "OCRConfig",
    "Document",
    "Page",
    "Chunk",
]
__version__ = "0.1.0"
