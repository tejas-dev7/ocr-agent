"""PDF processing - extract text or convert pages to images for OCR."""

from pathlib import Path
from typing import Iterator

from PIL import Image

from ocr_agent.models import Page


def _has_extractable_text(page) -> bool:
    """Check if page has extractable text (not just images)."""
    text = page.get_text().strip()
    return len(text) > 50  # Heuristic: minimal text suggests OCR layer or real text


def process_pdf(pdf_path: str | Path) -> list[Page]:
    """
    Process PDF: extract native text where available, return Page objects.
    For scanned pages, text will be empty - caller must run OCR on images.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF is required. Install with: pip install PyMuPDF")

    doc = fitz.open(pdf_path)
    pages: list[Page] = []

    for i, page in enumerate(doc):
        page_num = i + 1
        text = page.get_text().strip()

        # If little/no text, we'll need OCR - store empty for now
        # Caller can use get_page_images() to get images for OCR
        if not text or not _has_extractable_text(page):
            text = ""  # Signal that OCR is needed

        pages.append(Page(page_num=page_num, text=text, tables=[]))

    doc.close()
    return pages


def get_page_as_image(pdf_path: str | Path, page_num: int, dpi: int = 200) -> Image.Image:
    """Convert a single PDF page to PIL Image for OCR."""
    pdf_path = Path(pdf_path)
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF is required. Install with: pip install PyMuPDF")

    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def iter_page_images(pdf_path: str | Path, dpi: int = 200) -> Iterator[tuple[int, Image.Image]]:
    """Yield (page_num, image) for each page in PDF."""
    pdf_path = Path(pdf_path)
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF is required. Install with: pip install PyMuPDF")

    doc = fitz.open(pdf_path)
    for i in range(len(doc)):
        page = doc[i]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        yield (i + 1, img)
    doc.close()


def needs_ocr(pages: list[Page]) -> bool:
    """Return True if any page needs OCR (has empty/minimal text)."""
    return any(not p.text.strip() or len(p.text.strip()) < 50 for p in pages)
