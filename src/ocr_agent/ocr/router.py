"""LLM-based OCR model router - analyzes sample pages and selects best OCR provider."""

import base64
import io
import re
from pathlib import Path

from PIL import Image

from ocr_agent.config import OCRConfig, get_config
from ocr_agent.pdf_processor import iter_page_images

# Document type -> best OCR provider
# ollama = minicpm-v4.5 (handwriting, tables, annotations)
# tesseract = fast fallback for simple printed text
OCR_SELECTION_PROMPT = """You are an OCR model selector. Analyze these sample pages from a PDF document and choose the BEST OCR model for accurate text extraction.

## Document characteristics to detect:
- **Tables**: Grids, rows/columns, structured data
- **Forms**: Fillable fields, checkboxes, structured layout
- **Handwritten text**: Pen annotations, signatures, handwritten notes (including blue ink)
- **Scientific/technical**: Formulas, symbols, regulatory content
- **Receipts/invoices**: Receipt-style layout
- **Multi-language**: Non-English text
- **Simple printed text**: Clean black/white text, no complex layout
- **Mixed content**: B&W text + colored annotations (e.g. blue pen)

## OCR model options (respond with EXACTLY one word):
- **ollama**: minicpm-v4.5 (Ollama) - Best for handwriting, tables, scanned docs. Use when Ollama available.
- **tesseract**: Fast, simple printed text only. Use only when document is clean B&W with no tables or handwriting.

## Your task:
Look at the sample page(s) and respond with ONLY one of: ollama, tesseract

Response (one word only):"""


def _image_to_base64_url(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 data URL for vision API."""
    buf = io.BytesIO()
    image.save(buf, format=format)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{b64}"


def select_ocr_provider(
    pdf_path: str | Path,
    config: OCRConfig | None = None,
    sample_pages: int = 3,
    llm_model: str | None = None,
) -> str:
    """
    Use a vision LLM to analyze sample PDF pages and select the best OCR provider.

    Returns one of: "ollama", "tesseract"
    """
    config = config or get_config()
    pdf_path = Path(pdf_path)

    # Collect sample page images (first, middle, last)
    pages = list(iter_page_images(pdf_path, dpi=150))
    if not pages:
        return "tesseract"  # Fallback if no pages

    total = len(pages)
    if total <= sample_pages:
        indices = list(range(total))
    else:
        # First, middle, last
        indices = [0, total // 2, total - 1]

    sample_images = [pages[i][1] for i in indices]

    # Resize if too large (vision models have token limits)
    max_dim = 1024
    resized = []
    for img in sample_images:
        w, h = img.size
        if max(w, h) > max_dim:
            ratio = max_dim / max(w, h)
            new_size = (int(w * ratio), int(h * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        resized.append(img)

    # Build vision message for LiteLLM (OpenAI/Anthropic format)
    content: list[dict] = [{"type": "text", "text": OCR_SELECTION_PROMPT}]
    for img in resized:
        content.append({
            "type": "image_url",
            "image_url": {"url": _image_to_base64_url(img)},
        })

    messages = [{"role": "user", "content": content}]

    # Call vision LLM
    from litellm import completion

    from ocr_agent.llm.audit import ensure_audit_registered

    ensure_audit_registered()
    model = llm_model or config.llm_router_model or config.llm_model
    try:
        response = completion(
            model=model,
            messages=messages,
            metadata={"source": "ocr_router", "pdf_path": str(pdf_path)},
        )
        raw = (response.choices[0].message.content or "").strip().lower()
    except Exception:
        return "tesseract"  # Fallback on LLM failure

    # Parse response - extract ollama or tesseract
    match = re.search(r"\b(ollama|tesseract)\b", raw)
    if match:
        return match.group(1)
    return "ollama"  # Safe fallback
