"""Ollama vision model OCR provider (minicpm-v, llava, etc.)."""

import base64
import io
import json
from urllib.request import Request, urlopen
from urllib.error import URLError

from PIL import Image

from ocr_agent.ocr.base import OCRProvider


class OllamaVisionProvider:
    """OCR via Ollama vision models (minicpm-v, llava, llama3.2-vision)."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "minicpm-v",
        timeout: int = 1800,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def extract(self, image: Image.Image) -> str:
        """Extract text from image using Ollama vision model."""
        b64 = self._image_to_base64(image)
        payload = {
            "model": self.model,
            "prompt": """Extract all text from this image. Return only the extracted text.

For TABLES (including complex scientific/lab tables with nested headers and row-spans):
- Use Markdown table format with | for columns.
- NESTED HEADERS: When a parent header has sub-columns (e.g. "Area" with Inj1, Inj2, Inj3, Average Area), use format Parent::Sub for each sub-column:
  | Sr. No. | Stock Conc. | Area::Inj1 | Area::Inj2 | Area::Inj3 | Area::Avg | Slope | Intercept |
  This creates proper nesting: Area is the parent spanning 4 sub-columns.
- ROW-SPANS: When a value spans multiple rows (e.g. Sr. No. "1" spans 2 rows, Slope "54841.54" spans 14 rows), put the value ONLY in the first row. In subsequent rows that share that value, use ^ (caret) to mean "same as above".
  Example:
  | 1  | 0.1 | 1234 | 54841.54 |
  | ^  | ^   | 2345 | ^        |
  | 2  | 0.5 | 3456 | 54841.54 |
- MULTI-LEVEL HEADERS: Output each header row. Use Parent::Sub for nested columns.
- PRESERVE NUMBERS: Keep all decimal places exactly (e.g. 54841.54, 0.1234).
- UNITS: Include units in headers (mg/ml, µl, µg, %, etc.).
- DENSE DATA: Extract every cell. Do not skip rows or columns.

Simple table:
| Col A | Col B |
|-------|-------|
| a     | b     |

For regular text, preserve structure. No commentary.""",
            "images": [b64],
            "stream": False,
        }
        req = Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
        except URLError as e:
            raise ConnectionError(f"Ollama not reachable at {self.base_url}: {e}") from e
        return data.get("response", "").strip()

    def extract_batch(self, images: list[Image.Image]) -> list[str]:
        """Extract text from multiple images (sequential)."""
        return [self.extract(img) for img in images]
