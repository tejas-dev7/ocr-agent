"""Tesseract OCR provider."""

from PIL import Image

from ocr_agent.ocr.base import OCRProvider


class TesseractProvider:
    """Tesseract-based OCR."""

    def __init__(self, lang: str = "eng"):
        self.lang = lang

    def extract(self, image: Image.Image) -> str:
        """Extract text from image using Tesseract."""
        try:
            import pytesseract
        except ImportError:
            raise ImportError("pytesseract required. Install with: pip install ocr-agent[tesseract]")
        return pytesseract.image_to_string(image, lang=self.lang)

    def extract_batch(self, images: list[Image.Image]) -> list[str]:
        """Extract text from multiple images."""
        return [self.extract(img) for img in images]
