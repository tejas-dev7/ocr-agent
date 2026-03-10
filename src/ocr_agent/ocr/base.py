"""OCR provider protocol and factory."""

from typing import Protocol

from PIL import Image

from ocr_agent.config import OCRConfig, get_config


class OCRProvider(Protocol):
    """Protocol for OCR backends."""

    def extract(self, image: Image.Image) -> str:
        """Extract text from a single image."""
        ...

    def extract_batch(self, images: list[Image.Image]) -> list[str]:
        """Extract text from multiple images."""
        ...


def get_ocr_provider(config: OCRConfig | None = None) -> OCRProvider:
    """Get OCR provider based on config."""
    config = config or get_config()
    provider = config.ocr_provider

    if provider == "tesseract":
        from ocr_agent.ocr.tesseract import TesseractProvider
        return TesseractProvider()
    if provider == "ollama":
        from ocr_agent.ocr.ollama import OllamaVisionProvider
        return OllamaVisionProvider(
            base_url=config.ollama_base_url,
            model=config.ollama_vision_model,
            timeout=config.ollama_timeout,
        )

    raise ValueError(f"Unknown OCR provider: {provider}")
