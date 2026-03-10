"""FastAPI dependencies for pipeline and storage."""

from functools import lru_cache

from ocr_agent.config import OCRConfig, get_config
from ocr_agent.pipeline import OCRPipeline
from ocr_agent.storage.base import StorageProvider, get_storage_provider


@lru_cache
def get_ocr_config() -> OCRConfig:
    return get_config()


def get_pipeline() -> OCRPipeline:
    return OCRPipeline(config=get_ocr_config())


def get_storage() -> StorageProvider:
    return get_storage_provider(get_ocr_config())
