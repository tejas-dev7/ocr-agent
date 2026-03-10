"""Tests for OCR pipeline."""

import tempfile
from pathlib import Path

import pytest

from ocr_agent.config import OCRConfig
from ocr_agent.models import Document, Page
from ocr_agent.pdf_processor import needs_ocr, process_pdf


def test_needs_ocr_empty_pages():
    pages = [Page(page_num=1, text="", tables=[])]
    assert needs_ocr(pages) is True


def test_needs_ocr_with_text():
    pages = [Page(page_num=1, text="This is a long enough text to pass the heuristic.", tables=[])]
    assert needs_ocr(pages) is False


def test_document_to_dict():
    doc = Document(
        document_id="test-id",
        metadata={"filename": "test.pdf"},
        pages=[Page(page_num=1, text="hello", tables=[])],
        chunks=[],
        full_text="hello",
    )
    d = doc.to_dict()
    assert d["document_id"] == "test-id"
    assert d["metadata"]["filename"] == "test.pdf"
    assert d["pages"][0]["text"] == "hello"
