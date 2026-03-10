#!/usr/bin/env python3
"""Example: Process a PDF with OCR Agent."""

from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from ocr_agent import OCRPipeline, OCRConfig

# Configure - use tesseract by default (no extra deps beyond pytesseract)
config = OCRConfig(
    ocr_provider="tesseract",
    llm_model="openai/gpt-4o-mini",
    storage_backend="json",
    output_dir="./ocr_output",
)

pipeline = OCRPipeline(config)

# Process a PDF
pdf_path = Path(__file__).parent.parent / "MQP annexure for ASP200 PA-HPLC Signed copy_050225.pdf"
if pdf_path.exists():
    result = pipeline.process(pdf_path, store=True)
    print(f"Processed: {result.document_id}")
    print(f"Pages: {len(result.pages)}")
    print(f"Chunks: {len(result.chunks)}")
    print(f"Full text length: {len(result.full_text)} chars")
else:
    # Demo with any PDF
    print("Usage: Place a PDF in the project root or pass path to run_ocr.py")
    print("Example: python examples/run_ocr.py")
