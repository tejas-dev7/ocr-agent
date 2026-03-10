"""CLI for OCR Agent."""

import argparse
from pathlib import Path

from dotenv import load_dotenv

# Load .env from current directory or project root
load_dotenv(Path.cwd() / ".env")
import json
import sys
from pathlib import Path

from ocr_agent import OCRPipeline, OCRConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR Agent - Extract text from PDFs")
    parser.add_argument("pdf_path", type=Path, help="Path to PDF file")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file")
    parser.add_argument("--ocr", choices=["tesseract", "ollama", "auto"], default="ollama",
                        help="OCR engine; ollama=minicpm-v4.5; auto=LLM selects best")
    parser.add_argument("--storage", choices=["json", "postgres", "qdrant"], default="json")
    parser.add_argument("--no-store", action="store_true", help="Do not store to backend")
    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"Error: File not found: {args.pdf_path}", file=sys.stderr)
        return 1

    config = OCRConfig(ocr_provider=args.ocr, storage_backend=args.storage)
    pipeline = OCRPipeline(config)
    try:
        result = pipeline.process(args.pdf_path, store=not args.no_store)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    out = result.to_dict()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Saved to {args.output}")
    else:
        print(json.dumps(out, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
