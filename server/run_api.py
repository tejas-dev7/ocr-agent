#!/usr/bin/env python3
"""Run the OCR Agent REST API server."""

from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "ocr_agent.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
