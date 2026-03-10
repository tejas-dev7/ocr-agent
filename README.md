# OCR Agent

A scalable OCR agent that reads PDFs (including scanned images), supports OpenAI/Claude/Ollama via LiteLLM, and stores output for search and RAG.

## Features

- **PDF Processing**: Native text extraction + OCR for scanned pages (PyMuPDF, pdf2image)
- **OCR Providers**: Ollama (minicpm-v4.5), Tesseract
- **Auto OCR selection**: Set `ocr_provider=auto` – a vision LLM analyzes sample pages and picks the best model (tables/handwriting→ollama, simple text→tesseract)
- **LLM Support**: OpenAI, Claude, Gemini, Ollama via LiteLLM
- **Storage**: PostgreSQL+pgvector, Qdrant, or JSON file
- **REST API**: FastAPI with upload, search, and RAG query endpoints

## Installation

```bash
pip install ocr-agent
```

With optional dependencies:

```bash
pip install "ocr-agent[tesseract]"      # Tesseract OCR
pip install "ocr-agent[postgres]"       # PostgreSQL + pgvector
pip install "ocr-agent[qdrant]"         # Qdrant
pip install "ocr-agent[all]"            # All optional deps
```

## Starting the API Server

Step-by-step:

**1. Create a virtual environment**

```bash
python -m venv .venv
```

**2. Activate the virtual environment**

```bash
source .venv/bin/activate   # Unix (macOS, Linux)
# or
.venv\Scripts\activate     # Windows
```

**3. Install requirements**

```bash
pip install -e .
# With optional deps (Qdrant, Tesseract, etc.):
pip install -e ".[all]"
```

**4. Configure environment** (copy `.env.example` to `.env` and add your API keys)

```bash
cp .env.example .env
```

**5. Start the server**

```bash
python server/run_api.py
```

API docs: http://localhost:8000/docs

## Quick Start

### Library

```python
from ocr_agent import OCRPipeline, OCRConfig

config = OCRConfig(
    ocr_provider="ollama",
    llm_model="openai/gpt-4o",
    storage_backend="json",
)
pipeline = OCRPipeline(config)
result = pipeline.process("document.pdf")
```

### REST API

```bash
uvicorn ocr_agent.api.main:app --reload --host 0.0.0.0 --port 8000
```

Docs: http://localhost:8000/docs

### CLI

```bash
ocr-agent process document.pdf --output result.json
```

## Environment Variables

All configuration is loaded from a `.env` file. Copy the example and fill in your values:

```bash
cp .env.example .env
```

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (for LiteLLM) |
| `ANTHROPIC_API_KEY` | Anthropic API key (for LiteLLM) |
| `GEMINI_API_KEY` | Google Gemini API key (for LiteLLM) |
| `OLLAMA_BASE_URL` | Ollama base URL (default: http://localhost:11434) |
| `DATABASE_URL` | PostgreSQL connection string (when using postgres storage) |
| `OCR_*` | OCR Agent settings (see `.env.example` for full list) |

### LLM audit trail

All LLM API calls (RAG queries, OCR routing) are logged to a JSONL file for audit and debugging:

| Variable | Description |
|----------|-------------|
| `OCR_LLM_AUDIT_LOG_ENABLED` | Enable audit logging (default: true) |
| `OCR_LLM_AUDIT_LOG_PATH` | Path to JSONL log file (default: `./llm_audit.jsonl`) |

Each line is a JSON object with `timestamp`, `event`, `status`, `model`, `source`, `document_id`, `messages_preview`, `response_preview`, `duration_sec`, `cost`, and `metadata`.

### Preset profiles

Copy the preset that matches your setup:

| File | LLM | OCR | Storage |
|------|-----|-----|---------|
| `.env.claude` | Claude (Anthropic) | Ollama | Qdrant |
| `.env.gemini` | Gemini (Google) | Ollama | Qdrant |
| `.env.ollama` | Ollama (local) | Ollama minicpm-v4.5 | Qdrant |

```bash
cp .env.claude .env   # or .env.gemini / .env.ollama
# Edit .env and add your API keys
```

For Qdrant setup, see [docs/SETUP_QDRANT.md](docs/SETUP_QDRANT.md).

## License

MIT
