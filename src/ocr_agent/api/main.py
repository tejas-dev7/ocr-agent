"""FastAPI application for OCR Agent."""

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load .env from project root (parent of src/)
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from ocr_agent.api.routes import documents, logs, search
from ocr_agent.api.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Cleanup if needed


app = FastAPI(
    title="OCR Agent API",
    description="Extract text from PDFs, search, and RAG Q&A",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(search.router)
app.include_router(logs.router)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check."""
    return HealthResponse(status="ok")
