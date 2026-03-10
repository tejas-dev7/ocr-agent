"""Pydantic request/response schemas."""

from pydantic import BaseModel, Field


class DocumentListItem(BaseModel):
    document_id: str
    uploaded_at: str | None = None  # ISO 8601 or None if unknown


class DocumentListResponse(BaseModel):
    documents: list[DocumentListItem]


class UploadResponse(BaseModel):
    document_id: str
    status: str = "processing"
    message: str = "Document queued for OCR processing"


class DocumentStatus(BaseModel):
    document_id: str
    status: str  # pending, processing, completed, failed


class PageContent(BaseModel):
    page_num: int
    text: str
    tables: list[dict] = Field(default_factory=list)


class DocumentContent(BaseModel):
    document_id: str
    status: str
    metadata: dict
    full_text: str
    pages: list[PageContent] = Field(default_factory=list)
    chunks: list[dict] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    top_k: int = Field(default=5, ge=1, le=50)


class SearchResult(BaseModel):
    chunk_id: str
    content: str
    score: float
    page_range: list[int]


class SearchResponse(BaseModel):
    results: list[SearchResult]


class QueryRequest(BaseModel):
    question: str
    llm_model: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"
