"""Search and RAG query routes."""

import re
from fastapi import APIRouter, Depends, HTTPException

from ocr_agent.api.dependencies import get_ocr_config, get_storage
from ocr_agent.api.schemas import QueryRequest, QueryResponse, SearchRequest, SearchResponse
from ocr_agent.llm.client import LLMClient
from ocr_agent.storage.base import StorageProvider

router = APIRouter(prefix="/documents", tags=["search"])


def _extract_page_numbers(question: str) -> list[int]:
    """Extract page numbers from question (e.g. 'page 30', 'p.30', 'explain page 30')."""
    # Match: page 30, pages 5 and 10, p.30, 30th page
    patterns = [
        r"\bpages?\s+(\d+)\b",
        r"\b(?:and|,)\s*(\d+)\b",  # "pages 5 and 10" or "pages 5, 10"
        r"\bp\.?\s*(\d+)\b",
        r"\b(\d+)\s*th\s+page\b",
    ]
    seen = set()
    pages = []
    for pat in patterns:
        for m in re.finditer(pat, question, re.IGNORECASE):
            n = int(m.group(1))
            if n not in seen:
                seen.add(n)
                pages.append(n)
    return sorted(pages)


@router.post("/{document_id}/search", response_model=SearchResponse)
async def search_document(
    document_id: str,
    body: SearchRequest,
    storage: StorageProvider = Depends(get_storage),
):
    """Semantic/keyword search over document chunks."""
    results = storage.search(document_id, body.query, top_k=body.top_k, mode=body.mode)
    return SearchResponse(
        results=[
            {"chunk_id": r["chunk_id"], "content": r["content"], "score": r["score"], "page_range": r["page_range"]}
            for r in results
        ]
    )


def _format_tables(tables: list[dict]) -> str:
    """Format table data as markdown for context."""
    if not tables:
        return ""
    parts = []
    for t in tables:
        headers = t.get("headers", [])
        rows = t.get("rows", [])
        if headers:
            parts.append(" | ".join(headers))
            for row in rows:
                parts.append(" | ".join(str(c) for c in row[: len(headers)]))
    return "\n".join(parts) if parts else ""


def _build_page_context(doc, page_nums: list[int]) -> str:
    """Build context from specific pages (text + tables)."""
    parts = []
    for p in doc.pages:
        if p.page_num in page_nums and p.text.strip():
            page_text = f"--- Page {p.page_num} ---\n{p.text}"
            if p.tables:
                table_md = _format_tables(p.tables)
                if table_md:
                    page_text += "\n\nTable:\n" + table_md
            parts.append(page_text)
    return "\n\n".join(parts)


def _enrich_context_with_tables(doc, page_nums: set[int], base_context: str) -> str:
    """Append table data for given pages when tables exist and aren't already in context."""
    if not page_nums:
        return base_context
    table_parts = []
    for p in doc.pages:
        if p.page_num in page_nums and p.tables:
            table_md = _format_tables(p.tables)
            if table_md:
                table_parts.append(f"--- Page {p.page_num} (tables) ---\nTable:\n{table_md}")
    if not table_parts:
        return base_context
    return base_context + "\n\n" + "\n\n".join(table_parts)


@router.post("/{document_id}/query", response_model=QueryResponse)
async def query_document(
    document_id: str,
    body: QueryRequest,
    storage: StorageProvider = Depends(get_storage),
):
    """RAG-style Q&A over document."""
    doc = storage.get(document_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    # Detect page-specific queries (e.g. "explain page 30", "what's on page 30")
    page_nums = _extract_page_numbers(body.question)
    context_parts = []
    sources = []

    # If user asked about specific page(s), include that content
    if page_nums:
        page_context = _build_page_context(doc, page_nums)
        if page_context:
            context_parts.append(page_context)
            sources.extend([{"page": p, "chunk_id": ""} for p in page_nums])

    # Also get chunks via search (keyword/semantic)
    results = storage.search(document_id, body.question, top_k=8)
    search_page_nums: set[int] = set()
    if results:
        search_context = "\n\n".join(r["content"] for r in results)
        if search_context and search_context not in "\n\n".join(context_parts):
            context_parts.append(search_context)
        # Chunks only contain p.text (tables are in p.tables, not in chunk content)
        # Enrich with table data for pages referenced by search results
        for r in results:
            pr = r.get("page_range", [1, 1])
            for pn in range(pr[0], pr[1] + 1):
                search_page_nums.add(pn)
        if not sources:
            sources = [{"page": r.get("page_range", [1])[0], "chunk_id": r.get("chunk_id", "")} for r in results]

    context = "\n\n".join(context_parts)
    # Enrich search-based context with table data (chunkers never include p.tables)
    # Exclude pages already in page_nums (they already have tables from _build_page_context)
    pages_needing_tables = search_page_nums - set(page_nums) if page_nums else search_page_nums
    if pages_needing_tables:
        context = _enrich_context_with_tables(doc, pages_needing_tables, context)
    if not context:
        # Fallback: build from pages (text + tables), truncate to ~32k chars for context window
        fallback_parts = []
        for p in doc.pages:
            if p.text.strip():
                part = f"--- Page {p.page_num} ---\n{p.text}"
                if p.tables:
                    table_md = _format_tables(p.tables)
                    if table_md:
                        part += "\n\nTable:\n" + table_md
                fallback_parts.append(part)
        context = "\n\n".join(fallback_parts)[:32000]
    if not context:
        return QueryResponse(answer="No content available to answer the question.", sources=[])

    client = LLMClient(config=get_ocr_config())
    answer = client.query_with_context(
        body.question,
        context,
        model=body.llm_model,
        metadata={"source": "rag_query", "document_id": document_id},
    )
    return QueryResponse(answer=answer, sources=sources[:10])
