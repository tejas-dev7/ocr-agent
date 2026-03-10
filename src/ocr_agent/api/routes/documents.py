"""Document upload, status, content, delete routes."""

import json
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from ocr_agent.api.dependencies import get_pipeline, get_storage
from ocr_agent.api.document_registry import (
    ensure_loaded,
    get_status as registry_get_status,
    list_with_metadata as registry_list_with_metadata,
    remove as registry_remove,
    set_status as registry_set_status,
)
from ocr_agent.api.events import DocEvent, push_doc_event, stream_doc_events
from ocr_agent.api.unified_logs import push_ocr_log

logger = logging.getLogger(__name__)
from ocr_agent.api.schemas import (
    DocumentContent,
    DocumentListItem,
    DocumentListResponse,
    DocumentStatus,
    PageContent,
    UploadResponse,
)
from ocr_agent.pipeline import OCRPipeline
from ocr_agent.storage.base import StorageProvider

router = APIRouter(prefix="/documents", tags=["documents"])

# In-memory status (use Redis/Celery in production)
_document_status: dict[str, str] = {}


def _run_ocr(pipeline: OCRPipeline, path: Path, document_id: str) -> None:
    def on_progress(ev: dict) -> None:
        doc_ev = DocEvent(
            type=ev.get("type", "progress"),
            document_id=document_id,
            message=ev.get("message", ""),
            page_num=ev.get("page_num"),
            total_pages=ev.get("total_pages"),
            page_content=ev.get("page_content"),
            model_name=ev.get("model_name"),
        )
        push_doc_event(document_id, doc_ev)
        push_ocr_log({"document_id": document_id, **doc_ev.to_dict()})

    try:
        _document_status[document_id] = "processing"
        registry_set_status(document_id, "processing")
        pipeline.process(path, document_id=document_id, store=True, on_progress=on_progress)
        _document_status[document_id] = "completed"
        registry_set_status(document_id, "completed")
    except Exception as e:
        _document_status[document_id] = "failed"
        registry_set_status(document_id, "failed")
        fail_ev = DocEvent(type="failed", document_id=document_id, error=str(e), model_name=None)
        push_doc_event(document_id, fail_ev)
        push_ocr_log({"document_id": document_id, **fail_ev.to_dict()})
        logger.exception("OCR processing failed for document %s: %s", document_id, e)
    finally:
        if path.exists():
            path.unlink(missing_ok=True)


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    storage: StorageProvider = Depends(get_storage),
):
    """List all documents with uploaded_at (including failed)."""
    ensure_loaded()
    storage_docs = {d["document_id"]: d["uploaded_at"] for d in storage.list_documents_with_metadata()}
    registry_entries = {e["document_id"]: e["created_at"] for e in registry_list_with_metadata()}
    all_ids = storage_docs.keys() | registry_entries.keys()
    documents = [
        DocumentListItem(
            document_id=doc_id,
            uploaded_at=storage_docs.get(doc_id) or registry_entries.get(doc_id),
        )
        for doc_id in all_ids
    ]
    # Sort by uploaded_at descending (newest first), then by document_id
    documents.sort(key=lambda d: (d.uploaded_at or "", d.document_id), reverse=True)
    return DocumentListResponse(documents=documents)


@router.post("", response_model=UploadResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    pipeline: OCRPipeline = Depends(get_pipeline),
    storage: StorageProvider = Depends(get_storage),
):
    """Upload PDF for OCR processing."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "File must be a PDF")
    document_id = str(uuid4())
    _document_status[document_id] = "pending"
    registry_set_status(document_id, "pending")
    # Save to temp file
    path = Path(f"/tmp/ocr_{document_id}.pdf")
    content = await file.read()
    path.write_bytes(content)
    background_tasks.add_task(_run_ocr, pipeline, path, document_id)
    return UploadResponse(
        document_id=document_id,
        status="processing",
        message="Document queued for OCR processing",
    )


@router.get("/{document_id}/stream")
async def stream_document_events(
    document_id: str,
    storage: StorageProvider = Depends(get_storage),
):
    """SSE stream of document processing events (progress, pages as they complete)."""
    async def status_check(doc_id: str) -> str:
        s = _document_status.get(doc_id)
        if s:
            return s
        s = registry_get_status(doc_id)
        if s:
            return s
        return "completed" if storage.get(doc_id) else "not_found"

    async def event_generator():
        async for ev in stream_doc_events(document_id, status_check=status_check):
            yield f"data: {json.dumps(ev, default=str)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.get("/{document_id}", response_model=DocumentStatus)
async def get_document_status(
    document_id: str,
    storage: StorageProvider = Depends(get_storage),
):
    """Get document processing status."""
    status = _document_status.get(document_id)
    if status is None:
        status = registry_get_status(document_id)
    if status is None:
        doc = storage.get(document_id)
        status = "completed" if doc else "not_found"
    if status == "not_found":
        raise HTTPException(404, "Document not found")
    return DocumentStatus(document_id=document_id, status=status)


@router.get("/{document_id}/content", response_model=DocumentContent)
async def get_document_content(
    document_id: str,
    storage: StorageProvider = Depends(get_storage),
):
    """Get full extracted text and metadata. Returns partial content if processing was interrupted."""
    doc = storage.get(document_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    status = _document_status.get(document_id)
    if status is None:
        status = registry_get_status(document_id) or "completed"
    return DocumentContent(
        document_id=doc.document_id,
        status=status,
        metadata=doc.metadata,
        full_text=doc.full_text,
        pages=[PageContent(page_num=p.page_num, text=p.text, tables=p.tables) for p in doc.pages],
        chunks=[{"chunk_id": c.chunk_id, "content": c.content, "page_range": list(c.page_range)} for c in doc.chunks],
    )


@router.get("/{document_id}/pages")
async def get_document_pages(
    document_id: str,
    page: int = 1,
    per_page: int = 10,
    storage: StorageProvider = Depends(get_storage),
):
    """Get page-level content (paginated)."""
    doc = storage.get(document_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    start = (page - 1) * per_page
    end = start + per_page
    pages = doc.pages[start:end]
    return {
        "document_id": document_id,
        "pages": [PageContent(page_num=p.page_num, text=p.text, tables=p.tables) for p in pages],
        "total": len(doc.pages),
        "page": page,
        "per_page": per_page,
    }


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    storage: StorageProvider = Depends(get_storage),
):
    """Remove document and its stored data."""
    ok = storage.delete(document_id)
    removed_from_registry = registry_remove(document_id)
    if document_id in _document_status:
        del _document_status[document_id]
    if not ok and not removed_from_registry:
        raise HTTPException(404, "Document not found")
    return {"deleted": document_id}
