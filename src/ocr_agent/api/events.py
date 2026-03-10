"""Document processing event queue for real-time SSE streaming."""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Per-document event queues (document_id -> deque of events)
# push_doc_event is called from sync pipeline; stream from async
_doc_queues: dict[str, deque] = {}
_doc_queue_lock = Lock()


@dataclass
class DocEvent:
    """Event emitted during document processing."""

    type: str  # pdf_started, page_converted, page_ocr_done, chunking, completed, failed
    document_id: str
    message: str = ""
    page_num: int | None = None
    total_pages: int | None = None
    page_content: dict[str, Any] | None = None  # {page_num, text, tables} when page done
    error: str | None = None
    model_name: str | None = None  # OCR/model used for this operation
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "document_id": self.document_id,
            "message": self.message,
            "page_num": self.page_num,
            "total_pages": self.total_pages,
            "page_content": self.page_content,
            "error": self.error,
            "model_name": self.model_name,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


def push_doc_event(document_id: str, event: DocEvent) -> None:
    """Push event to document queue (called from sync pipeline)."""
    with _doc_queue_lock:
        if document_id not in _doc_queues:
            _doc_queues[document_id] = deque(maxlen=500)
        _doc_queues[document_id].append(event)


def _get_doc_events_sync(document_id: str, since_index: int = 0) -> tuple[list[dict], int]:
    """Sync helper to read events (run in executor to avoid blocking event loop)."""
    with _doc_queue_lock:
        queue = _doc_queues.get(document_id)
        if not queue:
            return [], 0
        events = list(queue)
    start = max(0, since_index)
    if start >= len(events):
        return [], len(events)
    out = [e.to_dict() if hasattr(e, "to_dict") else e for e in events[start:]]
    return out, len(events)


async def get_doc_events(document_id: str, since_index: int = 0) -> tuple[list[dict], int]:
    """Get events for document since index. Returns (events, new_index)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_doc_events_sync, document_id, since_index)


async def stream_doc_events(document_id: str, status_check: Callable | None = None):
    """Async generator yielding SSE events for document processing.
    status_check: optional async (doc_id) -> str to check status when queue empty.
    """
    last_index = 0
    done = False
    empty_iters = 0
    while not done:
        events, new_index = await get_doc_events(document_id, last_index)
        if events:
            empty_iters = 0
            for ev in events:
                yield ev
                if ev.get("type") in ("completed", "failed"):
                    done = True
                    break
            last_index = new_index
        else:
            empty_iters += 1
            if status_check and empty_iters > 10:  # ~3s with no events
                try:
                    status = await status_check(document_id)
                    if status in ("completed", "failed", "not_found"):
                        yield {"type": status, "document_id": document_id, "message": f"Document {status}"}
                        done = True
                except Exception:
                    pass
            if not done:
                await asyncio.sleep(0.3)


def cleanup_doc_queue(document_id: str) -> None:
    """Remove queue after processing complete (optional, for memory)."""
    with _doc_queue_lock:
        if document_id in _doc_queues:
            del _doc_queues[document_id]
