"""Unified log buffer for LLM audit + OCR/pipeline events."""

from datetime import datetime, timezone
from threading import Lock

# OCR/pipeline events (LLM comes from llm.audit)
_ocr_buffer: list[dict] = []
_ocr_buffer_max = 500
_ocr_lock = Lock()


def push_ocr_log(entry: dict) -> None:
    """Push OCR/pipeline event to unified log buffer and persist to DB."""
    if "timestamp" not in entry:
        entry = {**entry, "timestamp": datetime.now(timezone.utc).isoformat()}
    if "source" not in entry:
        entry = {**entry, "source": "ocr"}
    if "event" not in entry:
        entry = {**entry, "event": entry.get("type", "progress")}
    with _ocr_lock:
        _ocr_buffer.append(entry)
        if len(_ocr_buffer) > _ocr_buffer_max:
            _ocr_buffer[:] = _ocr_buffer[-_ocr_buffer_max:]
    try:
        from ocr_agent.storage.logs import store_log
        store_log(entry)
    except Exception:
        pass


def get_ocr_logs(limit: int = 100) -> list[dict]:
    """Get recent OCR log entries."""
    with _ocr_lock:
        return list(_ocr_buffer[-limit:])


def get_unified_logs(limit: int = 100, llm_logs: list[dict] | None = None) -> list[dict]:
    """Merge LLM and OCR logs, sorted by timestamp."""
    from ocr_agent.llm.audit import get_recent_audit_logs

    llm = llm_logs if llm_logs is not None else get_recent_audit_logs(limit=limit * 2)
    ocr = get_ocr_logs(limit=limit * 2)
    merged = []
    for e in llm:
        merged.append({**e, "source": e.get("source", "llm"), "event": e.get("event", "llm_completion")})
    for e in ocr:
        merged.append(e)
    merged.sort(key=lambda x: x.get("timestamp", ""))
    return merged[-limit:]
