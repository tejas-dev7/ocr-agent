"""LLM API call audit trail - logs all LiteLLM completions for compliance and debugging."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

# Truncate long content in audit logs to avoid huge files
MAX_MESSAGE_LEN = 2000
MAX_RESPONSE_LEN = 4000


def _truncate(text: str, max_len: int) -> str:
    if not text or len(text) <= max_len:
        return text or ""
    return text[:max_len] + f"... [truncated, total {len(text)} chars]"


def _messages_summary(messages: list) -> list[dict]:
    """Summarize messages for audit (truncate content)."""
    out = []
    for m in messages or []:
        role = m.get("role", "unknown")
        content = m.get("content")
        if isinstance(content, list):
            # Vision: text + image_urls
            parts = []
            for p in content:
                if isinstance(p, dict):
                    if p.get("type") == "text":
                        parts.append(_truncate(p.get("text", ""), MAX_MESSAGE_LEN))
                    elif p.get("type") == "image_url":
                        parts.append("[image]")
            content = " | ".join(parts) if parts else "[vision content]"
        else:
            content = _truncate(str(content) if content else "", MAX_MESSAGE_LEN)
        out.append({"role": role, "content_preview": content})
    return out


def _extract_response_text(response_obj) -> str:
    """Extract response text from LiteLLM response."""
    try:
        if response_obj and hasattr(response_obj, "choices") and response_obj.choices:
            msg = response_obj.choices[0].message
            if hasattr(msg, "content"):
                return msg.content or ""
    except Exception:
        pass
    return ""


def audit_callback(kwargs: dict, completion_response, start_time, end_time) -> None:
    """
    LiteLLM success_callback - logs each LLM API call as a JSON line.
    """
    from ocr_agent.config import get_config

    config = get_config()
    if not config.llm_audit_log_enabled:
        return

    try:
        duration_sec = (end_time - start_time).total_seconds() if start_time and end_time else None
        metadata = (
            kwargs.get("metadata")
            or (kwargs.get("litellm_params") or {}).get("metadata")
            or {}
        )

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "llm_completion",
            "status": "success",
            "model": kwargs.get("model", ""),
            "source": metadata.get("source", "unknown"),
            "document_id": metadata.get("document_id"),
            "messages_preview": _messages_summary(kwargs.get("messages", [])),
            "response_preview": _truncate(_extract_response_text(completion_response), MAX_RESPONSE_LEN),
            "duration_sec": round(duration_sec, 3) if duration_sec is not None else None,
            "cost": kwargs.get("response_cost"),
            "cache_hit": kwargs.get("cache_hit", False),
            "metadata": {k: v for k, v in metadata.items() if k not in ("source", "document_id")},
        }

        log_path = Path(config.llm_audit_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with _file_lock:
            line = json.dumps(entry, default=str) + "\n"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line)
        _push_to_buffer(entry)
        _store_log_db(entry)
        logger.info("LLM_AUDIT success | model=%s source=%s doc=%s duration=%.3fs cost=%s", entry["model"], entry["source"], entry.get("document_id"), entry.get("duration_sec") or 0, entry.get("cost"))

    except Exception as e:
        logger.warning("LLM audit log failed: %s", e)


def audit_failure_callback(kwargs: dict, completion_response, start_time, end_time) -> None:
    """
    LiteLLM failure_callback - logs failed LLM API calls.
    """
    from ocr_agent.config import get_config

    config = get_config()
    if not config.llm_audit_log_enabled:
        return

    try:
        duration_sec = (end_time - start_time).total_seconds() if start_time and end_time else None
        metadata = (
            kwargs.get("metadata")
            or (kwargs.get("litellm_params") or {}).get("metadata")
            or {}
        )
        exc = (
            kwargs.get("exception")
            or (completion_response if isinstance(completion_response, Exception) else None)
        )

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "llm_completion",
            "status": "failure",
            "model": kwargs.get("model", ""),
            "source": metadata.get("source", "unknown"),
            "document_id": metadata.get("document_id"),
            "messages_preview": _messages_summary(kwargs.get("messages", [])),
            "duration_sec": round(duration_sec, 3) if duration_sec is not None else None,
            "error": str(exc) if exc else "unknown",
            "metadata": {k: v for k, v in metadata.items() if k not in ("source", "document_id")},
        }

        log_path = Path(config.llm_audit_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with _file_lock:
            line = json.dumps(entry, default=str) + "\n"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line)
        _push_to_buffer(entry)
        _store_log_db(entry)
        logger.info("LLM_AUDIT failure | model=%s source=%s doc=%s duration=%.3fs error=%s", entry["model"], entry["source"], entry.get("document_id"), entry.get("duration_sec") or 0, entry.get("error", "")[:200])

    except Exception as e:
        logger.warning("LLM audit failure log failed: %s", e)


_file_lock = Lock()
_audit_registered = False

# In-memory buffer for real-time UI (last N entries)
_log_buffer: list[dict] = []
_log_buffer_max = 500
_log_buffer_lock = Lock()


def _push_to_buffer(entry: dict) -> None:
    """Append entry to in-memory buffer for SSE streaming."""
    global _log_buffer
    with _log_buffer_lock:
        _log_buffer.append(entry)
        if len(_log_buffer) > _log_buffer_max:
            _log_buffer[:] = _log_buffer[-_log_buffer_max:]


def _store_log_db(entry: dict) -> None:
    """Persist LLM audit entry to database."""
    try:
        from ocr_agent.storage.logs import store_log
        out = {**entry, "source": entry.get("source", "llm"), "event": entry.get("event", "llm_completion")}
        if "model" in entry and "model_name" not in out:
            out["model_name"] = entry["model"]
        store_log(out)
    except Exception:
        pass


def get_recent_audit_logs(limit: int = 100) -> list[dict]:
    """Return recent audit log entries for UI."""
    with _log_buffer_lock:
        return list(_log_buffer[-limit:])


def ensure_audit_registered() -> None:
    """Register audit callbacks with LiteLLM (idempotent)."""
    global _audit_registered
    from ocr_agent.config import get_config

    if _audit_registered:
        return
    config = get_config()
    if not config.llm_audit_log_enabled:
        return

    import litellm

    callbacks = list(litellm.success_callback or [])
    if audit_callback not in callbacks:
        callbacks.append(audit_callback)
    litellm.success_callback = callbacks

    fail_callbacks = list(litellm.failure_callback or [])
    if audit_failure_callback not in fail_callbacks:
        fail_callbacks.append(audit_failure_callback)
    litellm.failure_callback = fail_callbacks

    _audit_registered = True
