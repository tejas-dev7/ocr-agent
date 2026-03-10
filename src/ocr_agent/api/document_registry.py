"""Persistent document status registry - tracks all documents including failed ones."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

# document_id -> {status, created_at} or legacy: document_id -> status (str)
_registry: dict = {}
_registry_lock = Lock()
_registry_path: Path | None = None


def _normalize_entry(v) -> dict:
    """Normalize registry entry to {status, created_at}."""
    if isinstance(v, dict):
        return {"status": v.get("status", ""), "created_at": v.get("created_at")}
    return {"status": str(v), "created_at": None}


def _get_registry_path() -> Path:
    global _registry_path
    if _registry_path is None:
        from ocr_agent.config import get_config
        config = get_config()
        _registry_path = Path(config.output_dir) / "documents_registry.json"
    return _registry_path


def _load_registry() -> None:
    global _registry
    path = _get_registry_path()
    if not path.exists():
        return
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        with _registry_lock:
            for k, v in data.items():
                _registry[k] = _normalize_entry(v)
    except Exception as e:
        logger.warning("Failed to load document registry: %s", e)


def _save_registry() -> None:
    path = _get_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with _registry_lock:
            data = dict(_registry)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=0)
    except Exception as e:
        logger.warning("Failed to save document registry: %s", e)


def set_status(document_id: str, status: str) -> None:
    """Set document status (pending, processing, completed, failed)."""
    with _registry_lock:
        entry = _registry.get(document_id)
        if entry is None:
            created_at = datetime.now(timezone.utc).isoformat()
            _registry[document_id] = {"status": status, "created_at": created_at}
        else:
            entry = _normalize_entry(entry)
            _registry[document_id] = {"status": status, "created_at": entry.get("created_at")}
    _save_registry()


def get_status(document_id: str) -> str | None:
    """Get document status from registry."""
    with _registry_lock:
        v = _registry.get(document_id)
        if v is None:
            return None
        return _normalize_entry(v)["status"]


def remove(document_id: str) -> bool:
    """Remove document from registry. Returns True if it was present."""
    with _registry_lock:
        if document_id in _registry:
            del _registry[document_id]
            found = True
        else:
            found = False
    if found:
        _save_registry()
    return found


def list_ids() -> list[str]:
    """Get all document IDs from registry."""
    with _registry_lock:
        return list(_registry.keys())


def get_all() -> dict:
    """Get full registry (for merging with in-memory)."""
    with _registry_lock:
        return dict(_registry)


def list_with_metadata() -> list[dict]:
    """List registry entries with document_id, status, created_at."""
    with _registry_lock:
        result = []
        for doc_id, v in _registry.items():
            entry = _normalize_entry(v)
            result.append({
                "document_id": doc_id,
                "status": entry["status"],
                "created_at": entry.get("created_at"),
            })
        return result


def ensure_loaded() -> None:
    """Load registry from disk if not yet loaded."""
    if not _registry and _get_registry_path().exists():
        _load_registry()
