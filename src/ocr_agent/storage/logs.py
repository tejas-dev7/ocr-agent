"""Log storage - PostgreSQL or JSONL fallback."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_log_storage() -> "PostgresLogStorage | JSONLLogStorage":
    """Get log storage backend based on config. Prefers Postgres when available."""
    from ocr_agent.config import get_config

    config = get_config()
    if not getattr(config, "logs_store_in_db", True):
        log_path = Path(config.output_dir) / "logs.jsonl"
        return JSONLLogStorage(str(log_path))
    url = getattr(config, "logs_database_url", None) or config.database_url
    if url and ("postgresql" in url or "postgres" in url):
        try:
            return PostgresLogStorage(url)
        except ImportError:
            pass
    log_path = Path(config.output_dir) / "logs.jsonl"
    return JSONLLogStorage(str(log_path))


class PostgresLogStorage:
    """Store logs in PostgreSQL."""

    def __init__(self, database_url: str):
        self.database_url = database_url

    def store(self, entry: dict) -> None:
        """Store a single log entry."""
        try:
            import psycopg
        except ImportError:
            raise ImportError("PostgreSQL log storage requires: pip install ocr-agent[postgres]") from None

        source = entry.get("source", "unknown")
        event = entry.get("event", entry.get("type", "log"))
        payload = {k: v for k, v in entry.items() if k not in ("source", "event")}

        try:
            with psycopg.connect(self.database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS logs (
                            id BIGSERIAL PRIMARY KEY,
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            source TEXT NOT NULL,
                            event TEXT NOT NULL,
                            payload JSONB NOT NULL
                        )
                        """
                    )
                    cur.execute(
                        "INSERT INTO logs (source, event, payload) VALUES (%s, %s, %s)",
                        (source, event, psycopg.types.json.Jsonb(payload)),
                    )
                conn.commit()
        except Exception as e:
            logger.warning("Failed to store log in DB: %s", e)

    def get_recent(self, limit: int = 100) -> list[dict]:
        """Get recent logs, merged into full entry format."""
        try:
            import psycopg
        except ImportError:
            return []

        try:
            with psycopg.connect(self.database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT source, event, payload, created_at
                        FROM logs
                        ORDER BY id DESC
                        LIMIT %s
                        """,
                        (limit,),
                    )
                    rows = cur.fetchall()
            out = []
            for source, event, payload, created_at in reversed(rows):
                entry = dict(payload or {})
                entry["source"] = source
                entry["event"] = event
                entry["timestamp"] = created_at.isoformat() if created_at else ""
                out.append(entry)
            return out
        except Exception as e:
            logger.warning("Failed to read logs from DB: %s", e)
            return []


class JSONLLogStorage:
    """Store logs in a JSONL file."""

    def __init__(self, path: str):
        self.path = Path(path)

    def store(self, entry: dict) -> None:
        """Append a log entry to the JSONL file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.warning("Failed to store log in file: %s", e)

    def get_recent(self, limit: int = 100) -> list[dict]:
        """Get recent logs from the file."""
        if not self.path.exists():
            return []
        try:
            lines = []
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        lines.append(line)
            entries = []
            for line in lines[-limit:]:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            return entries
        except Exception as e:
            logger.warning("Failed to read logs from file: %s", e)
            return []


# Lazy singleton
_log_storage: PostgresLogStorage | JSONLLogStorage | None = None


def get_log_storage() -> PostgresLogStorage | JSONLLogStorage:
    """Get the log storage instance."""
    global _log_storage
    if _log_storage is None:
        _log_storage = _get_log_storage()
    return _log_storage


def store_log(entry: dict) -> None:
    """Store a log entry in the configured backend."""
    get_log_storage().store(entry)


def get_logs_from_db(limit: int = 100) -> list[dict]:
    """Get recent logs from the database/file."""
    return get_log_storage().get_recent(limit)
