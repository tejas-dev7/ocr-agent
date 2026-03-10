"""Logs and SSE streaming for real-time UI - unified LLM + OCR logs."""

import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ocr_agent.api.unified_logs import get_unified_logs

router = APIRouter(prefix="/logs", tags=["logs"])


@router.get("")
async def get_logs(limit: int = 100):
    """Get recent unified logs (LLM audit + OCR/pipeline)."""
    return {"logs": get_unified_logs(limit=limit)}


@router.get("/stream")
async def stream_logs():
    """SSE stream of unified logs - LLM + OCR events in real time."""

    async def event_generator():
        logs = get_unified_logs(limit=500)
        for entry in logs:
            yield f"data: {json.dumps(entry, default=str)}\n\n"
        last_count = len(logs)
        while True:
            await asyncio.sleep(0.5)
            logs = get_unified_logs(limit=500)
            if len(logs) > last_count:
                for entry in logs[last_count:]:
                    yield f"data: {json.dumps(entry, default=str)}\n\n"
                last_count = len(logs)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
