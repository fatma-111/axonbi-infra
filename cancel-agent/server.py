"""HTTP wrapper around the LangGraph cancellation agent (n8n / Messenger)."""
import logging
import os
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from config import configure_logging
from main import (
    start_cancellation_from_message,
    resume_with_message,
    pending_interrupt,
)

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Guest Booking Cancellation Agent API", version="1.0.0")

API_KEY = os.getenv("API_KEY", "")
_sessions: set[str] = set()


class ChatIn(BaseModel):
    session_id: str
    client_id: str
    message: str
    channel_phone: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(body: ChatIn, x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")

    try:
        if body.session_id in _sessions:
            result = resume_with_message(body.session_id, body.message)
        else:
            _sessions.add(body.session_id)
            result = start_cancellation_from_message(
                body.client_id, body.session_id, body.message, body.channel_phone
            )
    except Exception:
        logger.exception("agent error session=%s", body.session_id)
        raise HTTPException(status_code=500, detail="agent error")

    interrupt = pending_interrupt(result)
    return {
        "session_id": body.session_id,
        "done": interrupt is None,
        "interrupt": interrupt,
        "reply": (interrupt or {}).get("message") or result.get("response"),
        "intent": result.get("intent"),
        "current_step": result.get("current_step"),
    }
