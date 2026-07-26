"""
Agent entrypoint / orchestration layer.

Wraps the compiled LangGraph `graph` (graph.py) behind plain-Python
functions so callers (CLI, a future REST/WhatsApp layer, tests) never
need to know the AgentState shape, thread-id convention, or how to build
a `Command(resume=...)`.
"""

import logging
from typing import Optional, Union

from langgraph.types import Command

from config import CLIENT_ID_HEADER, THREAD_ID_PREFIX, configure_logging
from graph import graph

configure_logging()
logger = logging.getLogger(__name__)


# ==========================================================
# Thread / state helpers
# ==========================================================

def _make_thread_id(session_id: str) -> str:
    """LangGraph checkpoints are keyed by thread_id; this scopes it per
    conversation/session so unrelated users never share a checkpoint."""

    return f"{THREAD_ID_PREFIX}:{session_id}"


def _config_for(session_id: str) -> dict:
    return {"configurable": {"thread_id": _make_thread_id(session_id)}}


def _base_state(client_id: str, session_id: str, user_message: str = "Cancel my appointment",
                 channel_phone: Optional[str] = None) -> dict:
    """Build a fresh AgentState dict with safe defaults filled in."""

    return {
        "client_id": client_id,
        "session_id": session_id,
        "channel_phone": channel_phone,

        "client_config": {},
        "dialect_templates": {},
        "messages": {},

        "user_message": user_message,
        "language": None,
        "dialect": None,
        "intent": None,
        "step_back": None,

        "input_type": None,
        "appointment_id": None,

        "phone_input": None,
        "normalized_phone": None,
        "phone_format_valid": False,
        "phone_format_retries": 0,

        "phone_matched": None,
        "otp_target_phone": None,
        "otp": None,
        "otp_sent": False,
        "otp_verified": None,
        "otp_retries": 0,

        "appointments": [],
        "selected_appointment": None,
        "selection": None,
        "selection_error": None,
        "selection_retries": 0,

        "booking_ref_number": None,
        "booking_guid": None,
        "booking_status": None,

        "confirmation_pending": False,
        "confirmed": None,
        "confirmation_retries": 0,

        "fresh_appointment": None,

        "cancel_result": None,
        "response": None,
        "current_step": "",
    }


# ==========================================================
# Public API - Flow 1 (Booking Reference)
# ==========================================================

def start_cancellation_by_reference(client_id: str, session_id: str, booking_ref: str) -> dict:
    """Start (or restart) a cancellation using a booking reference number."""

    state = _base_state(client_id, session_id)
    state["input_type"] = "appointment_id"
    state["appointment_id"] = booking_ref

    logger.info("Starting reference-based cancellation for session=%s", session_id)

    return graph.invoke(state, config=_config_for(session_id))


# ==========================================================
# Public API - Flow 2 (Phone + OTP)
# ==========================================================

def start_cancellation_by_phone(client_id: str, session_id: str, phone_input: str,
                                 channel_phone: Optional[str] = None) -> dict:
    """
    Start (or restart) a cancellation using a phone number.

    `channel_phone` is the verified identity the message arrived from
    (e.g. a WhatsApp sender id). If it matches `phone_input` after
    normalization, OTP is skipped; if it differs (or is unknown), OTP is
    mandatory - sent to the number ON FILE for the booking, not the one
    typed. See tools.compare_phone / node.lookup_by_phone /
    graph.route_after_compare_phone.
    """

    state = _base_state(client_id, session_id, channel_phone=channel_phone)
    state["input_type"] = "phone"
    state["phone_input"] = phone_input

    logger.info("Starting phone-based cancellation for session=%s", session_id)

    return graph.invoke(state, config=_config_for(session_id))


def resume_with_value(session_id: str, value: Union[str, int]) -> dict:
    """
    Resume whichever node is currently paused (wait_for_valid_phone,
    wait_for_otp, wait_for_selection, wait_for_confirmation) with `value`.
    """

    logger.info("Resuming session=%s", session_id)

    return graph.invoke(Command(resume=value), config=_config_for(session_id))


# ==========================================================
# Free-text entrypoints (e.g. a future WhatsApp webhook / chat UI)
# ==========================================================

def start_cancellation_from_message(client_id: str, session_id: str, user_message: str,
                                     channel_phone: Optional[str] = None) -> dict:
    """Start a new cancellation flow from raw free text; booking-ref vs.
    phone extraction happens inside the graph (identify_cancel_method)."""

    state = _base_state(client_id, session_id, user_message=user_message, channel_phone=channel_phone)

    logger.info("Starting message-based cancellation for session=%s", session_id)

    return graph.invoke(state, config=_config_for(session_id))


def resume_with_message(session_id: str, user_message: str) -> dict:
    """Resume a paused session from a raw free-text follow-up message."""

    return resume_with_value(session_id, user_message.strip())


# ==========================================================
# Interrupt inspection helper
# ==========================================================

def pending_interrupt(result: dict) -> Optional[dict]:
    """Return the pending interrupt payload if `result` represents a
    paused graph, otherwise None. The payload's "type" key is one of:
    "phone_format" | "otp" | "selection" | "confirmation"."""

    interrupts = result.get("__interrupt__")

    if not interrupts:
        return None

    return interrupts[0].value


# ==========================================================
# CLI (local smoke test)
# ==========================================================

def _print_appointments(appointments: list, language: str) -> None:
    import tools

    for i, appt in enumerate(appointments, start=1):
        print()
        print(tools.format_booking_card(appt, index=i, language=language))


def _run_cli() -> None:
    print("=== Guest Booking Cancellation Agent (CLI) ===")

    client_id = input("Client id [Dar El Oyoun-demo]: ").strip() or "Dar El Oyoun-demo"
    session_id = input("Session id [demo-session]: ").strip() or "demo-session"

    mode = input("Cancel by (1) Booking Reference or (2) Phone Number? [1/2]: ").strip()

    if mode == "1":
        booking_ref = input("Enter booking reference: ").strip()
        result = start_cancellation_by_reference(client_id, session_id, booking_ref)
    else:
        phone_input = input("Enter your phone number (e.g. +201001234567): ").strip()
        channel_phone = input(
            "Enter the channel/WhatsApp sender number (demo only - normally known "
            "automatically from the channel), or leave blank: "
        ).strip() or None
        result = start_cancellation_by_phone(client_id, session_id, phone_input, channel_phone)

    language = result.get("language", "en")
    interrupt = pending_interrupt(result)

    while interrupt is not None:
        print(f"\n{interrupt['message']}")

        if interrupt.get("notice"):
            print(f"({interrupt['notice']})")

        if interrupt["type"] == "selection":
            _print_appointments(interrupt["appointments"], language)
            reply = input("\nYour choice (number, \"first\"/\"last\", doctor's name...): ").strip()
        elif interrupt["type"] == "confirmation":
            reply = input("\nConfirm? (yes/no): ").strip()
        else:
            reply = input("\n> ").strip()

        result = resume_with_value(session_id, reply)
        language = result.get("language", language)
        interrupt = pending_interrupt(result)

    print("\n=== Final Response ===")
    print(result.get("response"))


if __name__ == "__main__":
    _run_cli()
