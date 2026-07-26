"""
LangGraph node functions and graph construction.

Per the hybrid design: this file contains ONLY deterministic routing and
state manipulation. The three/four LLM touchpoints are hidden behind
tools.detect_message / tools.extract_input_details / tools.resolve_selection
/ tools.parse_confirmation, each of which already falls back to a
heuristic without an LLM configured - graph.py never talks to OpenAI
directly.

Every node that pauses the graph for external input (interrupt()) is
split from any node that has a side effect, per the same rule the
existing cancel_agent_ai_fixed project follows: LangGraph re-runs an
interrupted node from the top on resume, so a node must never both cause
a side effect (like sending an OTP) AND call interrupt() - otherwise the
side effect repeats on every resume.
"""

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt

import tools
from config import (
    CANCELLABLE_STATUSES,
    CANCELLED_STATUS_NAME,
    MAX_CONFIRMATION_RETRIES,
    MAX_OTP_RETRIES,
    MAX_PHONE_FORMAT_RETRIES,
    MAX_SELECTION_RETRIES,
    get_messages,
)
from state import AgentState

logger = logging.getLogger(__name__)


# ==========================================================
# Nodes
# ==========================================================

def load_config(state: AgentState) -> AgentState:
    """Loads client_config.csv / dialect_templates.csv and merges them
    into state["messages"]. Runs first, once, per conversation."""

    messages = get_messages(state["client_id"], state.get("dialect"))

    state["client_config"] = {k: v for k, v in messages.items() if not k.startswith("_")}
    state["messages"] = messages

    return state


def understand_message(state: AgentState) -> AgentState:
    """LLM-assisted language/dialect/intent detection, plus deterministic
    step-back phrase detection."""

    result = tools.detect_message(state["user_message"])

    state["intent"] = result["intent"]
    state["language"] = result["language"]
    state["dialect"] = result["dialect"] or state.get("dialect")
    state["step_back"] = tools.detect_step_back(state["user_message"])

    # Dialect wasn't known when load_config ran (it needs understand_message
    # to have run first) - re-merge messages now that we actually know it,
    # so client_config overrides still apply and dialect defaults are correct.
    state["messages"] = get_messages(state["client_id"], state["dialect"])

    return state


def identify_cancel_method(state: AgentState) -> AgentState:
    """Structured callers already set appointment_id/phone_input directly
    and skip extraction. Free-text callers get classified here."""

    if not state.get("appointment_id") and not state.get("phone_input"):
        details = tools.extract_input_details(state["user_message"])

        state["input_type"] = details.get("input_type")

        if state["input_type"] == "appointment_id":
            state["appointment_id"] = details.get("appointment_id")
        elif state["input_type"] == "phone":
            state["phone_input"] = details.get("phone")
    else:
        state["input_type"] = "appointment_id" if state.get("appointment_id") else "phone"

    return state


def validate_phone_format(state: AgentState) -> AgentState:
    """STEP 1 rule: phone must start with '+' + country code."""

    state["phone_format_valid"] = tools.is_valid_phone_format(state.get("phone_input"))

    return state


def wait_for_valid_phone(state: AgentState) -> AgentState:
    """Interrupt-only node: asks the user to re-enter a correctly
    formatted phone number. Side-effect-free, so it's safe to re-run on
    resume."""

    example = state["messages"].get("_phone_example") or "+201001234567"
    template = tools.format_message(state["messages"], "phone_format_invalid", state.get("language"))

    reply = interrupt({
        "type": "phone_format",
        "message": f"{template} ({example})",
    })

    state["phone_input"] = reply
    state["phone_format_retries"] = state.get("phone_format_retries", 0) + 1

    return state


def normalize_phone(state: AgentState) -> AgentState:
    state["normalized_phone"] = tools.normalize_phone_number(state["phone_input"])

    if state.get("channel_phone"):
        state["normalized_phone_channel"] = tools.normalize_phone_number(state["channel_phone"])

    return state


def compare_phone_node(state: AgentState) -> AgentState:
    """Never compares phones inline - always goes through the
    compare_phone tool, per the n8n prompt's hard rule."""

    if not state.get("channel_phone"):
        # No channel identity known (e.g. plain API call, no WhatsApp
        # context) - OTP is always required in this case, since there is
        # nothing to trust-by-default against.
        state["phone_matched"] = False
        return state

    result = tools.compare_phone.invoke({
        "provided_phone": state["normalized_phone"],
        "channel_phone": state["channel_phone"],
    })

    state["phone_matched"] = bool(result.get("match"))

    return state


def lookup_by_id(state: AgentState) -> AgentState:
    base_url = state["messages"].get("_base_url")

    result = tools.lookup_appointment.invoke({
        "ref_number": state["appointment_id"],
        "phone": "",
        "base_url": base_url,
        "language": state.get("language") or "en",
    })

    if not result["found"]:
        state["appointments"] = []
        state["selected_appointment"] = None
        return state

    appointments = result["appointments"]
    state["appointments"] = appointments
    state["selected_appointment"] = appointments[0]
    state["booking_ref_number"] = appointments[0].get("booking_ref_Number")
    state["booking_guid"] = appointments[0].get("id")
    state["booking_status"] = appointments[0].get("statusName")

    return state


def lookup_by_phone(state: AgentState) -> AgentState:
    base_url = state["messages"].get("_base_url")

    result = tools.lookup_appointment.invoke({
        "ref_number": "",
        "phone": state["normalized_phone"],
        "base_url": base_url,
        "language": state.get("language") or "en",
    })

    appointments = result["appointments"] if result["found"] else []
    state["appointments"] = appointments

    if len(appointments) == 1:
        state["selected_appointment"] = appointments[0]
        state["booking_ref_number"] = appointments[0].get("booking_ref_Number")
        state["booking_guid"] = appointments[0].get("id")
        state["booking_status"] = appointments[0].get("statusName")

    if appointments:
        registered = appointments[0].get("mobileNumber")
        if registered:
            state["otp_target_phone"] = tools.normalize_phone_number(registered)

    return state


def send_otp_node(state: AgentState) -> AgentState:
    """Side-effect-only node - never interrupts. Runs exactly once per
    flow so the OTP is never regenerated/resent on resume."""

    target_phone = state.get("otp_target_phone") or state["normalized_phone"]

    if not state.get("otp_sent"):
        tools.send_otp(target_phone)
        state["otp_sent"] = True
        state["otp_target_phone"] = target_phone

    return state


def wait_for_otp(state: AgentState) -> AgentState:
    """Interrupt-only node - never has a side effect."""

    message = tools.format_message(state["messages"], "msg_patient_booking_number", state.get("language")) \
        if state["messages"].get("msg_patient_booking_number") else \
        ("An OTP code has been sent to the number on file. Please enter it below."
         if (state.get("language") or "en") == "en" else
         "تم إرسال رمز التحقق إلى الرقم المسجل. الرجاء إدخاله أدناه.")

    otp = interrupt({"type": "otp", "message": message})

    state["otp"] = otp

    return state


def verify_otp_node(state: AgentState) -> AgentState:
    target_phone = state.get("otp_target_phone") or state["normalized_phone"]

    verified = tools.verify_otp(target_phone, state["otp"])

    state["otp_verified"] = verified

    if not verified:
        state["otp_retries"] = state.get("otp_retries", 0) + 1

    return state


def wait_for_selection(state: AgentState) -> AgentState:
    language = state.get("language")

    message = tools.format_message(state["messages"], "msg_multi_appointments", language) \
        if state["messages"].get("msg_multi_appointments") else \
        ("You have multiple bookings. Which one would you like to cancel?"
         if (language or "en") == "en" else
         "لديك أكثر من حجز. أي حجز تريد إلغاءه؟")

    payload = {
        "type": "selection",
        "message": message,
        "appointments": state["appointments"],
    }

    if state.get("selection_error"):
        payload["notice"] = state["selection_error"]

    selection = interrupt(payload)

    state["selection"] = selection
    state["selection_error"] = None

    return state


def select_appointment(state: AgentState) -> AgentState:
    raw_selection = state.get("selection")
    appointments = state.get("appointments", [])
    language = state.get("language")

    index = None

    if isinstance(raw_selection, bool):
        index = None
    elif isinstance(raw_selection, int):
        index = raw_selection
    elif isinstance(raw_selection, str) and raw_selection.strip().lstrip("-").isdigit():
        index = int(raw_selection.strip())
    elif raw_selection is not None:
        index = tools.resolve_selection(str(raw_selection), appointments, language)

    if index is None or not (1 <= index <= len(appointments)):
        state["selected_appointment"] = None
        state["selection_error"] = tools.format_message(state["messages"], "selection_not_understood", language)
        state["selection_retries"] = state.get("selection_retries", 0) + 1
        return state

    selected = appointments[index - 1]
    state["selected_appointment"] = selected
    state["booking_ref_number"] = selected.get("booking_ref_Number")
    state["booking_guid"] = selected.get("id")
    state["booking_status"] = selected.get("statusName")
    state["selection_error"] = None

    return state


def show_confirmation(state: AgentState) -> AgentState:
    state["confirmation_pending"] = True
    return state


def wait_for_confirmation(state: AgentState) -> AgentState:
    language = state.get("language")
    appt = state.get("selected_appointment") or {}

    template = tools.format_message(state["messages"], "msg_cancellation_confirmation", language)

    message = (
        f"{template}\n\n{tools.format_booking_card(appt, language=language)}"
        if template else tools.format_booking_card(appt, language=language)
    )

    reply = interrupt({"type": "confirmation", "message": message})

    state["confirmed"] = tools.parse_confirmation(str(reply), language)

    if state["confirmed"] is None:
        state["confirmation_retries"] = state.get("confirmation_retries", 0) + 1

    return state


def refresh_before_cancel(state: AgentState) -> AgentState:
    """Mandatory re-lookup immediately before cancelling - "do NOT use any
    value from memory" per the n8n agent's STEP 4. Re-fetches by whichever
    identifier was originally used and re-matches the previously selected
    appointment by doctor + start time + branch."""

    base_url = state["messages"].get("_base_url")

    if state.get("input_type") == "appointment_id":
        result = tools.lookup_appointment.invoke({
            "ref_number": state["appointment_id"],
            "phone": "",
            "base_url": base_url,
            "language": state.get("language") or "en",
        })
    else:
        result = tools.lookup_appointment.invoke({
            "ref_number": "",
            "phone": state["normalized_phone"],
            "base_url": base_url,
            "language": state.get("language") or "en",
        })

    fresh_list = result["appointments"] if result["found"] else []

    match = tools.find_matching_appointment(fresh_list, state.get("selected_appointment"))

    # Single-booking flows (ref lookup, or phone lookup with exactly one
    # active result) never went through select_appointment, so there's
    # nothing to match by doctor/time/branch against - the fresh single
    # result IS the booking.
    if match is None and len(fresh_list) == 1:
        match = fresh_list[0]

    state["fresh_appointment"] = match

    if match:
        state["booking_ref_number"] = match.get("booking_ref_Number")
        state["booking_guid"] = match.get("id")
        state["booking_status"] = match.get("statusName")

    return state


def check_booking_status(state: AgentState) -> AgentState:
    appt = state.get("fresh_appointment") or state.get("selected_appointment")

    if appt:
        state["booking_status"] = appt.get("statusName")

    return state


def cancel_appointment_node(state: AgentState) -> AgentState:
    base_url = state["messages"].get("_base_url")

    result = tools.cancel_appointment.invoke({
        "booking_guid": state["booking_guid"],
        "base_url": base_url,
    })

    state["cancel_result"] = result

    return state


def build_response(state: AgentState) -> AgentState:
    language = state.get("language")
    messages = state["messages"]

    if (state.get("cancel_result") or {}).get("status") == "success":
        state["response"] = tools.format_message(messages, "msg_cancel_success", language) \
            if messages.get("msg_cancel_success") else tools.format_message(messages, "cancelled", language)

    elif state.get("booking_status") == CANCELLED_STATUS_NAME and not state.get("cancel_result"):
        state["response"] = tools.format_message(messages, "already_cancelled", language)

    elif (state.get("cancel_result") or {}).get("status") == "error":
        state["response"] = tools.format_message(messages, "msg_On_failure", language) \
            if messages.get("msg_On_failure") else tools.format_message(messages, "failed", language)

    elif state.get("confirmed") is False:
        state["response"] = tools.format_message(messages, "msg_back_to_ai", language) \
            if messages.get("msg_back_to_ai") else "Okay, no changes were made."

    elif state.get("confirmation_retries", 0) >= MAX_CONFIRMATION_RETRIES:
        state["response"] = tools.format_message(messages, "msg_handoff_confirmation", language) \
            if messages.get("msg_handoff_confirmation") else tools.format_message(messages, "handoff", language)

    elif state.get("phone_format_retries", 0) >= MAX_PHONE_FORMAT_RETRIES:
        state["response"] = tools.format_message(messages, "msg_handoff_confirmation", language) \
            if messages.get("msg_handoff_confirmation") else tools.format_message(messages, "handoff", language)

    elif state.get("otp_sent") and not state.get("otp_verified"):
        state["response"] = tools.format_message(messages, "otp_invalid", language)

    elif state.get("otp_retries", 0) >= MAX_OTP_RETRIES:
        state["response"] = tools.format_message(messages, "msg_handoff_confirmation", language) \
            if messages.get("msg_handoff_confirmation") else tools.format_message(messages, "handoff", language)

    elif state.get("selection_retries", 0) >= MAX_SELECTION_RETRIES:
        state["response"] = tools.format_message(messages, "msg_handoff_confirmation", language) \
            if messages.get("msg_handoff_confirmation") else tools.format_message(messages, "handoff", language)

    elif state.get("input_type") == "phone" and not state.get("appointments"):
        state["response"] = tools.format_message(messages, "no_bookings", language)

    elif state.get("input_type") == "appointment_id" and not state.get("selected_appointment"):
        state["response"] = tools.format_message(messages, "not_found", language)

    elif state.get("fresh_appointment") is None and state.get("confirmed"):
        # Race condition: booking vanished/changed between selection and
        # the mandatory pre-cancel re-lookup.
        state["response"] = tools.format_message(messages, "not_found", language)

    elif state.get("booking_status") and state["booking_status"] not in CANCELLABLE_STATUSES \
            and state["booking_status"] != CANCELLED_STATUS_NAME:
        state["response"] = f"This booking can no longer be cancelled (status: {state['booking_status']})."

    else:
        state["response"] = tools.format_message(messages, "technical_error", language)

    return state


# ==========================================================
# Conditional edge functions
# ==========================================================

def route_after_understand(state: AgentState) -> str:
    """
    NOTE on step-back scope: like route_after_identify in the reference
    cancel_agent_ai_fixed project, this only ever runs on a FRESH
    graph.invoke(state, ...) call - resuming a paused graph via
    Command(resume=...) re-enters directly at the interrupted node
    (wait_for_otp / wait_for_selection / wait_for_confirmation /
    wait_for_valid_phone), never back through understand_message. So
    "ابدأ من جديد"/"start over" said WHILE paused for OTP/selection/
    confirmation is handled by that node's own resumed value, not here.
    This node only catches a step-back phrase in a brand new message
    (e.g. a free-text webhook restarting mid multi-message exchange
    before anything has interrupted yet). A full mid-interrupt "back one
    step" would need an explicit step-history stack in state, which is
    intentionally out of scope for this rebuild - documented here rather
    than silently unimplemented.
    """

    if state.get("step_back") == "restart":
        for field in (
            "appointment_id", "phone_input", "normalized_phone", "input_type",
            "appointments", "selected_appointment", "selection", "selection_error",
            "otp", "otp_sent", "otp_verified", "otp_target_phone", "phone_matched",
            "confirmation_pending", "confirmed", "fresh_appointment",
            "booking_ref_number", "booking_guid", "booking_status", "cancel_result",
        ):
            state[field] = [] if field == "appointments" else (False if field in ("otp_sent",) else None)

    return "identify_cancel_method"


def route_after_identify(state: AgentState) -> str:
    if state.get("input_type") == "appointment_id":
        return "lookup_by_id"
    if state.get("input_type") == "phone":
        return "validate_phone_format"
    return "build_response"


def route_after_phone_validation(state: AgentState) -> str:
    if state.get("phone_format_valid"):
        return "normalize_phone"
    if state.get("phone_format_retries", 0) >= MAX_PHONE_FORMAT_RETRIES:
        return "build_response"
    return "wait_for_valid_phone"


def route_after_lookup_by_id(state: AgentState) -> str:
    if not state.get("selected_appointment"):
        return "build_response"
    return "show_confirmation"


def route_after_compare_phone(state: AgentState) -> str:
    """
    Always proceeds to lookup_by_phone regardless of match result -
    mirrors the original n8n/cancel_agent_ai_fixed design: lookup_by_phone
    is the ONLY place the registered ("on file") mobile number gets read
    off the booking record, and OTP (when required) must be sent to THAT
    number, not the one the user typed. Deciding OTP-vs-not happens in
    route_after_lookup_by_phone, after the lookup has actually run.
    """

    return "lookup_by_phone"


def route_after_verify_otp(state: AgentState) -> str:
    if state.get("otp_verified"):
        # appointments were already fetched by lookup_by_phone before OTP
        # was ever sent - no need to look them up again.
        return _next_step_for_appointments(state)
    if state.get("otp_retries", 0) >= MAX_OTP_RETRIES:
        return "build_response"
    return "wait_for_otp"


def _next_step_for_appointments(state: AgentState) -> str:
    appointments = state.get("appointments", [])
    if len(appointments) == 1:
        return "show_confirmation"
    return "wait_for_selection"


def route_after_lookup_by_phone(state: AgentState) -> str:
    if not state.get("appointments"):
        return "build_response"

    if state.get("phone_matched"):
        return _next_step_for_appointments(state)

    return "send_otp"


def route_after_selection(state: AgentState) -> str:
    if state.get("selected_appointment"):
        return "show_confirmation"
    if state.get("selection_retries", 0) >= MAX_SELECTION_RETRIES:
        return "build_response"
    return "wait_for_selection"


def route_after_confirmation(state: AgentState) -> str:
    if state.get("confirmed") is True:
        return "refresh_before_cancel"
    if state.get("confirmed") is False:
        return "build_response"
    if state.get("confirmation_retries", 0) >= MAX_CONFIRMATION_RETRIES:
        return "build_response"
    return "wait_for_confirmation"


def route_after_refresh(state: AgentState) -> str:
    if state.get("fresh_appointment"):
        return "check_booking_status"
    return "build_response"


def route_after_status(state: AgentState) -> str:
    if state.get("booking_status") == CANCELLED_STATUS_NAME:
        return "build_response"
    return "cancel_appointment"


# ==========================================================
# Build graph
# ==========================================================

builder = StateGraph(AgentState)

builder.add_node("load_config", load_config)
builder.add_node("understand_message", understand_message)
builder.add_node("identify_cancel_method", identify_cancel_method)
builder.add_node("validate_phone_format", validate_phone_format)
builder.add_node("wait_for_valid_phone", wait_for_valid_phone)
builder.add_node("normalize_phone", normalize_phone)
builder.add_node("compare_phone", compare_phone_node)
builder.add_node("lookup_by_id", lookup_by_id)
builder.add_node("lookup_by_phone", lookup_by_phone)
builder.add_node("send_otp", send_otp_node)
builder.add_node("wait_for_otp", wait_for_otp)
builder.add_node("verify_otp", verify_otp_node)
builder.add_node("wait_for_selection", wait_for_selection)
builder.add_node("select_appointment", select_appointment)
builder.add_node("show_confirmation", show_confirmation)
builder.add_node("wait_for_confirmation", wait_for_confirmation)
builder.add_node("refresh_before_cancel", refresh_before_cancel)
builder.add_node("check_booking_status", check_booking_status)
builder.add_node("cancel_appointment", cancel_appointment_node)
builder.add_node("build_response", build_response)

builder.set_entry_point("load_config")

builder.add_edge("load_config", "understand_message")
builder.add_conditional_edges("understand_message", route_after_understand)
builder.add_conditional_edges("identify_cancel_method", route_after_identify)

builder.add_conditional_edges("lookup_by_id", route_after_lookup_by_id)

builder.add_conditional_edges("validate_phone_format", route_after_phone_validation)
builder.add_edge("wait_for_valid_phone", "validate_phone_format")
builder.add_edge("normalize_phone", "compare_phone")
builder.add_conditional_edges("compare_phone", route_after_compare_phone)

builder.add_edge("send_otp", "wait_for_otp")
builder.add_edge("wait_for_otp", "verify_otp")
builder.add_conditional_edges("verify_otp", route_after_verify_otp)

builder.add_conditional_edges("lookup_by_phone", route_after_lookup_by_phone)

builder.add_edge("wait_for_selection", "select_appointment")
builder.add_conditional_edges("select_appointment", route_after_selection)

builder.add_edge("show_confirmation", "wait_for_confirmation")
builder.add_conditional_edges("wait_for_confirmation", route_after_confirmation)

builder.add_conditional_edges("refresh_before_cancel", route_after_refresh)
builder.add_conditional_edges("check_booking_status", route_after_status)

builder.add_edge("cancel_appointment", "build_response")
builder.add_edge("build_response", END)

checkpointer = MemorySaver()

graph = builder.compile(checkpointer=checkpointer)
