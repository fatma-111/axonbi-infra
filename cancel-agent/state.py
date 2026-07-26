"""
Shared LangGraph state for the Guest Booking Cancellation Agent.

IMPORTANT: LangGraph derives its state "channels" from this TypedDict's
declared fields. Any key a node sets that is NOT declared here is
silently dropped when that node's update is merged into the checkpoint.
Every field any node in graph.py/node.py ever assigns MUST be declared
below, or it will vanish from the final result without warning.
"""

from typing import Optional, TypedDict


class AgentState(TypedDict):

    # ==========================================================
    # Identity / tenancy
    # ==========================================================

    client_id: str
    session_id: str

    # Generic stand-in for the n8n prompt's "wa_id" - the verified channel
    # identity (e.g. WhatsApp sender id) used by compare_phone. A future
    # WhatsApp adapter sets this from the webhook payload; the CLI/API/
    # tests set it directly. None if the channel doesn't carry an
    # identity of its own (e.g. a pure API call with no known caller phone).
    channel_phone: Optional[str]

    # ==========================================================
    # Config (loaded once by node.load_config, read everywhere else)
    # ==========================================================

    client_config: dict          # raw client_config.csv row (may be {})
    dialect_templates: dict      # raw dialect_templates.csv row (may be {})
    messages: dict               # merged message-template dict, see config.get_messages

    # ==========================================================
    # NLU (LLM-assisted: understand_message / identify_cancel_method)
    # ==========================================================

    user_message: str
    language: Optional[str]      # "ar" | "en"
    dialect: Optional[str]       # "Egyptian" | "Saudi" | ... | None
    intent: Optional[str]
    step_back: Optional[str]     # "restart" | "back" | None

    # ==========================================================
    # Input identification
    # ==========================================================

    input_type: Optional[str]    # "appointment_id" | "phone"
    appointment_id: Optional[str]

    phone_input: Optional[str]
    normalized_phone: Optional[str]
    phone_format_valid: bool
    phone_format_retries: int

    # ==========================================================
    # Identity verification
    # ==========================================================

    phone_matched: Optional[bool]     # result of compare_phone tool
    otp_target_phone: Optional[str]
    otp: Optional[str]
    otp_sent: bool
    otp_verified: Optional[bool]
    otp_retries: int

    # ==========================================================
    # Booking data
    # ==========================================================

    appointments: list
    selected_appointment: Optional[dict]
    selection: Optional[object]       # 1-based int OR natural-language string
    selection_error: Optional[str]
    selection_retries: int

    booking_ref_number: Optional[str]  # human-facing ref, e.g. "GBN-2026-04-28-049"
    booking_guid: Optional[str]        # internal id used by the Cancel endpoint
    booking_status: Optional[str]

    # ==========================================================
    # Confirmation gate (STEP 4 in the n8n agent prompt)
    # ==========================================================

    confirmation_pending: bool
    confirmed: Optional[bool]
    confirmation_retries: int

    # ==========================================================
    # Anti-staleness re-lookup ("do NOT use any value from memory" rule)
    # ==========================================================

    fresh_appointment: Optional[dict]

    # ==========================================================
    # Result
    # ==========================================================

    cancel_result: Optional[dict]     # {"status": "success"|"already_cancelled"|"error", ...}
    response: Optional[str]
    current_step: str
