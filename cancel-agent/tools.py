"""
Business logic layer: LangChain tools + deterministic/LLM-backed helpers.

This is where n8n's Code nodes (toRiyadh conversion, active-booking
filtering), Switch nodes (ref-vs-phone routing input), and the agent's
classification responsibilities (previously done inline by the LLM
inside Cancel Agent1) all live as plain, testable Python functions -
plus the handful of @tool-wrapped functions the graph calls exactly like
n8n's toolWorkflow / httpRequestTool nodes did.
"""

import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, TypedDict

from langchain_core.tools import tool

import api
from config import (
    BOOKING_TIME_UTC_OFFSET_HOURS,
    CANCELLED_STATUS_CODE,
    CANCELLED_STATUS_NAME,
    DEFAULT_COUNTRY_CODE,
    LLM_CLASSIFICATION_ENABLED,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_TIMEOUT_SECONDS,
    OTP_PROVIDER,
    OTP_TTL_SECONDS,
    TEST_OTP,
)
from prompts import (
    CONFIRMATION_PROMPT,
    IDENTIFY_INPUT_PROMPT,
    RESOLVE_SELECTION_PROMPT,
    UNDERSTAND_MESSAGE_PROMPT,
)

logger = logging.getLogger(__name__)


# ==========================================================
# OpenAI call helper (shared by all three/four LLM touchpoints)
# ==========================================================

def _call_openai_json(system_prompt: str, user_message: str) -> Optional[dict]:
    """Call OpenAI with a JSON-only system prompt and parse the result.
    Returns None on ANY failure (no key configured, network error,
    timeout, invalid JSON) so every caller can fall back to a
    deterministic heuristic instead of raising."""

    if not LLM_CLASSIFICATION_ENABLED:
        return None

    try:
        from openai import OpenAI  # imported lazily: only required with a key

        client = OpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT_SECONDS)

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        return json.loads(response.choices[0].message.content)

    except Exception:
        logger.warning("OpenAI classification call failed, falling back to heuristic", exc_info=True)
        return None


# ==========================================================
# Message understanding (language / dialect / intent)
# ==========================================================

_ARABIC_RANGE = re.compile(r"[\u0600-\u06FF]")


def _looks_arabic(text: str) -> bool:
    return bool(_ARABIC_RANGE.search(text or ""))


def detect_message(message: str) -> dict:
    """Detect intent/language/dialect. Falls back to an Arabic-script
    heuristic (language only, no dialect) with no LLM configured."""

    result = _call_openai_json(UNDERSTAND_MESSAGE_PROMPT, message)

    if result and "language" in result:
        return {
            "intent": result.get("intent", "cancel_appointment"),
            "language": result.get("language", "en"),
            "dialect": result.get("dialect"),
        }

    return {
        "intent": "cancel_appointment",
        "language": "ar" if _looks_arabic(message) else "en",
        "dialect": None,
    }


# ==========================================================
# Step-back detection (deterministic - not an LLM call, per Step 2 design)
# ==========================================================

_RESTART_PHRASES = ("ابدأ من جديد", "رقم ثاني", "start over", "restart")
_BACK_PHRASES = ("رجوع", "back", "go back")


def detect_step_back(message: str) -> Optional[str]:
    """Returns "restart", "back", or None. Deterministic keyword match,
    matching the n8n prompt's STEP-BACK DETECTION block exactly."""

    lowered = (message or "").strip().lower()

    if any(phrase in lowered for phrase in _RESTART_PHRASES):
        return "restart"

    if any(phrase in lowered for phrase in _BACK_PHRASES):
        return "back"

    return None


# ==========================================================
# Free-text input classification (booking ref vs. phone)
# ==========================================================

def extract_input_details(message: str) -> dict:
    """Classify free text into a booking reference or a phone number."""

    result = _call_openai_json(IDENTIFY_INPUT_PROMPT, message)

    if result and result.get("input_type") in ("appointment_id", "phone"):
        return result

    stripped = (message or "").strip()

    # Phone-shaped: starts with + or a run of digits long enough to be a
    # phone number.
    if re.match(r"^\+?\d{8,15}$", stripped.replace(" ", "").replace("-", "")):
        return {"input_type": "phone", "phone": stripped}

    if stripped:
        return {"input_type": "appointment_id", "appointment_id": stripped}

    return {"input_type": "unknown"}


# ==========================================================
# Phone normalization / format validation
# ==========================================================

def normalize_phone_number(phone: Optional[str]) -> Optional[str]:
    """Normalize a phone number to E.164 (e.g. "+201001255864").

    Handles "+20...", "0020...", "20...", and local "01..." shapes.
    """

    if not phone:
        return phone

    cleaned = re.sub(r"[\s\-().]", "", phone.strip())

    if cleaned.startswith("+"):
        return cleaned

    if cleaned.startswith("00"):
        return "+" + cleaned[2:]

    if cleaned.startswith(DEFAULT_COUNTRY_CODE):
        return "+" + cleaned

    if cleaned.startswith("0"):
        return "+" + DEFAULT_COUNTRY_CODE + cleaned[1:]

    return "+" + DEFAULT_COUNTRY_CODE + cleaned


def is_valid_phone_format(phone: Optional[str]) -> bool:
    """Mirrors the n8n prompt's STEP 1 rule: "Validate it starts with +
    followed by country code." Applied to the RAW user input, before
    normalization (which would silently add the "+" for a local number
    and hide a genuinely malformed entry)."""

    if not phone:
        return False

    return bool(re.match(r"^\+\d{7,15}$", phone.strip()))


# ==========================================================
# Identity comparison (compare_phone tool)
# ==========================================================

@tool
def compare_phone(provided_phone: str, channel_phone: str) -> dict:
    """Compare a user-provided phone number against the channel/session
    identity phone number. Returns {"match": true|false}. Mirrors
    compare_phone_cancel1 in langchain_cancellation.json - the hard rule
    from the agent prompt ("NEVER do the phone comparison yourself") is
    honored by always going through this function rather than an inline
    "==" in a node."""

    a = normalize_phone_number(provided_phone)
    b = normalize_phone_number(channel_phone)

    return {"match": bool(a and b and a == b)}


# ==========================================================
# Appointment shaping (f_lookup_appointment.json Code nodes)
# ==========================================================

_FIELD_MAP = (
    ("booking_ref_Number", ("bookingRefNum",)),
    ("servicePrice", ("servicePrice",)),
    ("patientFullName", ("patientFullName",)),
    ("mobileNumber", ("mobileNumber",)),
    ("email", ("email",)),
    ("statusName", ("statusName",)),
    ("branchName", ("branchName",)),
    ("doctorName", ("doctorName",)),
    ("serviceName", ("serviceName",)),
    ("specialtyName", ("specialtyName",)),
)


def to_riyadh(utc_string: Optional[str]) -> Optional[str]:
    """UTC ISO string -> Asia/Riyadh (+3h) ISO string. Mirrors the
    "toRiyadh" helper duplicated in both Code nodes of
    f_lookup_appointment.json."""

    if not utc_string:
        return None

    cleaned = utc_string.replace("Z", "")

    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(cleaned, fmt)
            break
        except ValueError:
            continue
    else:
        try:
            dt = datetime.fromisoformat(cleaned)
        except ValueError:
            return utc_string

    riyadh = dt + timedelta(hours=BOOKING_TIME_UTC_OFFSET_HOURS)
    return riyadh.isoformat() + "+03:00"


def shape_appointment(item: dict) -> dict:
    """Flatten one raw API booking item into the shape used throughout
    the graph, converting timestamps to Riyadh time. Mirrors "filter only
    the important info" / "filter only the important info1" Code nodes."""

    shaped = {}
    for name, keys in _FIELD_MAP:
        for key in keys:
            if key in item:
                shaped[name] = item[key]
                break

    shaped["bookingTimeFrom"] = to_riyadh(item.get("bookingTimeFrom"))
    shaped["bookingTimeTo"] = to_riyadh(item.get("bookingTimeTo"))

    # Keep the raw id (GUID) and raw status code around for cancel/status
    # logic even though they're not in the user-facing field map.
    shaped["id"] = item.get("id")
    shaped["status"] = item.get("status")

    return shaped


def filter_active_appointments(items: list) -> list:
    """Phone-path-only filter from f_lookup_appointment.json: excludes
    already-cancelled (status == 6) and past bookings. NOTE: the
    reference-number path does NOT apply this filter in the original
    workflow - that asymmetry is intentionally preserved, not "fixed"."""

    now = datetime.utcnow()
    active = []

    for item in items:
        if item.get("status") == CANCELLED_STATUS_CODE:
            continue

        raw_from = item.get("bookingTimeFrom")
        if not raw_from:
            continue

        try:
            dt = datetime.fromisoformat(raw_from.replace("Z", ""))
        except ValueError:
            continue

        if dt > now:
            active.append(item)

    return active


# ==========================================================
# LangChain tools: lookup / cancel / OTP
# (thin wrappers around api.py, matching the n8n toolWorkflow interface)
# ==========================================================

@tool
def lookup_appointment(ref_number: str = "", phone: str = "", base_url: str = "", language: str = "en") -> dict:
    """Look up bookings by reference number OR phone number (whichever is
    non-empty), mirroring f_lookup_appointment.json's Switch. Returns
    {"found": bool, "appointments": [...], "message": Optional[str]}."""

    if ref_number:
        result = api.get_bookings_by_ref(base_url, ref_number, language=language)
    elif phone:
        result = api.get_bookings_by_phone(base_url, phone, language=language)
    else:
        return {"found": False, "appointments": [], "message": "no ref_number or phone provided"}

    if not result["success"]:
        return {"found": False, "appointments": [], "message": f"lookup failed: {result.get('error')}"}

    data = result["data"] or {}
    items = data.get("items", [])

    if not items:
        msg = "there is no booking with this reference number" if ref_number else "there is no booking with this phone number"
        return {"found": False, "appointments": [], "message": msg}

    shaped = [shape_appointment(i) for i in items]

    if phone:
        # Phone path applies the active-only filter (not cancelled, not in
        # the past); the reference-number path does not - see
        # filter_active_appointments' docstring for why that asymmetry is
        # intentional.
        active_raw = filter_active_appointments(items)
        active_shaped = [shape_appointment(i) for i in active_raw]

        if not active_shaped:
            return {
                "found": False,
                "appointments": [],
                "message": "all bookings for this phone number are in the past or cancelled",
            }

        return {"found": True, "appointments": active_shaped, "message": None}

    return {"found": True, "appointments": shaped, "message": None}


@tool
def cancel_appointment(booking_guid: str, base_url: str = "") -> dict:
    """Cancel a booking by its internal GUID. Mirrors
    f_cancel_appointment.json's idempotency guard: callers are expected
    to have already checked statusName != "Cancelled" via
    check_booking_status; this function itself just performs the PUT and
    reports success/error."""

    result = api.cancel_booking_by_guid(base_url, booking_guid)

    if result["success"]:
        return {"status": "success"}

    return {"status": "error", "error": result.get("error")}


# ==========================================================
# OTP (dummy provider by default, Authentica when configured)
# ==========================================================

_otp_storage: Dict[str, dict] = {}  # phone -> {"otp": str, "created_at": float}


def send_otp(phone: str) -> dict:
    if OTP_PROVIDER == "authentica":
        result = api.authentica_send_otp(phone)
        return {"success": result["success"]}

    # Dummy provider - mirrors OTP_Dummy_send.json
    _otp_storage[phone] = {"otp": TEST_OTP, "created_at": time.time()}
    logger.info("OTP sent for %s (test otp=%s)", phone, TEST_OTP)
    return {"success": True}


def verify_otp(phone: str, otp: str) -> bool:
    if OTP_PROVIDER == "authentica":
        result = api.authentica_verify_otp(phone, otp)
        return bool(result["success"])

    # Dummy provider - mirrors OTP_Dummy_verify.json, but with a real
    # comparison against TEST_OTP + TTL instead of always returning true,
    # so the graph's OTP-retry logic has something real to exercise.
    record = _otp_storage.get(phone)

    if not record:
        return False

    if time.time() - record["created_at"] > OTP_TTL_SECONDS:
        return False

    return str(otp).strip() == str(record["otp"])


# ==========================================================
# Natural-language appointment selection
# ==========================================================

_ORDINAL_WORDS = {
    "first": 1, "1st": 1, "one": 1,
    "second": 2, "2nd": 2, "two": 2,
    "third": 3, "3rd": 3, "three": 3,
    "fourth": 4, "4th": 4, "four": 4,
    "fifth": 5, "5th": 5, "five": 5,
    "الأول": 1, "الاول": 1, "اول": 1, "أول": 1,
    "الثاني": 2, "ثاني": 2,
    "الثالث": 3, "ثالث": 3,
    "الرابع": 4, "رابع": 4,
    "الخامس": 5, "خامس": 5,
}

_LAST_WORDS = {"last", "latest", "الأخير", "الاخير", "آخر", "اخر"}


def _booking_field(appt: dict, field: str) -> Optional[str]:
    mapping = {
        "doctor": "doctorName",
        "branch": "branchName",
        "service": "serviceName",
        "specialty": "specialtyName",
        "status": "statusName",
        "ref": "booking_ref_Number",
    }
    return appt.get(mapping.get(field, field))


def _heuristic_resolve(text: str, appointments: list) -> Optional[int]:
    stripped = text.strip()
    lowered = stripped.lower()

    digits = re.findall(r"\d+", stripped)
    if digits:
        candidate = int(digits[0])
        if 1 <= candidate <= len(appointments):
            return candidate

    words = re.findall(r"[\w\u0600-\u06FF]+", lowered)
    for word in words:
        if word in _LAST_WORDS:
            return len(appointments)
        if word in _ORDINAL_WORDS:
            candidate = _ORDINAL_WORDS[word]
            if 1 <= candidate <= len(appointments):
                return candidate

    def _normalize(s: str) -> str:
        return re.sub(r"[^\w\s\u0600-\u06FF]", "", s).strip()

    normalized_text = _normalize(lowered)

    matches = []
    for i, appt in enumerate(appointments, start=1):
        haystacks = [
            _booking_field(appt, "doctor"),
            _booking_field(appt, "branch"),
            _booking_field(appt, "service"),
            _booking_field(appt, "specialty"),
            _booking_field(appt, "status"),
            _booking_field(appt, "ref"),
        ]
        haystacks = [_normalize(str(h).lower()) for h in haystacks if h]

        if any(h and (h in normalized_text or normalized_text in h) for h in haystacks):
            matches.append(i)

    return matches[0] if len(matches) == 1 else None


def resolve_selection(text: str, appointments: list, language: str = "en") -> Optional[int]:
    """Resolve free text to a 1-based index into `appointments`. Tries
    cheap deterministic heuristics first, then falls back to the LLM for
    trickier phrasing (dates, combined references)."""

    if not appointments:
        return None

    index = _heuristic_resolve(text, appointments)
    if index is not None:
        return index

    if not LLM_CLASSIFICATION_ENABLED:
        return None

    appointments_block = "\n\n".join(
        f"{i}. doctor={_booking_field(a, 'doctor')}, branch={_booking_field(a, 'branch')}, "
        f"date_from={a.get('bookingTimeFrom')}, status={_booking_field(a, 'status')}, "
        f"ref={_booking_field(a, 'ref')}"
        for i, a in enumerate(appointments, start=1)
    )

    result = _call_openai_json(RESOLVE_SELECTION_PROMPT.format(appointments_block=appointments_block), text)

    if not result:
        return None

    selection = result.get("selection")
    if isinstance(selection, int) and 1 <= selection <= len(appointments):
        return selection

    return None


def find_matching_appointment(appointments: list, reference: dict) -> Optional[dict]:
    """Match a freshly re-fetched appointment list against a previously
    selected appointment by doctorName + bookingTimeFrom + branchName -
    exactly the matching rule STEP 4 of the n8n agent prompt specifies
    for the mandatory pre-cancel re-lookup."""

    if not reference:
        return None

    for appt in appointments:
        if (
            appt.get("doctorName") == reference.get("doctorName")
            and appt.get("bookingTimeFrom") == reference.get("bookingTimeFrom")
            and appt.get("branchName") == reference.get("branchName")
        ):
            return appt

    return None


# ==========================================================
# Confirmation parsing (STEP 4 - "Do NOT act on any other / ambiguous reply")
# ==========================================================

_YES_WORDS = {"yes", "yeah", "yep", "confirm", "confirmed", "go ahead", "ok", "okay",
              "نعم", "أيوة", "ايوة", "أكيد", "اكيد", "تمام", "ماشي"}
_NO_WORDS = {"no", "nope", "don't", "dont", "cancel that", "لا", "متلغيش", "مش عايز"}


def parse_confirmation(text: str, language: str = "en") -> Optional[bool]:
    """Returns True (confirmed), False (declined), or None (unclear -
    caller must re-ask, never guess)."""

    stripped = (text or "").strip().lower()

    if stripped in _YES_WORDS or any(w in stripped for w in _YES_WORDS):
        return True

    if stripped in _NO_WORDS or any(w in stripped for w in _NO_WORDS):
        return False

    result = _call_openai_json(CONFIRMATION_PROMPT, text)

    if result and "confirmed" in result:
        value = result["confirmed"]
        return value if isinstance(value, bool) else None

    return None


# ==========================================================
# Message formatting (client_config/dialect_templates -> user-facing text)
# ==========================================================

_BUILTIN_FALLBACKS = {
    "en": {
        "cancelled": "Your appointment has been cancelled successfully.",
        "already_cancelled": "This booking has already been cancelled.",
        "failed": "We couldn't cancel your appointment. Please contact support.",
        "no_bookings": "We couldn't find any bookings for this phone number.",
        "not_found": "We couldn't find a booking with that reference number.",
        "otp_invalid": "That OTP code is incorrect. Please try again.",
        "phone_format_invalid": "Please enter your phone number starting with + and the country code.",
        "selection_not_understood": "Sorry, I couldn't match that to one of your appointments. Please try again.",
        "confirmation_unclear": "Sorry, I didn't understand. Please reply yes to confirm cancellation, or no to stop.",
        "handoff": "Let's connect you with a member of our staff for further help.",
        "technical_error": "A technical error occurred. Please try again.",
    },
    "ar": {
        "cancelled": "تم إلغاء الحجز بنجاح.",
        "already_cancelled": "هذا الحجز تم إلغاؤه بالفعل.",
        "failed": "تعذر إلغاء الحجز. يرجى التواصل مع الدعم.",
        "no_bookings": "لم يتم العثور على أي حجوزات لهذا الرقم.",
        "not_found": "لم يتم العثور على حجز بهذا الرقم المرجعي.",
        "otp_invalid": "رمز التحقق غير صحيح. حاول مرة أخرى.",
        "phone_format_invalid": "الرجاء إدخال رقم الهاتف بالصيغة + ثم رمز الدولة.",
        "selection_not_understood": "عذرًا، لم أستطع تحديد الحجز المقصود. حاول مرة أخرى.",
        "confirmation_unclear": "عذرًا، لم أفهم. الرجاء الرد بنعم للتأكيد أو لا للإيقاف.",
        "handoff": "سيتم تحويلك لأحد ممثلي خدمة العملاء للمساعدة.",
        "technical_error": "حدث خطأ تقني. حاول مرة أخرى.",
    },
}


def _lang_key(language: Optional[str]) -> str:
    return "ar" if (language or "").lower().startswith("ar") else "en"


def format_message(messages: dict, key: str, language: Optional[str] = "en", **kwargs) -> str:
    """Look up a message. `messages` is state["messages"] (the merged
    client_config/dialect_templates dict from config.get_messages). Falls
    back to the built-in bilingual defaults above for keys that neither
    CSV happens to define for this client/dialect (e.g. genuinely new
    outcomes like "already_cancelled" that this rebuild introduces)."""

    lang = _lang_key(language)

    template = messages.get(key) or _BUILTIN_FALLBACKS[lang].get(key) or _BUILTIN_FALLBACKS["en"].get(key, key)

    try:
        return template.format(**kwargs) if kwargs else template
    except (KeyError, IndexError):
        return template


def format_time_12h(iso_string: Optional[str], language: str = "en") -> str:
    """Render an ISO timestamp in 12-hour format with AM/PM (or ص/م for
    Arabic) - the n8n agent prompt's hard TIME FORMAT rule: "NEVER show
    24-hour or ISO times; convert before displaying." """

    if not iso_string:
        return "\u2014"

    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "").split("+")[0])
    except ValueError:
        return iso_string

    if _lang_key(language) == "ar":
        period = "\u0635\u0628\u0627\u062d\u064b\u0627" if dt.hour < 12 else "\u0645\u0633\u0627\u0621\u064b"
        hour_12 = dt.hour % 12 or 12
        return f"{hour_12:02d}:{dt.minute:02d} {period}"

    return dt.strftime("%I:%M %p").lstrip("0") or dt.strftime("%I:%M %p")


def format_date(iso_string: Optional[str]) -> str:
    if not iso_string:
        return "\u2014"

    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "").split("+")[0])
    except ValueError:
        return iso_string

    return dt.strftime("%d/%m/%Y")


def format_booking_card(appt: dict, index: Optional[int] = None, language: str = "en") -> str:
    """Human-readable single-booking card, used when presenting a list
    for selection."""

    lang = _lang_key(language)
    header = f"{index}) " if index else ""

    date_str = format_date(appt.get("bookingTimeFrom"))
    time_str = format_time_12h(appt.get("bookingTimeFrom"), language)

    if lang == "ar":
        lines = [
            f"{header}رقم الحجز: {appt.get('booking_ref_Number', '—')}",
            f"الطبيب: {appt.get('doctorName', '—')}",
            f"الفرع: {appt.get('branchName', '—')}",
            f"التاريخ: {date_str}",
            f"الوقت: {time_str}",
            f"الحالة: {appt.get('statusName', '—')}",
        ]
    else:
        lines = [
            f"{header}Booking: {appt.get('booking_ref_Number', '—')}",
            f"Doctor: {appt.get('doctorName', '—')}",
            f"Branch: {appt.get('branchName', '—')}",
            f"Date: {date_str}",
            f"Time: {time_str}",
            f"Status: {appt.get('statusName', '—')}",
        ]

    return "\n".join(lines)


def format_booking_list(appointments: list, language: str = "en") -> str:
    separator = "\n" + "-" * 32 + "\n"
    return separator.join(
        format_booking_card(a, index=i + 1, language=language) for i, a in enumerate(appointments)
    )
