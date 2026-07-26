"""
Raw HTTP client layer.

Every external HTTP call in the system lives here, one function per call,
mirroring the n8n HTTP Request nodes 1:1:

  - GuestBookings/GetList (by ref)    <- f_lookup_appointment.json "HTTP Request"
                                         f_cancel_appointment.json "HTTP Request"
  - GuestBookings/GetList (by phone)  <- f_lookup_appointment.json "HTTP Request2"
                                         f_cancel_appointment.json "HTTP Request2"
  - GuestBookings/Cancel/{id}         <- f_cancel_appointment.json "HTTP Request1"/"HTTP Request3"/"HTTP Request4"
  - Authentica send-otp / verify-otp  <- langchain_cancellation.json "send_otp5"/"verify_otp5"

No business logic (filtering, selection, formatting) lives here - that's
tools.py's job. Every function catches network failures itself and
returns a structured result rather than raising, so graph nodes never
need a try/except around a tool call.
"""

import logging
from typing import Optional

import requests

from config import (
    AUTHENTICA_API_KEY,
    AUTHENTICA_BASE_URL,
    AUTHENTICA_FALLBACK_EMAIL,
    AUTHENTICA_TEMPLATE_ID,
    CLIENT_ID_HEADER,
    REQUEST_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


# ==========================================================
# Result helper
# ==========================================================

def _result(success: bool, status_code: Optional[int] = None, data=None, error: Optional[str] = None) -> dict:
    return {"success": success, "status_code": status_code, "data": data, "error": error}


def _headers(client_id: Optional[str] = None, language: Optional[str] = None) -> dict:
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    if client_id:
        headers[CLIENT_ID_HEADER] = client_id
    if language:
        headers["accept-language"] = language
    return headers


# ==========================================================
# Guest Bookings API
# ==========================================================

def get_bookings_by_ref(base_url: str, ref_number: str, language: Optional[str] = None, client_id: Optional[str] = None) -> dict:
    """POST {base_url}/api/GuestBookings/GetList with bookingRefNum.

    Mirrors f_lookup_appointment.json "HTTP Request" / f_cancel_appointment.json "HTTP Request".
    """

    url = f"{base_url}/api/GuestBookings/GetList"
    payload = {"bookingRefNum": ref_number}

    return _post_bookings(url, payload, language, client_id)


def get_bookings_by_phone(
    base_url: str,
    phone: str,
    language: Optional[str] = None,
    client_id: Optional[str] = None,
    page_size: int = 1000,
) -> dict:
    """POST {base_url}/api/GuestBookings/GetList with mobileNumber + pageSize.

    Mirrors f_lookup_appointment.json "HTTP Request2" (pageSize: 1000).
    """

    url = f"{base_url}/api/GuestBookings/GetList"
    payload = {"mobileNumber": phone, "pageSize": page_size}

    return _post_bookings(url, payload, language, client_id)


def _post_bookings(url: str, payload: dict, language: Optional[str], client_id: Optional[str]) -> dict:
    logger.debug("POST %s payload=%s", url, payload)

    try:
        response = requests.post(
            url,
            json=payload,
            headers=_headers(client_id=client_id, language=language),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        logger.warning("Booking lookup timed out: %s", url)
        return _result(False, error="timeout")
    except requests.RequestException as exc:
        logger.exception("Booking lookup request failed: %s", url)
        return _result(False, error=str(exc))

    if response.status_code >= 500:
        return _result(False, response.status_code, error="server_error")

    if response.status_code >= 400:
        return _result(False, response.status_code, error="validation_error")

    try:
        body = response.json()
    except ValueError:
        return _result(False, response.status_code, error="invalid_json_response")

    if not body:
        return _result(False, response.status_code, error="empty_response")

    if not body.get("isSuccess"):
        return _result(False, response.status_code, data=body, error="api_reported_failure")

    return _result(True, response.status_code, data=body.get("data", {}))


def cancel_booking_by_guid(base_url: str, booking_guid: str, client_id: Optional[str] = None) -> dict:
    """PUT {base_url}/api/GuestBookings/Cancel/{booking_guid}.

    Mirrors f_cancel_appointment.json "HTTP Request1"/"HTTP Request3"
    (onError: continueErrorOutput -> here, a structured failure result
    instead of a raised exception achieves the same thing).
    """

    url = f"{base_url}/api/GuestBookings/Cancel/{booking_guid}"

    logger.debug("PUT %s", url)

    try:
        response = requests.put(
            url,
            headers=_headers(client_id=client_id),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        logger.warning("Cancel request timed out: %s", url)
        return _result(False, error="timeout")
    except requests.RequestException as exc:
        logger.exception("Cancel request failed: %s", url)
        return _result(False, error=str(exc))

    if response.status_code >= 500:
        return _result(False, response.status_code, error="server_error")

    if response.status_code >= 400:
        return _result(False, response.status_code, error="validation_error")

    try:
        body = response.json()
    except ValueError:
        return _result(False, response.status_code, error="invalid_json_response")

    if not body:
        return _result(False, response.status_code, error="empty_response")

    if not body.get("isSuccess"):
        return _result(False, response.status_code, data=body, error="api_reported_failure")

    return _result(True, response.status_code, data=body)


# ==========================================================
# Authentica OTP API (real provider - langchain_cancellation.json
# "send_otp5" / "verify_otp5"). Only used when config.OTP_PROVIDER ==
# "authentica"; see services in tools.py for the dummy alternative.
# ==========================================================

def authentica_send_otp(phone: str) -> dict:
    url = f"{AUTHENTICA_BASE_URL}/send-otp"

    payload = {
        "method": "sms",
        "template_id": AUTHENTICA_TEMPLATE_ID,
        "fallback_email": AUTHENTICA_FALLBACK_EMAIL,
        "phone": phone,
    }
    headers = {"Accept": "application/json", "X-Authorization": AUTHENTICA_API_KEY}

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.Timeout:
        return _result(False, error="timeout")
    except requests.RequestException as exc:
        logger.exception("Authentica send_otp failed")
        return _result(False, error=str(exc))

    if response.status_code >= 400:
        return _result(False, response.status_code, error="send_otp_failed")

    try:
        body = response.json()
    except ValueError:
        body = {}

    return _result(True, response.status_code, data=body)


def authentica_verify_otp(phone: str, otp: str, email: str = "") -> dict:
    url = f"{AUTHENTICA_BASE_URL}/verify-otp"

    payload = {"otp": otp, "email": email, "phone": phone}
    headers = {"Accept": "application/json", "X-Authorization": AUTHENTICA_API_KEY}

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.Timeout:
        return _result(False, error="timeout")
    except requests.RequestException as exc:
        logger.exception("Authentica verify_otp failed")
        return _result(False, error=str(exc))

    if response.status_code >= 400:
        return _result(False, response.status_code, error="verify_otp_failed")

    try:
        body = response.json()
    except ValueError:
        body = {}

    verified = bool(body.get("isSuccess") or body.get("success") or body.get("verified"))

    return _result(verified, response.status_code, data=body)
