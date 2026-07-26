"""
Standalone smoke test - mocks api.py's HTTP functions so the full graph
(including interrupts/resume) can be exercised with NO real network
access and NO OpenAI key.

Run with:
    python3 test_mocked.py
"""

from unittest.mock import patch

import main


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _api_success(items):
    return {"success": True, "status_code": 200, "data": {"items": items, "totalCount": len(items)}, "error": None}


def test_reference_flow_success():
    section("SCENARIO 1: cancel by reference -> confirm -> cancelled")

    booking = {
        "id": "GUID-REF-1", "bookingRefNum": "GBN-2026-06-20-151", "statusName": "New",
        "status": 1, "doctorName": "Dr. Omar", "branchName": "Downtown",
        "bookingTimeFrom": "2026-08-20T13:00:00", "bookingTimeTo": "2026-08-20T13:30:00",
        "mobileNumber": "+201001255864", "patientFullName": "Sara Ali",
    }

    with patch("api.get_bookings_by_ref", return_value=_api_success([booking])) as mock_lookup, \
         patch("api.cancel_booking_by_guid", return_value={"success": True, "status_code": 200, "data": {"isSuccess": True}, "error": None}) as mock_cancel:

        result = main.start_cancellation_by_reference("Dar El Oyoun-demo", "ref-1", "GBN-2026-06-20-151")
        interrupt = main.pending_interrupt(result)
        print("After lookup, interrupt:", interrupt["type"] if interrupt else None)
        assert interrupt is not None and interrupt["type"] == "confirmation"

        result = main.resume_with_value("ref-1", "yes")
        print("Response:", result.get("response"))
        print("cancel_result:", result.get("cancel_result"))

        assert result.get("cancel_result", {}).get("status") == "success"
        assert mock_lookup.call_count == 2, "expected initial lookup + mandatory pre-cancel re-lookup"
        assert mock_cancel.call_count == 1

    print("PASSED")


def test_reference_not_found():
    section("SCENARIO 2: cancel by reference -> not found")

    with patch("api.get_bookings_by_ref", return_value={"success": True, "status_code": 200, "data": {"items": [], "totalCount": 0}, "error": None}):
        result = main.start_cancellation_by_reference("Dar El Oyoun-demo", "ref-2", "GBN-DOES-NOT-EXIST")
        print("Response:", result.get("response"))
        assert main.pending_interrupt(result) is None
        assert result.get("response")

    print("PASSED")


def test_phone_matches_channel_skips_otp():
    section("SCENARIO 3: phone == channel identity -> OTP skipped, single booking -> confirm -> cancel")

    booking = {
        "id": "GUID-P1", "bookingRefNum": "GBN-P1", "statusName": "Confirmed", "status": 2,
        "doctorName": "Dr. Mashael", "branchName": "Al Manar",
        "bookingTimeFrom": "2026-09-01T09:00:00", "bookingTimeTo": "2026-09-01T09:30:00",
        "mobileNumber": "+201003365691",
    }

    with patch("api.get_bookings_by_phone", return_value=_api_success([booking])), \
         patch("api.cancel_booking_by_guid", return_value={"success": True, "status_code": 200, "data": {"isSuccess": True}, "error": None}), \
         patch("tools.send_otp") as mock_send_otp:

        result = main.start_cancellation_by_phone(
            "Dar El Oyoun-demo", "phone-same-1", "+201003365691", channel_phone="+201003365691"
        )
        interrupt = main.pending_interrupt(result)
        print("Interrupt after lookup:", interrupt["type"] if interrupt else None)
        assert interrupt is not None and interrupt["type"] == "confirmation"
        assert mock_send_otp.call_count == 0, "OTP must not be sent when phone matches channel identity"

        result = main.resume_with_value("phone-same-1", "yes")
        print("Response:", result.get("response"))
        assert result.get("cancel_result", {}).get("status") == "success"

    print("PASSED")


def test_phone_mismatch_requires_otp():
    section("SCENARIO 4: phone != channel identity -> OTP required -> verify -> confirm -> cancel")

    booking = {
        "id": "GUID-P2", "bookingRefNum": "GBN-P2", "statusName": "New", "status": 1,
        "doctorName": "Dr. Omar", "branchName": "Downtown",
        "bookingTimeFrom": "2026-09-05T15:00:00", "bookingTimeTo": "2026-09-05T15:30:00",
        "mobileNumber": "+201099999999",
    }

    with patch("api.get_bookings_by_phone", return_value=_api_success([booking])), \
         patch("api.cancel_booking_by_guid", return_value={"success": True, "status_code": 200, "data": {"isSuccess": True}, "error": None}):

        result = main.start_cancellation_by_phone(
            "Dar El Oyoun-demo", "phone-diff-1", "+201003365691", channel_phone="+201111111111"
        )
        interrupt = main.pending_interrupt(result)
        print("Interrupt (expect otp):", interrupt["type"] if interrupt else None)
        assert interrupt is not None and interrupt["type"] == "otp"

        # OTP sent to the registered number ON THE BOOKING (+201099999999),
        # not the typed number (+201003365691) or the channel number.
        import tools
        assert "+201099999999" in tools._otp_storage, "OTP must be sent to the number on file, not the typed number"

        result = main.resume_with_value("phone-diff-1", tools.TEST_OTP if hasattr(tools, "TEST_OTP") else "123456")
        interrupt = main.pending_interrupt(result)
        print("Interrupt after OTP (expect confirmation):", interrupt["type"] if interrupt else None)
        assert interrupt is not None and interrupt["type"] == "confirmation"

        result = main.resume_with_value("phone-diff-1", "yes")
        print("Response:", result.get("response"))
        assert result.get("cancel_result", {}).get("status") == "success"

    print("PASSED")


def test_multi_appointment_selection_and_decline():
    section("SCENARIO 5: multiple appointments -> natural-language selection -> decline confirmation")

    bookings = [
        {"id": "GUID-A", "bookingRefNum": "GBN-A", "statusName": "New", "status": 1,
         "doctorName": "Dr. Mashael Alshalan", "branchName": "Al Manar",
         "bookingTimeFrom": "2026-08-24T16:00:00", "mobileNumber": "+201001255864"},
        {"id": "GUID-B", "bookingRefNum": "GBN-B", "statusName": "Confirmed", "status": 2,
         "doctorName": "Dr. Omar", "branchName": "Downtown",
         "bookingTimeFrom": "2026-09-14T09:00:00", "mobileNumber": "+201001255864"},
    ]

    with patch("api.get_bookings_by_phone", return_value=_api_success(bookings)):
        result = main.start_cancellation_by_phone(
            "Dar El Oyoun-demo", "multi-1", "+201001255864", channel_phone="+201001255864"
        )
        interrupt = main.pending_interrupt(result)
        print("Interrupt (expect selection):", interrupt["type"] if interrupt else None)
        assert interrupt is not None and interrupt["type"] == "selection"

        result = main.resume_with_value("multi-1", "last")
        interrupt = main.pending_interrupt(result)
        print("After 'last', selected booking_guid:", result.get("booking_guid"))
        assert result.get("booking_guid") == "GUID-B"
        assert interrupt is not None and interrupt["type"] == "confirmation"

        result = main.resume_with_value("multi-1", "no")
        print("Response after decline:", result.get("response"))
        assert result.get("cancel_result") is None

    print("PASSED")


def test_invalid_phone_format_retry():
    section("SCENARIO 6: invalid phone format -> retry -> valid")

    booking = {
        "id": "GUID-F1", "bookingRefNum": "GBN-F1", "statusName": "New", "status": 1,
        "doctorName": "Dr. Omar", "branchName": "Downtown",
        "bookingTimeFrom": "2026-10-01T10:00:00", "mobileNumber": "+201001255864",
    }

    with patch("api.get_bookings_by_phone", return_value=_api_success([booking])):
        result = main.start_cancellation_by_phone(
            "Dar El Oyoun-demo", "fmt-1", "01001255864", channel_phone="+201001255864"
        )
        interrupt = main.pending_interrupt(result)
        print("Interrupt (expect phone_format):", interrupt["type"] if interrupt else None)
        assert interrupt is not None and interrupt["type"] == "phone_format"

        result = main.resume_with_value("fmt-1", "+201001255864")
        interrupt = main.pending_interrupt(result)
        print("Interrupt after fix (expect confirmation):", interrupt["type"] if interrupt else None)
        assert interrupt is not None and interrupt["type"] == "confirmation"

    print("PASSED")


def test_already_cancelled_idempotency():
    section("SCENARIO 7: booking already cancelled -> idempotency short-circuit")

    booking = {
        "id": "GUID-C1", "bookingRefNum": "GBN-C1", "statusName": "Cancelled", "status": 6,
        "doctorName": "Dr. Omar", "branchName": "Downtown",
        "bookingTimeFrom": "2026-07-01T10:00:00", "mobileNumber": "+201001255864",
    }

    with patch("api.get_bookings_by_ref", return_value=_api_success([booking])) as mock_lookup, \
         patch("api.cancel_booking_by_guid") as mock_cancel:

        result = main.start_cancellation_by_reference("Dar El Oyoun-demo", "cancelled-1", "GBN-C1")
        interrupt = main.pending_interrupt(result)
        assert interrupt is not None and interrupt["type"] == "confirmation"

        result = main.resume_with_value("cancelled-1", "yes")
        print("Response:", result.get("response"))
        assert mock_cancel.call_count == 0, "must never call cancel on an already-cancelled booking"

    print("PASSED")


if __name__ == "__main__":
    tests = [
        test_reference_flow_success,
        test_reference_not_found,
        test_phone_matches_channel_skips_otp,
        test_phone_mismatch_requires_otp,
        test_multi_appointment_selection_and_decline,
        test_invalid_phone_format_retry,
        test_already_cancelled_idempotency,
    ]

    for t in tests:
        t()

    print("\nALL TESTS PASSED\n")
