"""
LLM prompt templates for the Guest Booking Cancellation Agent.

Per the hybrid design (Step 2), the LLM is used ONLY for narrow
classification tasks - never to drive the conversation itself:

  1. UNDERSTAND_MESSAGE_PROMPT   - language / dialect / intent detection
  2. IDENTIFY_INPUT_PROMPT       - free-text -> booking ref vs. phone number
  3. RESOLVE_SELECTION_PROMPT    - free-text -> which appointment (1-based index)
  4. CONFIRMATION_PROMPT         - free-text -> yes / no / unclear

All four demand JSON-only output so callers (utils/language.py-equivalent
code in tools.py) can parse deterministically. Every one of these has a
non-LLM fallback elsewhere in the code, so the agent keeps working (with
reduced flexibility) if OPENAI_API_KEY isn't configured - see
config.LLM_CLASSIFICATION_ENABLED.
"""

# ==========================================================
# 1. Message understanding (language / dialect / intent)
# ==========================================================
#
# Adapted from the n8n agent's own language/dialect awareness
# (dialect_instruction per client) and the existing project's
# UNDERSTAND_MESSAGE_PROMPT. This is classification only - it does not
# generate any user-facing text itself.

UNDERSTAND_MESSAGE_PROMPT = """
You are a classifier for a hospital appointment-cancellation assistant.

Your job is to analyze the user's message. Return ONLY valid JSON.

Detect:

1. intent
Possible values:
- cancel_appointment
- unknown

2. language
Possible values:
- ar
- en

3. dialect
If language is "ar", detect one of:
- Egyptian
- Saudi
- Emirati
- Kuwaiti
- Iraqi
- Levantine
- Modern Standard Arabic
If language is "en", return null.

Rules:
- Detect the user's language from the message itself.
- Detect the Arabic dialect only if the language is Arabic; otherwise null.
- Never explain your answer.
- Never return markdown.
- Return JSON only, matching exactly this shape:

{"intent": "cancel_appointment", "language": "ar", "dialect": "Egyptian"}
"""


# ==========================================================
# 2. Free-text input classification (booking ref vs. phone)
# ==========================================================
#
# Kept close to the existing project's IDENTIFY_INPUT_PROMPT. Only two
# input_type values are relevant to this graph (appointment_id, phone) -
# OTP and selection are never extracted here, since those are only ever
# collected while a specific interrupt (wait_for_otp / wait_for_selection)
# is already pending, where the paused node itself defines what the reply
# means (see node.py).

IDENTIFY_INPUT_PROMPT = """
You are an AI assistant for a hospital appointment-cancellation flow.

Your task is to classify the user's message. Return ONLY valid JSON.

Possible input types:

1. appointment_id
If the message contains a booking reference number
(e.g. "GBN-2026-04-21-021", "GuestBookingNum-2026-04-21-021").

2. phone
If the message contains a phone number, with or without a leading "+"
and country code.

3. unknown
If neither a booking reference nor a phone number can be identified.

Return JSON only, matching exactly one of these shapes:

{"input_type": "appointment_id", "appointment_id": "GBN-2026-04-21-021"}
{"input_type": "phone", "phone": "+201001234567"}
{"input_type": "unknown"}

Never explain your answer. Never return markdown.
"""


# ==========================================================
# 3. Natural-language appointment selection
# ==========================================================
#
# Used only as a fallback when tools.resolve_selection's deterministic
# heuristics (digits, ordinal words in EN/AR, unique field substring
# match) can't confidently resolve the user's reply. {appointments_block}
# is filled in at call time with one line per appointment.

RESOLVE_SELECTION_PROMPT = """
You are helping a user pick exactly ONE appointment from a numbered list.

The user's message may refer to it by:
- position ("first", "second", "last", "number 2", "the 2nd one")
- doctor name ("Dr Omar's appointment", "cancel Mashael booking")
- date or time ("cancel June 24 booking", "the one at 4 PM")
- status ("the confirmed one")
- branch name

The message may be in English or Arabic (any dialect).

Return ONLY valid JSON: {{"selection": <1-based integer>}} if you are
confident about exactly one match, otherwise {{"selection": null}}.
Never explain your answer. Never return markdown.

Appointments:
{appointments_block}
"""


# ==========================================================
# 4. Confirmation parsing (STEP 4 in the n8n agent prompt)
# ==========================================================
#
# The n8n prompt's hard rule: "Do NOT act on any other / ambiguous
# reply." This prompt is deliberately conservative - it must return
# "unclear" rather than guess whenever the reply isn't a clear yes/no,
# since a false-positive "yes" here would cancel a booking the user
# never actually confirmed.

CONFIRMATION_PROMPT = """
You are checking whether a user has clearly confirmed they want to
proceed with cancelling a hospital appointment.

Return ONLY valid JSON, matching exactly one of these shapes:

{"confirmed": true}
{"confirmed": false}
{"confirmed": null}

Rules:
- Return {"confirmed": true} ONLY if the message is an unambiguous
  affirmative (e.g. "yes", "yeah", "go ahead", "confirmed", "نعم",
  "أيوة", "أكيد", "ماشي", "تمام كده").
- Return {"confirmed": false} ONLY if the message is an unambiguous
  negative (e.g. "no", "don't cancel", "لا", "متلغيش").
- Return {"confirmed": null} for anything else, including questions,
  requests to change the selection, or anything you are not fully
  confident about. Never guess.
- Never explain your answer. Never return markdown.
"""
