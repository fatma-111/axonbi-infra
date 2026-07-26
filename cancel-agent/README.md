# Guest Booking Cancellation Agent (LangGraph rebuild)

A LangGraph reimplementation of the n8n workflow set:
`langchain_cancellation.json` (WhatsApp agent orchestrator) +
`f_lookup_appointment.json` + `f_cancel_appointment.json` +
`OTP_Dummy_send.json` / `OTP_Dummy_verify.json`, using `client_config.csv`
and `Arabic_Dialect_templates.csv` for multi-tenant branding/localization.

**Design: hybrid.** The graph itself is fully deterministic (LangGraph
nodes + conditional edges). The LLM is used only for four narrow
classification tasks - language/dialect detection, free-text
ref-vs-phone extraction, natural-language appointment selection, and
yes/no confirmation parsing - each with a deterministic fallback so the
whole agent runs correctly with **no OpenAI key configured**.

Out of scope (by request): the WhatsApp webhook / Meta Graph API layer.
The graph exchanges plain `(message in) -> (response out)`, channel-agnostic,
the same way a future webhook adapter would call `main.py`'s functions.

## Project layout

```
config.py       - env vars, retry limits, CSV loading/merging (client_config + dialects)
state.py        - AgentState TypedDict (every field any node sets)
prompts.py      - the 4 LLM prompt templates (JSON-only outputs)
api.py          - raw HTTP calls: Guest Bookings API + Authentica OTP API
tools.py        - business logic: LangChain tools, NLU (LLM+heuristic), formatting
graph.py        - node functions + StateGraph construction (the actual flow)
main.py         - session/thread helpers, public entrypoints, CLI
test_mocked.py  - 7-scenario smoke test, no network/OpenAI required
data/
  client_config.csv       - per-clinic branding/routing/message overrides
  dialect_templates.csv   - per-dialect default message templates
requirements.txt
```

## Running it

```bash
pip install -r requirements.txt --break-system-packages   # or a venv
python3 test_mocked.py     # verifies the whole graph with mocked HTTP calls
python3 main.py            # interactive CLI against the real Guest Bookings API
```

Environment variables (all optional - sane defaults for local/dev use):

| Variable | Default | Purpose |
|---|---|---|
| `BOOKING_API_BASE_URL` | `https://demo.catalystsystems.io:1302` | fallback if a client's `base_url` column is empty |
| `OTP_PROVIDER` | `dummy` | `dummy` (always-works, mirrors `OTP_Dummy_*.json`) or `authentica` (real `api.authentica.sa`) |
| `AUTHENTICA_API_KEY` | *(empty)* | required if `OTP_PROVIDER=authentica` |
| `OPENAI_API_KEY` | *(empty)* | enables LLM classification; without it, deterministic heuristics run everywhere |
| `DEFAULT_COUNTRY_CODE` | `20` | used when normalizing a local-format phone number |
| `AGENT_DATA_DIR` | `./data` | where `client_config.csv`/`dialect_templates.csv` are read from |

## Business rules preserved from the original workflow

- **Ref-lookup vs. phone-lookup asymmetry**: the phone path filters out
  cancelled/past bookings; the reference path does not. This is a real
  difference in the original `f_lookup_appointment.json`, not a bug -
  preserved as-is.
- **OTP goes to the number on file**, not the number the user typed, and
  not the channel/WhatsApp sender number - only when the typed number
  doesn't match the channel identity.
- **OTP is sent exactly once per flow.** The side-effect (`send_otp`) and
  the pause (`wait_for_otp`) are split into two nodes because LangGraph
  re-runs an interrupted node from the top on every resume; combining
  them would resend/regenerate the OTP on every retry.
- **Mandatory re-lookup immediately before cancelling** ("do NOT use any
  value from memory"): `refresh_before_cancel` re-fetches the booking and
  matches it by `doctorName + bookingTimeFrom + branchName` before the
  actual cancel call ever fires.
- **Idempotency**: a booking already in `Cancelled` status is never
  re-cancelled; `check_booking_status` short-circuits straight to the
  final response.
- **Explicit confirmation gate**: cancellation never proceeds on an
  ambiguous reply - `parse_confirmation` returns `None` (not a guess) for
  anything that isn't a clear yes/no, which re-prompts instead of acting.
- **Bounded retries everywhere**: phone-format, OTP, selection, and
  confirmation loops each have their own counter and hand off to a
  "please contact support" message after `config.MAX_*_RETRIES` attempts,
  instead of looping forever.

## Known simplifications (documented, not hidden)

- **Step-back ("start over"/"back") only works before the first
  interrupt** in a given flow. Resuming a paused graph via
  `Command(resume=...)` re-enters directly at the interrupted node, never
  back through `understand_message` - so a step-back phrase typed *while
  paused* is handled by whichever node is paused, not by the step-back
  router. A true multi-level "go back one step" would need an explicit
  step-history stack in `AgentState`, intentionally left out of scope.
- Multi-tenant config assumes one `client_id` per conversation, resolved
  by the caller (CLI/API), matching how `base_url`/`client_id` were
  supplied as workflow inputs in the original n8n sub-workflows.
