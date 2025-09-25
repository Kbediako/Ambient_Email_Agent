# Local Development & Deployment Notes

This document contains recipes and notes for running the Email Assistant components in a local development environment, particularly under WSL.

## Dependency Pins

- Core libraries are currently pinned to `langchain==0.3.27`, `langsmith[pytest]==0.4.30`, and `langgraph==0.6.7` (Phase 1 of the library upgrade ticket).
- After adjusting pins, run `uv lock` so the resolver captures the new versions and transitive updates (for example, `yarl` and `vcrpy` now appear through the LangSmith pytest extras).
- Live-mode pytest (`pytest tests/test_response.py --agent-module=email_assistant_hitl_memory_gmail -k tool_calls`) may surface LangSmith queue telemetry without Gmail credentials; these warnings are expected when Gmail APIs are not configured locally.
- LangGraph 0.6 agents now load a SQLite checkpointer/store by default. Override paths with `EMAIL_ASSISTANT_CHECKPOINT_PATH` / `EMAIL_ASSISTANT_STORE_PATH` if you want the on-disk artefacts somewhere other than `~/.langgraph/`.
- HITL flows now emit interrupts via `langgraph.prebuilt.interrupt.HumanInterrupt`, and the LLM/tool nodes are wrapped with `@task` so durable replays persist side effects.
- LangSmith experiment pagination is handled through `iter_experiment_runs()` (see `src/email_assistant/eval/judges.py`), which leverages `Client.get_experiment_results()` to stream more than 100 judge records when reviewing datasets locally.

## Running the Reminder Worker

The reminder worker is a standalone script that should run as a persistent background process to check for due reminders.

### 1. Using tmux (interactive sessions)

tmux lets you run the worker in a session you can detach from and leave running.

- Start a new session:
  ```bash
  tmux new -s reminders
  ```
- Inside tmux, activate the env and start the worker in loop mode:
  ```bash
  source .venv/bin/activate
  python scripts/reminder_worker.py --loop
  ```
- Detach with Ctrl+b then d. Reattach later:
  ```bash
    tmux attach -t reminders
  ```

### 2. Using cron (automated execution)

Ideal for a production-like setup. Run the worker periodically with your project’s venv.

- Open your crontab:
  ```bash
  crontab -e
  ```
- Add this line to run every 15 minutes. Replace `/path/to/project` with the absolute path to this repo:
  ```cron
  */15 * * * * cd /path/to/project && source .venv/bin/activate && python scripts/reminder_worker.py --once >> _artifacts/reminders.log 2>&1
  ```

Notes
- Use absolute paths in cron; relative paths may not resolve.
- Ensure `.venv` exists and dependencies are installed before enabling the cron job.
- Logs are appended to `_artifacts/reminders.log` for inspection.

## Test & Evaluation Modes

This repo supports both offline-friendly tests and live model evaluation.

### Defaults

- The default agent for test runs is the Gmail HITL+memory agent: `email_assistant_hitl_memory_gmail`.
- **Live-first policy:** production prep and cloud Codex agents should run with real Gemini calls. Provide `GOOGLE_API_KEY` and leave `EMAIL_ASSISTANT_EVAL_MODE` unset/`0`. The offline toggles below are only for CI fallbacks or when credentials are unavailable.
- Deterministic/offline runs (use sparingly): set the following toggles when you need to avoid live calls:
  - `HITL_AUTO_ACCEPT=1`
  - `EMAIL_ASSISTANT_SKIP_MARK_AS_READ=1`
  - `EMAIL_ASSISTANT_EVAL_MODE=1` (synthesize tool calls without a live LLM)
  - Optional: `EMAIL_ASSISTANT_RECIPIENT_IN_EMAIL_ADDRESS=1` (compat mode for evaluators that expect the reply recipient in `send_email_tool.email_address` instead of your address). Off by default for live-correct Gmail behavior.
  - Optional: `EMAIL_ASSISTANT_SQLITE_TIMEOUT=60` (seconds) to extend SQLite busy handling when LangSmith tracing or judge runs create extra contention; default is 30.
  - Optional: `EMAIL_ASSISTANT_TRACE_TIMEZONE=Australia/Sydney` to change the timezone used for daily LangSmith project grouping. Defaults to Australia/Sydney.

### Notebooks

- Notebook tests set `NB_TEST_MODE=1` to skip long/online cells. Run notebooks interactively without this env to exercise full behavior.

### Running Tests

- Tool-call smoke tests (stable, offline-friendly):
  - `pytest tests/test_response.py --agent-module=email_assistant_hitl_memory_gmail -k tool_calls`
  - or `python scripts/run_tests_langsmith.py` (records to LangSmith if configured)
- Live Gemini coverage:
  - `pytest tests/test_live_smoke.py --agent-module=email_assistant_hitl_memory_gmail`
    - Uses two experiment cases to snapshot tool order + reply excerpts. Run with real Gemini creds (default) or set `EMAIL_ASSISTANT_EVAL_MODE=1` for deterministic runs. Update the baseline snapshot with `EMAIL_ASSISTANT_UPDATE_SNAPSHOTS=1` when intentional changes are made.
  - `pytest tests/test_live_hitl_spam.py --agent-module=email_assistant_hitl_memory_gmail`
    - Exercises the Question → `mark_as_spam_tool` HITL path end-to-end. Works live by default; set `EMAIL_ASSISTANT_EVAL_MODE=1` for offline CI paths.
- Runtime expectations: the live suites (particularly `test_live_smoke.py` and `test_live_hitl_spam.py`) often take 10–25 minutes end-to-end depending on Gemini latency, so configure CI timeouts accordingly.
- When `LANGSMITH_TRACING=true`, traces default to date-scoped projects `email-assistant:AGENT-YYYYMMDD` and `email-assistant:JUDGE-YYYYMMDD`. Override the assistant grouping with `EMAIL_ASSISTANT_TRACE_PROJECT` and judge grouping with `EMAIL_ASSISTANT_JUDGE_PROJECT` (or `EMAIL_ASSISTANT_JUDGE_PROJECT_OVERRIDE`). Use `EMAIL_ASSISTANT_TRACE_STAGE` / `EMAIL_ASSISTANT_TRACE_TAGS` to add extra tags (e.g. `pytest`, `tool_calls`). The runtime now also sets `GRPC_VERBOSITY=ERROR` automatically (unless you override it) so Gemini’s gRPC client doesn’t spam the console with ALTS warnings.
- Parent and child runs now log plain-text grid cells: inputs use a single-line email summary, outputs follow the two-line policy (`[reply]` etc.), and raw payloads live under metadata/extra (`tool_raw`, `email_markdown`). The formatter skips terminal `Done` tool calls so the LangSmith Output column highlights the final reply or tool action instead of the noop wrapper.
- LLM-as-judge (optional, Gemini 2.5 Flash):
  - `EMAIL_ASSISTANT_LLM_JUDGE=1` adds a post-test review powered by the Gemini judge for every `test_response.py` case. The prompt now makes the model list any missing or incorrect tool usages explicitly and clamps scores when issues exist, so flaky high scores are avoided.
  - Add `EMAIL_ASSISTANT_JUDGE_STRICT=1` to fail the test immediately when the judge's verdict is `fail`.
  - Judge inputs include `<tool_calls_summary>` and `<tool_calls_json>` blocks (ordered tool names, args, results) to keep Gemini focused on the relevant evidence.
  - The judge prompt and runner live in `src/email_assistant/eval/judges.py` and can also be consumed from LangSmith via `create_langsmith_correctness_evaluator()`.
  - Override the model with `EMAIL_ASSISTANT_JUDGE_MODEL=gemini-2.5-pro` (or another Gemini family model) if you want a different reviewer tier.
- Judge traces follow the same default project (`email-assistant:judge`). Enable tracing with `LANGSMITH_TRACING=true` so judge runs show up in the UI, or override per run via the judge env vars mentioned above.
  - When tracing is enabled, you’ll see LangSmith feedback keys for `verdict` (with the ≥0.70 threshold noted), `overall_correctness`, `content_alignment`, `tool_usage`, `notes`, any `missing_tools`, each incorrect tool (“tool” / “why”), and a bundled evidence summary—mirroring the hosted judge chips without duplicate verdict rows.
  - Guardrails: `pytest tests/test_judges.py` exercises the new tool-call summariser and post-processing clamps so CI catches accidental regressions.
  - Example (LangSmith evaluate API):
    ```python
    from langsmith import Client
    from email_assistant.eval.judges import create_langsmith_correctness_evaluator

    client = Client()
    judge = create_langsmith_correctness_evaluator()
    client.evaluate(target_fn, data="my_dataset", evaluators=[judge])
    ```
  - Feedback ordering: LangSmith displays the newest feedback first. Our judge logger posts low-priority diagnostics first and waits `EMAIL_ASSISTANT_FEEDBACK_DELAY` seconds (default `0.05`) between writes so verdict → notes → evidence appear at the top of the parent run. Set the env var to `0` to disable the pause if you do not mind LangSmith occasionally merging entries into a single timestamp.

Judge output structure example:
```
{
  "overall_correctness": 0.56,
  "verdict": "fail",
  "content_alignment": 3,
  "tool_usage": 2,
  "missing_tools": [],
  "incorrect_tool_uses": [
    {"tool": "schedule_meeting_tool", "why": "The assistant scheduled 45 min despite a 60 min request."}
  ],
  "evidence": [
    "Email: 'Could we schedule about 60 minutes sometime next week'",
    "schedule_meeting_tool: start=2025-05-22T14:00 end=2025-05-22T14:45",
    "assistant_reply: 'I've scheduled a 45-minute meeting'"
  ],
  "notes": "Match meeting durations to user constraints (requested ~60 minutes)."
}
```

Notes
- Gmail tools return mock results on missing credentials; tests assert tool-call presence, not delivery. When calendar access is unavailable, `schedule_meeting_tool` now surfaces the missing-credential reason and the Gmail agent stops retrying the tool in favor of manual follow-up. Replies still restate any requested meeting duration (e.g., “45 minutes”) so deterministic checks remain satisfied even without a booked event.
- Qualitative response grading is performed in LangStudio/LangSmith via the UI judge; the local `tests/test_judges.py` suite only sanity-checks judge invariants (summaries, score clamps).

### Quality Evaluation (UI Judge)

- Use the LangStudio/LangSmith UI judge (general LLM-as-Judge) to score correctness and tool-policy compliance.
- Each run includes a clear history with tool calls; the assistant’s final reply appears in the Output column.
- Recommended rubric (example):
  - Scheduling: `check_calendar_tool → schedule_meeting_tool → send_email_tool → Done`.
  - 90-minute availability: `check_calendar_tool → send_email_tool → Done` (no scheduling).
  - Default respond-only: `send_email_tool → Done`.
  - `send_email_tool` must include `email_id` and `email_address`; `schedule_meeting_tool` should include attendees (incl. organizer), organizer_email, and valid start/end times.
  - Exceptions: No‑reply/system “do not reply” can end with `Done` without a draft; spam may end with `mark_as_spam_tool` after HITL confirm.

### Special Cases and Behavior

- No‑reply/system notifications (e.g., sender contains `no-reply`, or body includes “do not reply”) may end with `Done` without drafting an email. This prevents loops and aligns with the “do not reply” instruction.
- Auto‑HITL mode (`HITL_AUTO_ACCEPT=1`):
  - Tool prompts that ask a Question will receive a minimal synthetic response so the flow can continue in demos/tests.
  - Deterministic fallbacks for premature `Done` are enabled only in auto/test modes to avoid loops; live HITL behavior is unchanged.
- Document-review replies now promise to tackle the technical sections and acknowledge the stated deadline (e.g., “feedback before Friday”).
- Swimming-class inquiries explicitly ask the city to reserve a spot for the sender’s daughter in one of the offered time slots.

### Gmail HITL Display and Structured Outputs

- HITL card (Gmail): For `send_email_tool`, the Agent Inbox card shows:
  - `To`: original sender (reply recipient)
  - `From`: your account (sender)
  - `Subject`: normalized with `Re:` when missing
  - Draft body content

- Final outputs (Gmail): The Gmail agent appends the following keys in the final state for external evaluators:
  - `assistant_reply`: concise summary of the reply that was sent
  - `tool_trace`: normalized conversation and tool-call trace
  - `email_markdown`: canonical email context block

### Studio Mapping (LLM-as-Judge)
- Recommended mapping for general correctness evaluations:
  - `email_markdown` → `output.email_markdown`
  - `assistant_reply` → `output.assistant_reply`
  - `tool_trace` → `output.tool_trace`
  - `raw_output_optional` → `output`
- Fallbacks (older builds):
  - `assistant_reply` → `output.messages[-1].content`
  - `tool_trace` → last AI message’s `tool_calls`

### Timezone Defaults
- The agent interprets dates/times in Australia/Melbourne by default unless the email specifies a timezone. `schedule_meeting_tool.timezone` defaults to this value.

### Experiment Dataset
- File: `datasets/experiment_gmail.jsonl`
- Register with LangSmith:
  - `PYTHONPATH=src python scripts/register_dataset_from_jsonl.py --jsonl datasets/experiment_gmail.jsonl --dataset-name standalone.email_assistant.experiments`
- Use graph: `email_assistant_hitl_memory_gmail`. Approve HITL cards or set `HITL_AUTO_ACCEPT=1` for faster runs.

### Troubleshooting

- Transient 5xx from model providers: retry the run. Tests do not require live LLMs; the UI judge is separate from local pytest.
- Recursion limit errors: in Studio, increase the `recursion_limit` in the request/config to `100` for long tool sequences.
