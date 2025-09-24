# Notebooks: Project Updates and Usage Notes

This file highlights recent project changes that affect the notebooks and how to run them.

## 2025-09-24 Notebook refresh (@notebook-refresh-ticket)

- Added live-first checklists to `agent.ipynb`, `hitl.ipynb`, `memory.ipynb`, and `evaluation.ipynb` covering `langgraph up`, Gemini env setup, and pytest commands.
- Documented reminder worker expectations (`scripts/reminder_worker.py`) and reminder env vars inside the agent notebook.
- Highlighted SQLite checkpoint/store overrides and LangGraph Studio memory viewer guidance.
- Captured the HITL resume payload cheatsheet and Gemini judge toggles inline so contributors do not need to hunt through code.


## Environment Flags

- `HITL_AUTO_ACCEPT=1`: Auto-accept HITL tool interrupts in demos/tests.
- `EMAIL_ASSISTANT_SKIP_MARK_AS_READ=1`: Skip Gmail `mark_as_read` in demos/tests.
- `EMAIL_ASSISTANT_EVAL_MODE=1`: Synthesize deterministic tool calls (no live LLM required).
- Optional: `EMAIL_ASSISTANT_RECIPIENT_IN_EMAIL_ADDRESS=1`
  - Compatibility toggle for some evaluators that expect `send_email_tool.email_address` to be the reply recipient (the original sender), not your address.
  - Leave unset for live Gmail usage (default behavior is correct for sending).
- Optional LLM judge: set `EMAIL_ASSISTANT_LLM_JUDGE=1` during pytest runs to capture Gemini 2.5 Flash correctness scores (see `src/email_assistant/eval/judges.py`). Use `EMAIL_ASSISTANT_JUDGE_STRICT=1` to fail on judge “fail”, and `EMAIL_ASSISTANT_JUDGE_MODEL` to swap the reviewer model.
- To trace judge runs in LangSmith, configure `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, and optionally `EMAIL_ASSISTANT_JUDGE_PROJECT` to pick a project name (default `email-assistant-judge`).
- Judge JSON sample:
```
{
  "overall_correctness": 0.6,
  "verdict": "fail",
  "content_alignment": 3,
  "tool_usage": 3,
  "missing_tools": [],
  "incorrect_tool_uses": [
    {"tool": "schedule_meeting_tool", "why": "Duration (45 min) did not match the ~60 min request."}
  ],
  "evidence": [
    "Email: 'Could we schedule about 60 minutes sometime in the next week'",
    "schedule_meeting_tool: start=2025-05-22T14:00 end=2025-05-22T14:45",
    "assistant_reply: 'I've scheduled a 45-minute meeting'"
  ],
  "notes": "Align meeting durations with the email ask before finalizing."
}
```
- Regular tests now auto-assign per-test tracing projects when `LANGSMITH_TRACING=true` (`AGENT-<module>-<test>[-<params>]-<YYYYMMDD Sydney>` for assistant traces, `JUDGE-<module>-<test>-<YYYYMMDD Sydney>` for judge traces). Set `EMAIL_ASSISTANT_TRACE_PROJECT` if you need to override that default.

## Gmail Agent HITL Display

For `email_assistant_hitl_memory_gmail`, the Agent Inbox card for `send_email_tool` is rendered with Gmail semantics:
- `To`: original sender (reply recipient)
- `From`: your account (sender)
- `Subject`: normalized with `Re:` if missing
- Draft body content

This is a display-only change to make approvals unambiguous and does not alter the sending logic.

## Structured Outputs (Gmail)

The Gmail agent now appends the following fields to the final state to support external evaluators (e.g., StructuredPrompt templates):
- `assistant_reply`: concise summary of the sent reply
- `tool_trace`: normalized conversation + tool-call trace
- `email_markdown`: canonical email context block

These are orthogonal to sending and safe for live runs.

## Notebook Tips

- Most notebooks assume the default agent is the Gmail HITL+memory agent: `email_assistant_hitl_memory_gmail`.
- If you run notebooks without credentials, keep the CI-style toggles set. With real Gmail, unset the eval/demo flags and configure credentials per `src/email_assistant/tools/gmail/README.md`.
- Reminders: The `agent.ipynb` notebook demonstrates the reminder flow. Ensure `REMINDER_DB_PATH` points to a writeable location (defaults to `../.local/reminders.db`).

## Live Gmail Runs

- Unset `EMAIL_ASSISTANT_EVAL_MODE` and `HITL_AUTO_ACCEPT` to review drafts in Agent Inbox.
- Keep `EMAIL_ASSISTANT_RECIPIENT_IN_EMAIL_ADDRESS` unset for correct live sending semantics.
- Ensure Gmail credentials are configured as described in `src/email_assistant/tools/gmail/README.md`.
