# Standalone Email Assistant

The Standalone Email Assistant is a LangGraph-powered workflow that triages incoming email, coordinates tools, and drafts high-quality replies using Google's Gemini models. The project showcases how to layer human-in-the-loop checkpoints, durable memory, and Gmail automation on top of a modular graph so you can evolve an assistant from a simple responder into a production-ready agent.

Defaults:
- Timezone: `Australia/Sydney` (override via `TIMEZONE`)
- Model selection is centralized via `email_assistant.configuration.get_llm(role=...)` with env overrides: `EMAIL_ASSISTANT_MODEL`, `EMAIL_ASSISTANT_ROUTER_MODEL`, `EMAIL_ASSISTANT_TOOL_MODEL`, `EMAIL_ASSISTANT_MEMORY_MODEL`.

## Highlights
- LangGraph 0.6 graph lineup with tiered capabilities: baseline responder, HITL, persistent memory, and Gmail-native automation.
- Gemini 2.5 tool orchestration (`tool_choice="any"`) with guardrails for spam handling, no-reply detection, and deterministic evaluation modes.
- Rich tool belt including calendar scheduling, email drafting, spam labeling, and reminder automation.
- Durable execution via SQLite-backed checkpoints/stores so interrupts and memory survive process restarts.
- Optional S-Class reminder worker that escalates follow-ups via Gmail labels and background polling.

## Agent Lineup
Name | What it adds | Typical use
--- | --- | ---
`email_assistant` | Core triage → respond loop with Gemini Pro and basic tools. | Playground runs, unit demos.
`email_assistant_hitl` | Human-in-the-loop approvals before tools execute. | Agent Inbox, supervised pilots.
`email_assistant_hitl_memory` | Persistent memory with `SqliteSaver` plus response preference routing. | Long-running or personalized assistants.
`email_assistant_hitl_memory_gmail` | Gmail API integration, spam tooling, and structured outputs for evaluators. | Default production/test target.

See `AGENTS.md` for a deep dive into routing, tool behavior, and the latest feature upgrades.

## Architecture at a Glance
1. **Triage** – Classifies each email (respond, ignore/notify, spam). Fallback logic defaults to respond so flows never stall.
2. **Respond** – Drafts replies, schedules meetings, or escalates reminders using LangGraph tools.
3. **Tools & Integrations** – `send_email_tool`, `schedule_meeting_tool`, Gmail spam helper, and more. HITL wraps sensitive actions.
4. **Memory & Checkpoints** – SQLite checkpoints and stores persist agent state and preference history across runs.
5. **Human Interrupts** – HITL agents pause on tool calls using `HumanInterrupt`. Auto-accept can be enabled for demos/tests.

## Getting Started
### Prerequisites
- Python 3.11+
- `uv` (recommended) or `pip`
- Google Gemini API access via `GOOGLE_API_KEY`
- Optional: Gmail API credentials for the Gmail agent (`setup_gmail.py`)

### Installation
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
# or: pip install -e .
```

### Configure Environment
Create a `.env` file (referenced by `langgraph.json`) with at least:
```
GOOGLE_API_KEY=...
GEMINI_MODEL=gemini-2.5-pro
```

Helpful toggles (leave unset for live runs):
- `HITL_AUTO_ACCEPT=1` – auto-accept tool interrupts.
- `EMAIL_ASSISTANT_SKIP_MARK_AS_READ=1` – skip Gmail `mark_as_read` call.
- `EMAIL_ASSISTANT_EVAL_MODE=1` – deterministic, offline tool calls.
- `EMAIL_ASSISTANT_RECIPIENT_IN_EMAIL_ADDRESS=1` – evaluator compatibility mode.

### Launch LangGraph Studio or CLI
```bash
langgraph up                    # Start LangGraph Studio
# or run a graph directly
langgraph run email_assistant_hitl_memory_gmail
```
Select the graph that matches your use case (`langgraph.json` lists all available graphs). Increase `recursion_limit` (e.g., 100) in Studio for long tool sequences.

## Working with HITL
- HITL agents surface approvals through LangGraph Agent Inbox (`HumanInterrupt`).
- Resume payloads can accept, edit, or provide feedback (see README_LOCAL.md for samples).
- Set `HITL_AUTO_ACCEPT=1` for automated Acceptance during tests or demos.

## Testing & Quality Assurance
- **Live Gemini suites:**
  - `pytest tests/test_live_smoke.py --agent-module=email_assistant_hitl_memory_gmail`
  - `pytest tests/test_live_hitl_spam.py --agent-module=email_assistant_hitl_memory_gmail`
  - `pytest tests/test_response.py --agent-module=email_assistant_hitl_memory_gmail -k tool_calls`
- **Offline/deterministic runs:** set `EMAIL_ASSISTANT_EVAL_MODE=1` (and optionally `EMAIL_ASSISTANT_UPDATE_SNAPSHOTS=1`).
- `python scripts/run_tests_langsmith.py` mirrors the tool-call suite and records traces when LangSmith is configured.
- Enable Gemini 2.5 Flash judging (`EMAIL_ASSISTANT_LLM_JUDGE=1`) to score correctness/tool usage; add `EMAIL_ASSISTANT_JUDGE_STRICT=1` to fail on judge verdicts.

## Reminders & Follow-ups
The reminder worker (`scripts/reminder_worker.py`) promotes important threads:
- Configure labels and timings with `REMINDER_*` environment variables.
- Run once: `python scripts/reminder_worker.py --once`
- Continuous loop: `python scripts/reminder_worker.py --loop` (tmux/cron examples live in README_LOCAL.md)

## Repository Layout
Path | Description
--- | ---
`src/email_assistant/` | Agent graphs, tools, prompts, checkpointing, and server helpers.
`scripts/` | Utilities including dataset registration and reminder worker.
`tests/` | Live/eval pytest suites, spam flow coverage, notebooks smoke tests.
`datasets/` | LangSmith experiment datasets (`register_dataset_from_jsonl.py`).
`docs/` | Placeholder for extended documentation.
`README_LOCAL.md` | Local development recipes, dependency pin notes, testing nuances.

## Additional Resources
- `README_LOCAL.md` – local dev/test details, tmux/cron recipes, notebook tips.
- `AGENTS.md` – agent evolution, feature changelog, evaluation modes.
- `CONTRIBUTING.md` – branching, review, and testing expectations.
- `system_prompt.md` – canonical assistant instructions.

## Contributing
Pull requests are welcome; follow the workflow in `CONTRIBUTING.md`. Expect to run the live Gemini suites unless credentials are unavailable.

## License
Distributed under the MIT License. See `LICENSE` for details.
