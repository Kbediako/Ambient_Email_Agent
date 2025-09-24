# Ticket: Refresh Jupyter Notebooks for Current Agent Stack

## Summary
- Update the notebooks in `notebooks/` so they reflect the LangGraph 0.6+ architecture, Gemini 2.5 tooling, and reminder/HITL safety flows now present in code.
- Replace deprecated instructions and screenshots with guidance that matches the live-first testing workflow (run with real Gemini keys before falling back to offline eval mode).
- Ensure notebook narratives surface new env toggles (`HITL_AUTO_ACCEPT`, `EMAIL_ASSISTANT_SKIP_MARK_AS_READ`, reminder worker vars) and link back to the relevant docs/tickets.

## Background
- The notebooks were last curated before the LangGraph durability overhaul, HumanInterrupt adoption, and Gmail spam guardrails described in recent tickets (`library-upgrade-ticket.md`, `adopt-new-features.md`).
- The earlier agent notebooks remain educational exercises that build toward understanding the primary Gmail HITL+memory implementation; the refresh should keep that progressive narrative intact.
- Since then, agent defaults shifted to the Gmail HITL+memory flow, test harnesses changed to live-first, and new env combinations need to be demonstrated for onboarding.

## Scope
- Audit `agent.ipynb`, `hitl.ipynb`, `memory.ipynb`, and `evaluation.ipynb` for outdated code cells, config snippets, and narrative steps.
- Sync notebook helper imports with `src/email_assistant/` (e.g., correct module paths, updated graph factory signatures, `@task`/`HumanInterrupt` usage).
- Incorporate reminder worker notes where relevant (e.g., pointer to `scripts/reminder_worker.py`, reminder env vars).
- Embed live-run checkpoints showing how to start LangGraph Studio (`langgraph up`) and run pytest suites in live mode first, with optional offline fallbacks clearly marked.
- Regenerate outputs/screenshots with the current UI where practical; redact secrets per repo policy.

## Out of Scope
- Creating new notebooks beyond the existing set.
- Refactoring agent code to better support notebooks (only notebook-side adjustments).
- Rewriting reminder worker CLI or automation scripts.

## Implementation Plan
- [ ] **Phase 1 – Discovery & Gap List**
  - [ ] Execute each notebook in a live configuration (valid `GOOGLE_API_KEY`, `GEMINI_MODEL=gemini-2.5-pro`) and note failures, deprecations, or missing steps. (pending access to live Gemini creds — owner: JR, target: 2025-09-30)
  - [x] Compare notebook import paths and helpers against current module structure to catalog required edits. (diff review complete)
- [x] **Phase 2 – Notebook Updates**
  - [x] Update code cells to call the latest agent constructors/tool registries, including memory/store paths and HITL resume payload helpers. (inlined live-first checklists & env guidance)
  - [x] Refresh markdown guidance to emphasize live-first testing, new env toggles, and spam/reminder safety notes.
  - [ ] Regenerate critical outputs (tool traces, HITL cards) so screenshots/JSON snippets match current behaviour. (pending live rerun for updated imagery — owner: JR, target: 2025-09-30)
- [ ] **Phase 3 – Validation & Handover**
  - [ ] Rerun notebooks end-to-end in live mode; capture verification notes in this ticket. (blocked locally without live credentials — owner: JR, target: 2025-09-30)
  - [x] Optional: run with `EMAIL_ASSISTANT_EVAL_MODE=1`, `HITL_AUTO_ACCEPT=1`, `EMAIL_ASSISTANT_SKIP_MARK_AS_READ=1` to demonstrate offline determinism. (pytest tool_calls suite in eval mode)
  - [x] Update `README.md` / `README_LOCAL.md` / `UPDATES.md` with references to the refreshed notebooks and live-testing workflow. (UPDATES.md refreshed; README already covered live-first policies)

## Acceptance Criteria
- `agent.ipynb`, `hitl.ipynb`, `memory.ipynb`, `evaluation.ipynb` execute without modification when run live with required env vars and current dependencies.
- Notebook markdown calls out live-first testing steps and clearly labels any offline/eval shortcuts as secondary.
- All referenced env toggles, reminder configuration, and spam safeguards align with the latest codebase behaviour.
- `UPDATES.md` (or equivalent changelog) records the refresh date and points to this ticket.

## Testing Notes
- Offline verification: `pytest tests/test_response.py --agent-module email_assistant_hitl_memory_gmail -k tool_calls` with `EMAIL_ASSISTANT_EVAL_MODE=1`, `HITL_AUTO_ACCEPT=1`, `EMAIL_ASSISTANT_SKIP_MARK_AS_READ=1` (Gemini credentials unavailable in this environment). Note: judge strict failure expected while live Gmail scheduling remains unavailable.
- Preferred verification order: execute notebooks in live mode first (`EMAIL_ASSISTANT_EVAL_MODE` unset) to observe real Gemini interactions; document results in the ticket.
- After live validation, optionally rerun in offline (`EMAIL_ASSISTANT_EVAL_MODE=1`, `HITL_AUTO_ACCEPT=1`, `EMAIL_ASSISTANT_SKIP_MARK_AS_READ=1`, `EMAIL_ASSISTANT_RECIPIENT_IN_EMAIL_ADDRESS=1`) and note any divergent tool sequences.
- If notebook updates rely on reminder worker examples, invoke `python scripts/reminder_worker.py --once` in live mode to confirm instructions remain accurate.

## Risks & Mitigations
- **Live execution cost/time**: Batch runs during low-traffic hours and reuse cached auth tokens where allowed; retain offline mode instructions for contributors without keys.
- **Stale screenshots**: Store regenerated assets under version control and date-stamp them to ease future refreshes.
- **Notebook drift**: Add a maintenance note in `UPDATES.md` recommending periodic re-runs (e.g., every major dependency bump).

## Dependencies
- Valid Gemini credentials and LangSmith tracing access (if demonstrating trace capture).
- Availability of Gmail sandbox/test credentials for HITL+memory demos.
- Coordination with docs maintainers to publish updated screenshots or cross-links if required.

## References
- `dev_tickets/library-upgrade-ticket.md`
- `dev_tickets/adopt-new-features.md`
- `README.md`, `README_LOCAL.md`, `AGENTS.md`
- `notebooks/UPDATES.md`
