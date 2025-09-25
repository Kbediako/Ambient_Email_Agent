# Implementation Ticket — Ambient Email Agent: LangChain/LangGraph v1.0 Upgrade

## Objective
Ship the LangChain/LangGraph v1.0 upgrade across the production Gmail HITL agent, shared tooling, and developer workflows using the research captured in `research/LangChain-LangGraph-v1-research-report.md`. The implementation should preserve current behaviour while enabling durability, Runtime/context plumbing, multi-mode streaming, and Gemini v1 conventions by default.

## References
- Research findings: `research/LangChain-LangGraph-v1-research-report.md`
- Research ticket (completed): `dev_tickets/LangChain-LangGraph-v-1-research-ticket.md`
- Proof artifacts: `research/_artifacts/`

## Scope
- Upgrade dependency pins to LangChain v1.x, LangChain Core v1.x, LangGraph v1.x, and langchain-google-genai >=2.1.12 (metadata already allows `langchain-core>=0.3.75`, so 1.x is supported) while keeping `langgraph-checkpoint-sqlite` pinned separately.
- Propagate `durability="sync"` to compiled graphs, runners, tests, scripts, and notebooks.
- Adopt `Runtime[Context]` usage (timezone, eval_mode, thread metadata) for the triage/router node and any shared helpers currently reading from `config["configurable"]`.
- Standardise multi-mode streaming (`["updates","messages","custom"]`) and surface progress events via `get_stream_writer()` where applicable.
- Align Gemini integration with v1 expectations: provider routing, structured output, tool calling, and handling of `ToolException` defaults.
- Refresh notebooks, docs, and tracing helpers to use v1 imports (`langchain.agents.create_agent`) and `.content_blocks` aware summaries.
- Ensure HITL flows continue to use `HumanInterrupt` helpers and resume cleanly after upgrade.

## Implementation Plan
### Phase 1 — Dependencies & Environment
- [ ] Update `pyproject.toml` / lockfile to LangChain v1.x, LangGraph v1.x, langchain-google-genai ≥2.1, langgraph-checkpoint-sqlite latest proved in research.
- [ ] Document new minimum Python version expectations if required by upstream packages.
- [ ] Verify local `uv pip install` / CI install paths remain valid.

### Phase 2 — Runtime & Durability Defaults
- [ ] Apply `durability="sync"` to all graph compilation helpers and high-level `invoke/stream` entrypoints (tests, scripts, notebooks).
- [ ] Update streaming loops to unpack `(mode, chunk)` and handle multi-mode lists.
- [ ] Implement the `Runtime[Context]` refactor for triage/router (and related nodes) and pass context from runners/tests.
- [ ] Ensure `prime_parent_run` / tracing metadata accepts the new context payloads.

### Phase 3 — Gemini & Tooling Updates
- [ ] Normalize Gemini provider routing via `init_chat_model(..., model_provider="google_genai")` or prefixed IDs; update docs/examples accordingly.
- [ ] Ensure structured output + tool calling continue to pass type checks; update wrappers/tests as needed.
- [ ] Audit tool nodes for `handle_tool_errors` / `ToolException` policy and adjust retries or messaging.
- [ ] Add/maintain sample tool emitting `custom` streaming events for progress logging.

### Phase 4 — UI, Docs, and Notebooks
- [ ] Refresh notebooks (`notebooks/*.ipynb`) to use new streaming API, context schema, and `create_agent` import path.
- [ ] Update README / README_LOCAL / developer docs with new env toggles (`EMAIL_ASSISTANT_TIMEZONE`, streaming expectations) and upgrade notes.
- [ ] Capture before/after screenshots or short clips demonstrating multi-mode streaming in Studio or CLI.

### Phase 5 — Validation & Rollout
- [ ] Run targeted pytest suite (see Testing Notes) against live Gemini where possible; capture logs and LangSmith links when tracing enabled.
- [ ] Regenerate or verify `tests/snapshots/live_smoke.json` and other fixtures if deterministic outputs change.
- [ ] Collect final pass/fail checklist evidence and link in this ticket (artifacts, logs, screenshots).
- [ ] Coordinate merge strategy (feature branch, rollout toggles) and note any fallback procedure.

## Acceptance Criteria
- [ ] Dependency bump merged with passing CI (unit + integration) on both offline eval and live Gemini runs.
- [ ] Email triage/router node uses `Runtime[Context]` and downstream code reads context via `runtime` APIs.
- [ ] All documented `.stream()` entrypoints request `["updates","messages","custom"]` and produce at least one `custom` event in demos/tests.
- [ ] HITL pause/resume works with sync durability on upgraded stack (log or LangSmith trace attached).
- [ ] Gemini structured output, tool-calling, and token streaming confirmed post-upgrade (artifact or test log).
- [ ] Tracing or logging utilities consume `.content_blocks` where available and fall back gracefully.
- [ ] README/docs updated to describe new defaults, env toggles, and migration notes; links back to research + this ticket included.

## Testing Notes
- Primary: `pytest tests/test_response.py --agent-module=email_assistant_hitl_memory_gmail -k tool_calls`
- Live coverage: `pytest tests/test_live_smoke.py --agent-module=email_assistant_hitl_memory_gmail`
- HITL spam flow: `pytest tests/test_live_hitl_spam.py --agent-module=email_assistant_hitl_memory_gmail`
- Optional full runner: `python scripts/run_tests_langsmith.py` (honours live vs offline env)
- When running offline/deterministic: set `EMAIL_ASSISTANT_EVAL_MODE=1`, `HITL_AUTO_ACCEPT=1`, `EMAIL_ASSISTANT_SKIP_MARK_AS_READ=1`, optionally `EMAIL_ASSISTANT_RECIPIENT_IN_EMAIL_ADDRESS=1`.

## Risks & Mitigations
- Provider routing regressions → add unit tests around `init_chat_model` helper and document env toggles.
- Streaming fan-out affecting CLI consumers → verify progress logging/backpressure in notebooks and CLI.
- Tool exception behaviour changes → ensure HITL prompts and retry logic surface actionable errors to operators.
- Notebook re-execution drift → capture updated snapshots or note cells that require live Gemini access.

## Rollout & Follow-up
- Stage rollout behind a feature branch; merge once acceptance checklist satisfied.
- After deployment, monitor Agent Inbox for duplicate sends or missed checkpoints.
- Plan subsequent work to expand `Runtime[Context]` usage across remaining nodes (memory updates, Gmail tool wrappers) as follow-up tasks if needed.

