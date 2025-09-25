# LangSmith / LangGraph / LangChain Upgrade Research

## Compatibility Snapshot

| Library | Current Pin | Target | Headline Changes | Key Risk Cue |
| --- | --- | --- | --- | --- |
| langsmith | 0.3.45 | 0.4.30 | OTEL hybrid mode, queue/back-pressure tuning, new experiment helper | Validate batching env vars + feedback flush behaviour |
| langgraph | 0.4.8 | 0.6.x | Durability flag, runtime context schema, refreshed interrupt payload | Confirm HITL/storage integrations after API shifts |
| langchain | 0.3.25 | 0.3.27 | JSON schema + XML parser fixes, pydantic cleanup | Re-run Gemini/tool parser tests |

**LangSmith table details:**
- Queue controls now expose `run_ops_buffer_size`, `run_ops_buffer_timeout_ms`, and `max_batch_size_bytes`; confirm whether legacy `LANGSMITH_MAX_BATCH_SIZE` still takes effect or is superseded.
- `_record_feedback` depends on synchronous flushing—exercise staging with custom `LANGSMITH_RUN_ENDPOINTS` to ensure the background worker respects overrides.
- `langsmith[pytest]` continues to pull in the wider pytest stack (`pytest-asyncio`, `pytest-xdist`, etc.); check our project pins for conflicts.

**LangGraph table details:**
- `durability` replaces the deprecated `checkpoint_during` flag on `invoke`/`ainvoke`; plan to set `durability="sync"` for tool calls we must persist before HITL review.
- Runtime context schemas and `Command.PARENT` are optional adoptions. They unlock nested graphs but are not required for the current architecture—note as follow-up enhancements.
- Interrupt payloads drop the `ns`/`resumable` fields; confirm Agent Inbox serializers ignore them and add regression tests if needed.

**LangChain table details:**
- JSON-schema dereference and XML parser patches stem from upstream commit notes; make sure our structured-output & XML-based tests cover these paths.
- Keep an eye on dependency changes (`SQLAlchemy`, `PyYAML`) when regenerating `uv.lock`.

## LangSmith 0.4.x Highlights (0.3.46 → 0.4.30)

- **OTEL hybrid mode (v0.4.0)** – `OTEL_ENABLED` now duplicates traces to both OTEL + LangSmith while `OTEL_ONLY` disables Smith ingestion. Useful if we want dual telemetry but need to guard against duplicate run costs in tests.
- **Client config caching & distributed tracing (v0.4.2 / v0.4.10 / v0.4.11)** – cached request options, support for multi-project tracing, replica endpoints, and sliced traces. These change how `Client` batches run trees; our manual `_start_langsmith_run` should keep using public client APIs (no direct queue hacks).
- **Feedback data & batching**
  - v0.4.6 adds feedback attribution for OTEL evals; v0.4.10 surfaces feedback notes in `get_test_results`.
  - v0.4.15 passes `LANGSMITH_RUN_ENDPOINTS` through background workers; the PR explicitly notes compression + feedback gaps. We should re-test `_record_feedback` when a custom endpoint is set.
  - v0.4.16/0.4.21 tune queue back-pressure (lower default max batch size; allow per-client overrides) and fix disabled compression edge cases. Our CI relies on deterministic uploads—ensure env vars like `LANGSMITH_MAX_BATCH_SIZE` or new client kwargs keep queues flushing promptly when tests exit.
  - v0.4.29 introduces `Client.get_experiment_results()`, potentially simplifying our judge integrations if we stop re-implementing pagination.
- **Org/API changes (v0.4.20 & v0.4.24)** – SDK updates for org-scoped keys and corresponding README guidance. Aligns with our use of per-project tracing env vars; confirm our docs mention new key formats.
- **Compression & memory (v0.4.5, v0.4.25–0.4.28)** – zstd level tweaks, guardrails against memory growth, and selective compression when manual API keys are configured. Should reduce queue memory blowups in long smoke tests.
- **`langsmith[pytest]` extra** – still installs broad pytest ecosystem (watch for conflicts with repo pins). No new hard pins observed in `pyproject`, but confirm `pytest>=8 xdist, watchers` match our workflow.

**Repo impact checkpoints**

- `src/email_assistant/tracing.py` manually clears cached env lookups; re-run this with 0.4.x to confirm functions still exist (PR #1804 introduced multi-project support we’ll benefit from).
- Feedback ingestion is synchronous today. Evaluate whether to adopt new presigned token helpers (`create_presigned_feedback_tokens`) for bulk judge uploads or keep current flow.
- Confirm batching knobs (buffer size/timeout/max batch bytes) meet our CI timings once 0.4.x is pinned.
- Our test utilities rely on `langsmith.testing` helpers; API remained stable across releases but we should run `pytest -k tool_calls --langsmith-output` in live mode once dependencies bump.

## LangGraph 0.5/0.6 Migration Notes

### API & Runtime changes

- **Durability replaces `checkpoint_during`** – graphs now accept `durability="sync"|"async"|"exit"` via `invoke`/`ainvoke` (the old `checkpoint_during` toggle is deprecated and slated for removal in v1.0). We currently rely on default async checkpointing; plan to pass explicit durability where deterministic tests need synchronous persistence.
- **Runtime context & typed schemas** – builders accept `context_schema` (v0.6) enabling dependency injection via `Runtime[ContextSchema]`. Our agents don’t pass custom context yet, but we must ensure node signatures stay compatible if LangGraph now injects `runtime` when annotated.
- **`Interrupt` payload** – 0.6 drops `ns` / `resumable` fields on `Interrupt`. Our HITL pipeline stores the raw interrupt payload for Agent Inbox, so verify we do not depend on removed keys (current `_maybe_interrupt` only cares about `action_request`).
- **HITL primitives** – `langgraph.prebuilt.interrupt.HumanInterrupt` wrapper formalizes request schema. Evaluate whether replacing manual dictionaries with this helper simplifies our Gmail agent’s resume payloads.
- **Checkpointer refresh** – `SqliteSaver` remains but emphasises context-manager usage; `AsyncSqliteSaver` requires `aiosqlite` and raises on sync use. Our `SqliteSaver.from_conn_string` usage matches new contract, but we should document the thread-safety note (`check_same_thread=False` + lock) for reviewers.
- **Command navigation improvements** – `Command.PARENT` + new reducer guidance for subgraph updates (important if we segment Gmail flows later). Adoption is optional for our current flat graph, but capture it as a future enhancement. `Command` still accepts `resume`, `sleep`, `goto`, yet double-check dataclass field names before we craft typed annotations.
- **Tasks & deterministic replay** – documentation stresses wrapping side effects in `@task`. Our graph nodes currently make API calls directly; not mandatory today but helpful if we adopt durable replays.

### Migration plan touchpoints

1. **Baseline bump**: keep graphs unchanged, upgrade to 0.6.x, run deterministic test suite with `HITL_AUTO_ACCEPT=1` to ensure interrupts/resumes still flow.
2. **Context schema adoption**: optional follow-up to type our runtime dependencies (models, stores). Will clean up dynamic env lookups.
3. **Durability audits**: mark nodes requiring synchronous persistence (e.g. Gmail tool actions) to avoid losing state if process crashes mid-HITL.
4. **Store interface check**: confirm `BaseStore` we implement still satisfies abstract methods (signatures unchanged in 0.6, but double-check for new kwargs).

## LangChain 0.3.26 / 0.3.27

- **JSON Schema deref fix (#32088)** – resolves list-index pointer lookups in `langchain_core.utils.json_schema`. Should make structured output parsing more reliable; ensure our Gemini output schema generator doesn’t rely on previous buggy behavior (tests around `ResponsePreferences` may now pass without workarounds).
- **`pydantic` compatibility (#32162, #32080)** – removed deprecated `.schema()` calls and `model_rebuild` usage. If we subclass Pydantic models (e.g., custom email schemas), confirm we don’t depend on `model_rebuild` shims.
- **XML output parser hardening (#31859)** – safer default parser for XML scratchpad agents. Our workflows don’t use XML agents, but note in case we lean on `XmlOutputParser` for Gmail formatting.
- **Evaluator ergonomics (#31910)** – fixes `Evaluator._check_evaluation_args`; relevant if our judge wrappers rely on default evaluators.
- **Progress bar callback (#31583)** – `ProgressBarCallback` output now configurable; trivial.
- Remaining commits largely add Ruff lint rules—no runtime change.

**Verification focus**

- Re-run Gemini tool binding smoke tests. No changes observed to `tool_choice` APIs in this range, but sanity-check `model.bind_tools` still works with the `langchain-google-genai` wrapper.
- Confirm `langchain` still depends on `langchain-core==0.3.76` (already our pin) and that extra deps (`SQLAlchemy`, `PyYAML`) align with repo usage.

## Next Steps for Implementation Ticket

1. **Prototype branch**
   - Update `uv.lock` to `langsmith==0.4.30`, `langchain==0.3.27`, leave `langgraph==0.4.8` initially.
   - Run `pytest tests/test_response.py --agent-module=email_assistant_hitl_memory_gmail -k tool_calls` with generous timeout.
   - Sanity-check LangSmith uploads (ensure queues flush; watch for warnings about org-scoped API keys).

2. **LangGraph major bump**
   - Read through `docs/docs/concepts/durable_execution.md` & `types.py` for new durability defaults.
   - Update graph builders if `compile` signature enforces new kwargs.
   - Validate SQLite saver integration (run memory + reminder tests).
   - Exercise HITL flows manually or via `tests/test_live_hitl_spam.py` to ensure interrupts/resumes serialize correctly.

3. **Adopt new features (optional follow-up)** — ❗️Completed (see `dev_tickets/adopt-new-features.md` for full notes)
   - `Client.get_experiment_results()` now powers judge pagination and `_record_feedback` fallbacks.
   - HITL requests use `HumanInterrupt`; tool/LLM nodes run through `@task` helpers for durable execution.
   - Remaining follow-ups: enhance deterministic fallback planners and re-run live suites with Gmail creds.

4. **Documentation & tooling**
   - Update README/AGENTS docs with new env expectations (`OTEL_ONLY`, org-scoped keys, durability option).
   - Capture testing matrix (live vs eval mode) in ticket once prototypes pass.

## Open Questions / Risks

- Does the new LangSmith background queue ever drop feedback events when `LANGSMITH_RUN_ENDPOINTS` overrides the default host? Need to test by pointing at staging endpoint.
- Are there latent dependencies on the removed `Interrupt.ns` metadata? If downstream HITL dashboards rely on it, we may need to store thread/node IDs separately.
- LangSmith’s larger pytest extra could bring in `pytest-asyncio>=0.21`; ensure this doesn’t conflict with our pinned `pytest` config in `pyproject.toml`.
- LangGraph 0.6 emphasises async checkpointers—should we plan migration to `AsyncSqliteSaver` for the reminder worker to avoid thread locks?

---

Prepared for the upcoming upgrade ticket to unblock dependency bumps and follow-on feature adoption.
