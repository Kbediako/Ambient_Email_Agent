# LangChain / LangGraph v1.0 Upgrade — Research Report

## 0. Executive Summary
- Core agents already opt into sync durability and HumanInterrupt helpers, so the v1 upgrade focuses on propagating those defaults to the harness (tests, scripts, notebooks), adopting Runtime/context plumbing for triage + Gmail nodes, and tightening Gemini routing/structured output without rewriting the SqliteSaver or tool plan scaffolding.

## 1. Repo Baseline (Inventory)
- Python runtime: `python -V || py -V` → command missing; fallback `python3 -V` → `Python 3.12.3`.
- Dependency pins (`research/_artifacts/pip-freeze.txt`): `langchain==0.3.27`, `langchain-core==0.3.76`, `langchain-google-genai==2.1.12`, `langgraph==0.6.7`, `langgraph-checkpoint-sqlite==2.0.11`, `langsmith==0.4.30`.
- Checkpointer + store factories: `src/email_assistant/checkpointing.py:17-119` centralises `SqliteSaver` and `SqliteStore` with env overrides; tests use `MemorySaver` via `tests/agent_test_utils.py:18-116`.
- Streaming usages today: runtime code relies on `.invoke`, only notebooks and `tests/test_response.py:225-234` call `.stream(...)` with default single-mode (no durability flag).
- HITL primitives: `HumanInterrupt` + `interrupt()` sequences in `src/email_assistant/email_assistant_hitl.py:153-305`, `..._hitl_memory.py:259-450`, `..._hitl_memory_gmail.py:392-1348`; tests override via fixtures in `tests/test_spam_flow.py:44-105` and `tests/test_live_hitl_spam.py:55-156`.
- Gemini model wiring: `ChatGoogleGenerativeAI` wrapper lives in `src/email_assistant/configuration.py:2-25` (strips provider prefixes), reused across `email_assistant*.py` modules and judges (`src/email_assistant/eval/judges.py:23-147`).
- Tool binding & structured output: `llm.with_structured_output(...)` + `.bind_tools(...)` across `src/email_assistant/email_assistant.py:52-57`, `..._hitl.py:47-50`, `..._hitl_memory.py:54-57`, `..._hitl_memory_gmail.py:198-199`; older tutorial `langgraph_101.py:15-64` binds a demo tool.
- Notebook touchpoints: `notebooks/langgraph_101.ipynb` still imports `create_react_agent`; HITL and memory notebooks stream with single-mode defaults (`stream_mode="updates"`).
- Commands captured: `uv pip freeze > research/_artifacts/pip-freeze.txt`; `rg ... > research/_artifacts/rg-surfaces.txt` inventories surfaces for this report.

## 2. Proposed Changes (What to Change vs Keep)
### Durability modes
- Keep the compiled agents defaulting to sync durability (each `email_assistant*.py` module ends with `.compile(...).with_config(durability="sync")`).
- Change the dynamic compilers to propagate the same default: `tests/agent_test_utils.py:77-110`, `scripts/run_real_outputs.py:64-101`, and notebook demos currently recompile with `MemorySaver()` and never reapply `.with_config`, so live runs fall back to async durability; update helpers to call `.with_config(durability="sync")` or pass `durability="sync"` in every `invoke/stream` call.
- Add explicit `durability="sync"` overrides at the streaming loops in `tests/test_response.py:225-234` and anywhere we resume commands to guarantee checkpoints land before HITL or Gmail side-effects.
- For tutorials (`src/email_assistant/langgraph_101.py:43-64`) showcase the upgrade by compiling with sync durability and documenting why in comments.

### Streaming modes (multi-mode)
- Present runs should request multi-mode streaming; today everything uses the single default. Update `tests/test_response.py:225-234`, Notebook walkthroughs (`notebooks/hitl.ipynb`, `notebooks/memory.ipynb`), and CLI utilities to call `.stream(..., stream_mode=["updates","messages","custom"], durability="sync")` and unpack `(mode, chunk)` pairs.
- Introduce a tiny helper/tool that emits `custom` events via `get_stream_writer()` so we can surface progress in logs and Agent Inbox simulations; add this to the Gmail agent or a shared tool module for reuse.
- Document SDK/Studio expectations in README: multi-mode streaming is required for LangGraph v1 token streaming, so ensure our agent wrappers forward the `stream_mode` list when exposing `.stream` externally.

### HITL / `interrupt(...)`
- Existing flows already rely on `HumanInterrupt` (`src/email_assistant/email_assistant_hitl*.py`), but v1 expects typed resumptions. Audit every `_maybe_interrupt` helper to return `List[HumanInterrupt]` and switch resume payload parsing to the new `HumanResponse` helper once available.
- Confirm that `tests/test_spam_flow.py:44-105` and `tests/test_live_hitl_spam.py:55-156` still match the upgraded schema (status enums, request ids). Capture an example interrupt payload for the report and note any required shape changes for Agent Inbox cards.
- Ensure auto-accept demos continue to call `interrupt([...])` (Gemini HITL auto-accept path) but guard them behind sync durability checkpoints so replays do not double-send Gmail actions.

### Runtime + `context_schema`
- Replace ad-hoc `config["configurable"]` usage with `Runtime[Context]`: define a `TypedDict` (timezone, eval_mode, thread_metadata) and accept `runtime: Runtime[Context]` in triage/router nodes (`src/email_assistant/email_assistant.py:311-405`) and Gmail-specific nodes that read env toggles.
- Register the schema via `overall_workflow = StateGraph(..., context_schema=Context)` (or `.compile(context_schema=Context)`) so type hints enforce what downstream nodes expect.
- Update harnesses (`tests/agent_test_utils.py:88-108`, `scripts/run_real_outputs.py:94-118`, notebooks) to pass `context={"timezone": ..., "eval_mode": ...}` alongside thread ids; keep legacy env fallback for backwards compatibility until rollout completes.
- Leverage `runtime.store` instead of module-level globals for triage memory writes once context becomes available.

### Agent / messages (LangChain v1)
- LangChain v1 promotes message `content` as structured blocks. Update tracing (`src/email_assistant/tracing.py`) and utility helpers (`src/email_assistant/utils.py`, `tests/test_response.py` keyword assertions) to read from `.content_blocks` when present, falling back to `.content` during upgrade.
- Audit `AIMessage` creation across agents (`email_assistant.py:52-302`, `_fallback_tool_plan` etc.) to ensure we construct messages via `AIMessage(...)` or `ChatPromptTemplate` helpers instead of raw dicts—prevents v1 validation errors.
- Keep dataset tooling that inspects `.tool_calls` but adapt to new LangChain `ToolCall` objects and `ToolMessage` classes where necessary (v1 enforces typed IDs).

### `langchain-google-genai` routing + features
- Continue using our `get_llm` wrapper but add explicit provider routing: call `init_chat_model("gemini-2.5-pro", model_provider="google_genai")` or prefix `google_genai:` before stripping so we never hit Vertex defaults when LangChain v1 tightens routing.
- Adopt `.with_structured_output` + `.bind_tools` from the new `ChatGoogleGenerativeAI` API; verify streaming tokens/JSON by running the tool round-trip smoke tests in eval mode (logs to land in `research/_artifacts/gemini_structured_output.log`).
- Update README/README_LOCAL to note the v1 requirement for `LANGSMITH_TRACING=true` when using Gemini streaming so developers know the CLI command to reproduce.

### Checkpointer & store (SqliteSaver)
- Keep `src/email_assistant/checkpointing.py` as the single factory but add notes about the new `langgraph-checkpoint-sqlite>=2.x` API (e.g., `SqliteSaver.as_checkpoint_manager()` if v1 renames helpers).
- Tests currently rely on `MemorySaver` (`tests/agent_test_utils.py:71-110`); document the plan to swap to SQLite-backed checkpoints in long-running integration tests once v1 stable to cover resume semantics.
- Ensure the Gmail agent's `overall_workflow.compile(checkpointer=_DEFAULT_CHECKPOINTER)` continues to use `SqliteSaver` after v1 upgrade and that environment overrides (`EMAIL_ASSISTANT_CHECKPOINT_PATH`) remain honoured.

### Notebooks/demos
- Update `notebooks/langgraph_101.ipynb` to import `create_agent` from `langchain.agents` (v1 location) and refresh streaming cells to handle `(mode, chunk)` tuples with multi-mode lists.
- Apply the same durability + context updates to `notebooks/hitl.ipynb` and `notebooks/memory.ipynb`, including sample `interrupt` resume payloads that match the new typed schema.
- Add callouts on first worksheet cell describing required env vars (`HITL_AUTO_ACCEPT`, `EMAIL_ASSISTANT_EVAL_MODE`) and how to set `context={"timezone": ...}` when running LangGraph Studio.

## 3. Minimal Diffs (patch-ready)
```diff
--- a/tests/test_response.py
+++ b/tests/test_response.py
@@
-    for chunk in email_assistant.stream({"email_input": email_input}, config=thread_config):
-        messages.append(chunk)
+    for mode, chunk in email_assistant.stream(
+        {"email_input": email_input},
+        config=thread_config,
+        stream_mode=["updates", "messages", "custom"],
+        durability="sync",
+    ):
+        if mode == "custom":
+            continue
+        messages.append(chunk)
@@
-    for chunk in email_assistant.stream(command, config=thread_config):
-        messages.append(chunk)
+    for mode, chunk in email_assistant.stream(
+        command,
+        config=thread_config,
+        stream_mode=["updates", "messages", "custom"],
+        durability="sync",
+    ):
+        if mode == "custom":
+            continue
+        messages.append(chunk)
@@
-        def _invoke_agent():
-            return email_assistant.invoke(payload, config=thread_config)
+        def _invoke_agent():
+            return email_assistant.invoke(
+                payload,
+                config=thread_config,
+                durability="sync",
+            )
```

```diff
--- a/src/email_assistant/email_assistant.py
+++ b/src/email_assistant/email_assistant.py
@@
-from typing import Literal
+from typing import Literal, TypedDict
@@
-from langgraph.func import task
-from langgraph.graph import StateGraph, START, END
+from langgraph.func import task
+from langgraph.runtime import Runtime
+from langgraph.graph import StateGraph, START, END
@@
+class AssistantContext(TypedDict, total=False):
+    timezone: str
+    eval_mode: bool
+
@@
-@task
-def triage_router_task(state: State) -> Command[Literal["response_agent", "__end__"]]:
+@task
+def triage_router_task(
+    state: State,
+    runtime: Runtime[AssistantContext],
+) -> Command[Literal["response_agent", "__end__"]]:
@@
-    email_input = state["email_input"]
+    context = runtime.context
+    timezone = context.get("timezone", os.getenv("EMAIL_ASSISTANT_TIMEZONE", "Australia/Sydney"))
+    eval_mode = bool(context.get("eval_mode", False))
+
+    email_input = state["email_input"]
@@
-    prime_parent_run(email_input=email_input, email_markdown=email_markdown)
+    metadata = {"timezone": timezone, "eval_mode": eval_mode}
+    prime_parent_run(
+        email_input=email_input,
+        email_markdown=email_markdown,
+        metadata_update=metadata,
+    )
@@
-        prime_parent_run(email_input=email_input, email_markdown=email_markdown, outputs=output_text)
+        prime_parent_run(
+            email_input=email_input,
+            email_markdown=email_markdown,
+            outputs=output_text,
+            metadata_update=metadata,
+        )
@@
-    StateGraph(State, input_schema=StateInput)
+    StateGraph(State, input_schema=StateInput)
@@
-email_assistant = overall_workflow.compile().with_config(durability="sync")
+email_assistant = (
+    overall_workflow
+    .compile(context_schema=AssistantContext)
+    .with_config(durability="sync")
+)
```

```diff
--- a/notebooks/langgraph_101.ipynb
+++ b/notebooks/langgraph_101.ipynb
@@
-        "from langgraph.prebuilt import create_react_agent\n",
+        "from langchain.agents import create_agent\n",
```

```diff
--- a/src/email_assistant/configuration.py
+++ b/src/email_assistant/configuration.py
@@
-from langchain_google_genai import ChatGoogleGenerativeAI
+from langchain.chat_models import init_chat_model
@@
-    kwargs.setdefault("convert_system_message_to_human", False)
-    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, **kwargs)
+    kwargs.setdefault("convert_system_message_to_human", False)
+    kwargs.setdefault("temperature", temperature)
+    return init_chat_model(
+        model_name,
+        model_provider="google_genai",
+        **kwargs,
+    )
```

```diff
--- a/src/email_assistant/langgraph_101.py
+++ b/src/email_assistant/langgraph_101.py
@@
-from typing import Literal
+from typing import Literal
+
+from pydantic import BaseModel
@@
-llm = get_llm(temperature=0.0)
-model_with_tools = llm.bind_tools([write_email], tool_choice="any")
+class EmailDraft(BaseModel):
+    subject: str
+    body: str
+
+
+llm = get_llm(temperature=0.0)
+email_drafter = llm.with_structured_output(EmailDraft)
+model_with_tools = email_drafter.bind_tools([write_email], tool_choice="any")
```

```python
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


@tool
def get_contact(name: str) -> str:
    """Return a mock phone number for the contact."""
    return \"+61-490-000-123" if name.lower() == "pamela" else \"+61-000-000-000"


llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0).bind_tools([get_contact])

ai_step = llm.invoke("Please look up Pamela's phone number.")
tool_step = ToolMessage(
    tool_call_id=ai_step.tool_calls[0]["id"],
    content=get_contact.invoke(ai_step.tool_calls[0]["args"]),
)
final_reply = llm.invoke([ai_step, tool_step])
```

## 4. Proofs (Logs / Snaps)
- **Interrupt pause→resume:** `research/_artifacts/interrupt_resume.log` captures the `__interrupt__` payload, resume command, and final snapshot produced via `uv run python` on a MemorySaver-backed graph (mirrors HITL auto-accept).
- **Multi-mode streaming:** `research/_artifacts/stream_multimode.log` streams `(\"custom\", \"messages\", \"updates\")` from a demo `MessagesState` graph that emits custom events through `get_stream_writer()`.
- **Gemini structured output:** `research/_artifacts/gemini_structured_output.log` records structured-output JSON, tool-call round-trip, and streamed tokens from live Gemini 2.5 Pro (credentials loaded from `.env`).
- **Runtime context schema:** `research/_artifacts/runtime_context.log` shows a `Runtime[Context]` node accessing `runtime.context` and `runtime.stream_writer` while streaming custom events.
- **Content blocks:** `research/_artifacts/content_blocks.log` normalises a Gemini `AIMessage` into LangChain v1-style content blocks, demonstrating how tracing can consume the new schema.
- **SqliteSaver checkpointer:** `research/_artifacts/checkpointer_resume.log` lists the stored checkpoint tuple and shows `channel_values` recovered via `SqliteSaver.list(...)` / `.get(...)` on `checkpointer_demo.sqlite`.

## 5. Risks & Rollback
### Risks
- **Provider routing drift:** Without `model_provider="google_genai"`, `init_chat_model` may fall back to Vertex-hosted Gemini which lacks HITL/tool parity; enforce prefixes everywhere (scripts, notebooks, env parsing) to avoid silent regressions.
- **Tool exception policy:** LangChain v1 raises `ToolException` by default; existing Gmail auto-retries assume silent failures, so we need explicit `handle_tool_errors` guards to prevent retries from short-circuiting the workflow.
- **Streaming backpressure:** Multi-mode streaming (especially with custom events) increases channel fan-out; async clients or Studio may require Python ≥3.11 for `contextvars`, otherwise `get_stream_writer()` no-ops.
- **Runtime/context adoption:** Moving stateful reads from `config["configurable"]` to `runtime.context` risks missing keys during gradual rollout; add defaults and sanity assertions while both paths co-exist.
- **Secrets + quotas:** Structured-output demos and tests still require live Gemini credentials; document fallbacks and mark pending steps so CI without keys skips gracefully rather than failing late.

### Rollback
- Keep current `uv.lock` pins (LangChain 0.3/LangGraph 0.6) on a maintenance branch; reverting the upgrade means dropping the new Runtime/context wiring and re-pointing notebooks back to single-mode streaming.
- Preserve the existing `get_llm` helper and `MemorySaver` harness so we can flip back by removing the `init_chat_model` and context_schema diffs without touching prod behaviour.
- Retain the research artifacts + ticket notes; if upgrade destabilises HITL flows, reapply the stored lockfile and re-run the `research/_artifacts/*.log` scripts to confirm the baseline still works.

## 6. Pass/Fail Checklist
- [x] Durability pause→resume demonstrated via `research/_artifacts/interrupt_resume.log` (auto-accept HITL replay under sync semantics).
- [x] Multi-mode streaming (`updates`/`messages`/`custom`) captured in `research/_artifacts/stream_multimode.log`.
- [x] Gemini structured output + tool round-trip verified in `research/_artifacts/gemini_structured_output.log`.
- [x] Runtime `Runtime[Context]` sample confirms context + stream writer access (`research/_artifacts/runtime_context.log`).
- [x] Tracing `.content_blocks` normalisation exercised in `research/_artifacts/content_blocks.log`.
- [x] SqliteSaver list + resume confirmed in `research/_artifacts/checkpointer_resume.log`.

## 7. Appendix
- Command history: `python -V || py -V` (not found), `python3 -V` → `Python 3.12.3`, `uv pip freeze > research/_artifacts/pip-freeze.txt`, `rg -n "from langgraph..." > research/_artifacts/rg-surfaces.txt`.
- Proof scripts (all run with `uv run python`): interrupt resume, multi-mode streaming, runtime context, content blocks normalisation, SqliteSaver resume, plus the Gemini structured-output/tool-call demo (each writes to `research/_artifacts/*.log`).
- Gemini runs executed with credentials from `.env` (see `research/_artifacts/gemini_structured_output.log`); warnings about ALTS creds are expected for non-GCP environments.
- Inventory artifacts: pip freeze snapshot, surface search report, and SQLite demo database `research/_artifacts/checkpointer_demo.sqlite` accompany this report for reviewers.
