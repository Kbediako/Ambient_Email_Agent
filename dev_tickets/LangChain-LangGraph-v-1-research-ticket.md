# Research Task — Ambient Email Agent: LangChain/LangGraph v1.0 Upgrade Readiness

## Objective

Map exactly what needs to change (and what can stay) when moving our codebase from **LangGraph 0.6.x / LangChain 0.3.x** to **v1.0**, focusing on durability, streaming, HITL, runtime/context, structured output, tool error handling, and our **`langchain-google-genai`** integration. Produce concrete diffs, risks, and a pass/fail checklist.

## Status

- [x] Research complete — findings captured in `research/LangChain-LangGraph-v1-research-report.md`.
- [x] Proof artifacts stored in `research/_artifacts/`.

## Primary references (skim first, cite in your notes)

* **LangChain Python v1 release notes** (content blocks, agents, error handling). ([LangChain Docs][1])
* **LangGraph concepts & reference** (durability modes, streaming modes, Runtime/context\_schema, StateGraph). ([LangChain AI][2])
* **LangGraph streaming modes** (“updates”, “messages”, “custom”, etc.; multi-mode). ([LangChain AI][3])
* **Human-in-the-loop via `interrupt(..)`** and Agent Inbox interop. ([LangChain AI][4])
* **SqliteSaver checkpointer (plugin package)**. ([LangChain Changelog][5])
* **`langchain-google-genai` / `ChatGoogleGenerativeAI`** (tool calling, structured output, streaming). ([LangChain Docs][6])
* **`init_chat_model` routing & provider prefixing** (force `google_genai` vs Vertex default for `gemini-*`). ([LangChain][7])

---

## Deliverables (drop these into the upgrade ticket)

1. **Findings doc** (1–2 pages): What we change vs keep, with file paths.
2. **Patch plan**: Minimal diffs for tests/runners/notebooks + 1 node migrated to Runtime/context.
3. **Proofs**:
   a) “sync” durability works for HITL/tool steps,
   b) multi-mode streaming works (`["updates","messages","custom"]`),
   c) Gemini tool-calling + structured output path works.
4. **Risk log** & **rollback notes**.
5. **Pass/Fail checklist** (below) with screenshots or LangSmith run links.

---

## Repo baseline — quick inventory (fill in)

* Current pins (paste from `pyproject.lock`/`pip freeze`).
* Where we import **`SqliteSaver`**, **HITL/interrupt**, **stream()** call sites, **`init_chat_model`**/`ChatGoogleGenerativeAI`, and any **pre-bound tools**.
* Which notebooks or demos show `create_react_agent` vs custom `StateGraph`.

> Tip: search patterns
> `rg -n "from langgraph\.checkpoint\.sqlite import SqliteSaver|stream\(|HumanInterrupt|interrupt\(|create_react_agent|create_agent|init_chat_model|ChatGoogleGenerativeAI|bind_tools"`

---

## Research workstreams & tasks

### A) Durability & side-effects

**Goal:** decide where to run with **`durability="sync"`** to guarantee crash-safe checkpoints before high-risk steps (HITL, email send, external APIs).
**Why:** v0.6 introduced explicit durability modes; v1 keeps them. `"sync"` persists state *before* the next step; `"async"` is default; `"exit"` writes only on completion. ([LangChain AI][2])

**Tasks**

* [x] List all graph/agent **invoke/stream** call sites. Propose **exact** ones to add `durability="sync"` (tests, runners, studio).
* [x] Note any nodes with multiple operations; confirm they’re wrapped as **`@task`** or split—so replays are deterministic (LangGraph recommends tasks for multi-ops nodes). ([LangChain AI][2])
* [x] Produce a tiny diff for one test (e.g., tool-calls test) adding `durability="sync"` and show a failing step still resumes after restart (screenshot or run link).

### B) Streaming UX (progress + tokens + custom)

**Goal:** standardize **multi-mode streaming** everywhere we present progress: `stream_mode=["updates","messages","custom"]`.
**Why:** “updates” = per-node state deltas; “messages” = token stream; “custom” = our custom events (progress from tools). ([LangChain AI][3])

**Tasks**

* [x] Inventory all `.stream()` usages; switch to **list** modes where applicable.
* [x] Add a **sample tool** that emits custom events via `get_stream_writer()` (document Python<3.11 async caveat). ([langgraphcn.org][8])
* [x] Verify we can co-stream progress and tokens in Studio/SDK (short screen capture).

### C) Runtime & `context_schema` (replace ad-hoc config)

**Goal:** adopt **`context_schema`** + **`Runtime`** so node code can read `runtime.context`, `runtime.store`, and write via `runtime.stream_writer`. This replaces relying on `config["configurable"]`. ([LangChain AI][9])

**Tasks**

* [x] Pick one central node (e.g., triage/router) and refactor its signature to `(state, runtime: Runtime[Context])`, defining a small `Context` `TypedDict/dataclass` (e.g., `timezone`, `eval_mode`).
* [x] Update one runner invocation to pass `context={...}` and confirm types are enforced.
* [x] Note any places we still rely on deprecated `config_schema` and mark them for follow-up (v0.6 deprecates it). ([LangChain AI][9])

### D) Human-in-the-Loop via `interrupt(...)`

**Goal:** ensure our HITL steps use the **`interrupt(...)` primitive** and match Agent Inbox schemas so approve/edit/accept flows resume cleanly. ([LangChain AI][4])

**Tasks**

* [x] Confirm we use **`interrupt(...)`** (not custom exceptions) at the user-approval boundary.
* [x] If we surface tool approvals in UI, verify payload shape matches **Agent Inbox**’s `HumanInterrupt`/`HumanResponse` contract. ([GitHub][10])
* [x] Show a single run paused at interrupt, then **resumed with `Command(resume=...)`**. Paste log/snaps. ([LangChain AI][4])

### E) `langchain-google-genai` integration (Gemini)

**Goal:** standardize on **`ChatGoogleGenerativeAI`** for Gemini, enable **tools** + **structured output** + **streaming**; fix provider routing for `init_chat_model`. ([LangChain Docs][6])

**Tasks**

* [x] Ensure dependency is current `langchain-google-genai` (2.x). Note exact version.
* [x] **Routing**: Wherever we use `init_chat_model("gemini-…")`, force the provider to **Google GenAI** (to avoid defaulting to Vertex):

  * `init_chat_model("google_genai:gemini-2.5-pro", temperature=0)` **or** `init_chat_model("gemini-2.5-pro", model_provider="google_genai")`. ([LangChain][7])
* [x] **Tool calling**: Add a minimal `@tool` and bind with `llm.bind_tools([tool])`, then verify `ai_msg.tool_calls` and round-trip a `ToolMessage`. Capture output. ([LangChain Docs][6])
* [x] **Structured output**: Create a tiny Pydantic `EmailDraft` model and call `llm.with_structured_output(EmailDraft)` to prove we get typed objects back **without a second formatting pass**. Paste result. ([LangChain Docs][6])
* [x] **Token streaming**: show token stream working (short clip or log). (Feature is supported). ([LangChain Docs][6])

### F) LangChain v1 agent/runtime changes

**Goal:** align with v1 agent ergonomics and message handling.

**What to validate**

* **Message `.content_blocks`**: adjust any tracing/summarization to leverage the standardized blocks (reasoning, citations, server-side tool calls). This helps fix “raw JSON” style inputs in review UIs. ([LangChain Docs][1])
* **Agent creation path**: in notebooks/demos, import from **`langchain.agents`** (v1) and confirm it runs on LangGraph runtime. ([LangChain Docs][11])
* **Structured output & errors**: v1 formalizes two structured-output strategies and updates error defaults: **tool execution failures raise `ToolException` by default** (prevents infinite loops). Review our tool nodes for desired `handle_tool_errors`. ([LangChain Docs][1])

**Tasks**

* [x] In one notebook, switch to `from langchain.agents import create_agent` and show parity. ([LangChain Docs][11])
* [x] In tracing, parse **`.content_blocks`** for cleaner Input/Output summaries (screenshot). ([LangChain Docs][1])
* [x] Confirm desired behavior for tool errors (e.g., fail-fast vs retry) using v1’s **`ToolException`** defaults and `handle_tool_errors`. ([LangChain Docs][1])

### G) Checkpointer & store

**Goal:** keep **`SqliteSaver`** and store usage; confirm no import path changes required (it remains a separate plugin pkg). ([LangChain Changelog][5])

**Tasks**

* [x] Note our import path is `from langgraph.checkpoint.sqlite import SqliteSaver`. Keep as-is; verify DB opened with `check_same_thread=False` if shared. ([langgraph.com.cn][12])
* [x] Prove we can list checkpoints and resume a thread mid-run (paste a code/CLI snippet). ([langgraph.theforage.cn][13])

---

## Minimal code experiments (include snippets + outputs in your notes)

1. **Durability on a real tool step**

```py
for mode, chunk in graph.stream(
    payload,
    config=thread_cfg,
    stream_mode=["updates","messages"],
    durability="sync",
):
    ...
```

Show the checkpoint id advancing on each step. ([LangChain AI][2])

2. **Runtime/context**

```py
class Ctx(TypedDict):
    timezone: str
def node(state, runtime: Runtime[Ctx]):
    tz = runtime.context["timezone"]
    ...
```

Invoke with `context={"timezone":"Australia/Sydney"}` and assert access. ([LangChain AI][14])

3. **Gemini structured output**

```py
class EmailDraft(BaseModel):
    subject: str
    body: str
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
typed = llm.with_structured_output(EmailDraft)
typed.invoke("Draft a 2-line confirmation email about Tuesday 10am.")
```

Paste the typed result. ([LangChain Docs][6])

4. **Tool calling round-trip**
   Bind a trivial tool, confirm `ai_msg.tool_calls` then send a `ToolMessage` and get a final `AIMessage`. Paste arrays/ids. ([LangChain Docs][6])

5. **Streaming multi-mode**
   Emit custom events from a tool via `get_stream_writer()` and stream `["updates","messages","custom"]`. Paste a joined log. ([LangChain AI][3])

---

## Risks / gotchas to investigate

* **Provider routing**: `init_chat_model("gemini-…")` **defaults to Vertex** unless you force `model_provider="google_genai"` or prefix `google_genai:`. Audit all `init_chat_model` calls. ([LangChain][7])
* **Async + Python <3.11**: `get_stream_writer()` custom streaming has caveats in async code on <3.11—note our runtime Python. ([langgraphcn.org][8])
* **Tool error policy**: v1 raises `ToolException` on execution failure by default—confirm our desired fallback (`handle_tool_errors`). ([LangChain Docs][1])
* **Deprecated `config_schema`**: migrate toward `context_schema`. ([LangChain AI][9])

---

## Pass/Fail checklist (attach screenshots or LangSmith links)

* [x] **Durability**: At least one run shows a pause (interrupt) → restart → **resume** with no lost state. ([LangChain AI][2])
* [x] **Streaming**: We see `updates` (per-node), `messages` (token stream), and one `custom` event in the same run. ([LangChain AI][3])
* [x] **Gemini**: Tool call round-trip works; **structured output** returns a Pydantic object; token streaming confirmed. ([LangChain Docs][6])
* [x] **Runtime**: One node uses `Runtime[Context]` and reads `runtime.context`. ([LangChain AI][14])
* [x] **Tracing**: Input/Output summaries leverage **`.content_blocks`** (no raw JSON). ([LangChain Docs][1])
* [x] **Checkpointer**: `SqliteSaver` import unchanged; checkpoint list/resume demonstrated. ([LangChain Changelog][5])

---

## Suggested upgrade branch plan (once research passes)

1. **Pin pre-release** (or stable once available):

```bash
uv pip install --pre -U langchain langchain-core langgraph langchain-google-genai langgraph-checkpoint-sqlite
# or: pip install --pre -U ...
```

(Keep `langgraph-checkpoint-sqlite` as a separate dep.) ([LangChain Changelog][5])
2\) Land the **minimal diffs** you proved (durability flags, 1 node → Runtime, streaming mode lists, notebook `create_agent` import, Gemini routing).
3\) Expand Runtime/context across nodes opportunistically.

---

[1]: https://docs.langchain.com/oss/python/releases/langchain-v1?utm_source=chatgpt.com "LangChain Python v1.0 - Docs by LangChain"
[2]: https://langchain-ai.github.io/langgraph/concepts/durable_execution/?utm_source=chatgpt.com "Overview"
[3]: https://langchain-ai.github.io/langgraph/how-tos/streaming/?utm_source=chatgpt.com "Stream outputs"
[4]: https://langchain-ai.lang.chat/langgraph/agents/human-in-the-loop/?utm_source=chatgpt.com "Human-in-the-loop"
[5]: https://changelog.langchain.com/announcements/langgraph-v0-2-increased-customization-with-new-checkpointers?utm_source=chatgpt.com "LangChain - Changelog | ✔️ LangGraph v0.2: Increased customization"
[6]: https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai?utm_source=chatgpt.com "ChatGoogleGenerativeAI - Docs by LangChain"
[7]: https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html?utm_source=chatgpt.com "init_chat_model — 🦜🔗 LangChain documentation"
[8]: https://langgraphcn.org/how-tos/streaming/?utm_source=chatgpt.com "“流式输出”"
[9]: https://langchain-ai.github.io/langgraph/reference/graphs/?utm_source=chatgpt.com "Graphs"
[10]: https://github.com/langchain-ai/agent-inbox?utm_source=chatgpt.com "GitHub - langchain-ai/agent-inbox: 📥 An inbox UX for interacting with human-in-the-loop agents."
[11]: https://docs.langchain.com/?utm_source=chatgpt.com "Overview - Docs by LangChain"
[12]: https://langgraph.com.cn/reference/checkpoints/index.html?utm_source=chatgpt.com "检查点 - LangChain 框架"
[13]: https://langgraph.theforage.cn/reference/checkpoints/?utm_source=chatgpt.com "Checkpointing"
[14]: https://langchain-ai.lang.chat/langgraph/reference/runtime/?utm_source=chatgpt.com "Runtime"
