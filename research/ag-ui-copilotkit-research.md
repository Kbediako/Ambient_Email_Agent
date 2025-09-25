# AG-UI & CopilotKit Frontend Integration Research

## Context
- Goal: evaluate AG-UI protocol and CopilotKit tooling for building a HITL-focused frontend that surfaces LangGraph email agent runs (triage, reply drafting, memory updates) and provides human approval controls.
- Target backend: `email_assistant_hitl_memory_gmail` graph with Gemini tools, SqliteSaver checkpoints, and HITL interrupts.

## AG-UI Findings
- **Protocol purpose**: AG-UI standardizes front-end ↔ agent communication through ~16 event types that cover lifecycle, text streaming, tool calls, shared state, and custom payloads (`docs/introduction.mdx`). The protocol is transport-agnostic (SSE, WebSocket, webhooks) and ships a reference HTTP implementation with TypeScript/Python SDKs (`README.md`).
- **Event plumbing**: Lifecycle events (`RunStarted`, `StepStarted`, `RunFinished`, `RunError`) let the UI mirror graph execution stages, including per-node progress (useful for our triage → respond flow). Text message events (`TextMessageStart/Content/End`) support incremental drafting, mapping directly to our reply draft streaming (`docs/concepts/events.mdx`).
- **Tool visibility**: Tool call events (`ToolCallStart/Args/End`) stream JSON argument chunks so the UI can reconstruct pending tool invocations, present approval UX, or surface errors. This matches our HITL needs around `send_email_tool`, `mark_as_spam_tool`, etc. (`docs/concepts/tools.mdx`).
- **Shared state**: AG-UI emits `STATE_SNAPSHOT` and `STATE_DELTA` (JSON Patch) to keep frontend and agent state aligned (`docs/concepts/state.mdx`). This can surface SqliteSaver-backed memory (e.g., conversation summaries, response preferences) and let users edit state via controlled patches.
- **Frontend-defined tools**: The protocol expects the UI to register tool schemas and pass them into agent runs, keeping sensitive actions (sending mail, spam labeling) under product control (`docs/concepts/tools.mdx`).
- **LangGraph alignment**: AG-UI lists LangGraph as a first-class integration; LangGraph graphs can emit protocol events via available connectors, and AG-UI middleware tolerates loosely matching event formats, easing adoption for our existing agent (`README.md`, `docs/introduction.mdx`).

## CopilotKit LangGraph Capabilities
- **AG-UI transport & SDKs**: CopilotKit builds on AG-UI, exposing React hooks and LangGraph helpers (`docs/content/docs/langgraph/index.mdx`). `LangGraphAGUIAgent` plus `add_langgraph_fastapi_endpoint` creates an AG-UI-compliant endpoint that wraps LangGraph compilation with persistence options (`docs/content/docs/langgraph/persistence/message-persistence.mdx`).
- **Shared state hooks**: `useCoAgent` / `useCoAgentState` expose real-time LangGraph state in React, syncing through AG-UI snapshots/deltas (`shared-state/index.mdx`). Input/output schema controls (CopilotKitState + LangGraph input/output annotations) filter which state fields sync back to UI—ideal for exposing memory summaries while hiding sensitive internals (`shared-state/state-inputs-outputs.mdx`).
- **Context injection**: `useCopilotReadable` adds live app context to the agent state so LangGraph nodes can reason over UI-sourced facts (e.g., selected email thread metadata, operator notes) (`agent-app-context.mdx`).
- **HITL flows**: `useLangGraphInterrupt` handles LangGraph `interrupt()` calls, rendering custom approval UIs and resolving with operator feedback. CopilotKit documents both interrupt-driven and node-driven HITL; the interrupt flow aligns with our existing human approval interceptors (`human-in-the-loop/interrupt-flow.mdx`).
- **Frontend actions as tools**: `useCopilotAction` lets React components declare actions (e.g., "approve_draft", "update_memory_preference") that are automatically bound as tools inside LangGraph nodes, matching AG-UI’s frontend-defined tool model and our HITL gating needs (`frontend-actions.mdx`).
- **Generative UI**: CopilotKit’s generative UI patterns render agent progress, structured outputs, and tool calls in custom components—useful for visualizing triage decisions, proposed email drafts, memory diff views, and calendar queries (`generative-ui/index.mdx`).
- **Persistence support**: The examples show wiring LangGraph graphs with AsyncSqlite/AsyncPostgres savers before wrapping them in AG-UI endpoints, aligning with our SqliteSaver requirement for conversation memory continuity (`message-persistence.mdx`).

## Applicability to `email_assistant_hitl_memory_gmail`
- **Run instrumentation**: Map LangGraph nodes (`triage`, `respond`, tool wrappers) to AG-UI lifecycle events. UI can show a timeline: triage decision, info-gathering steps, tool execution attempts, and final reply, leveraging event metadata for progress indicators.
- **Email context display**: Stream thread content via shared state or text events, letting UI render the original email, triage rationale, and assistant reply drafts side-by-side. CopilotKit’s generative UI can present structured email markdown and tool traces already produced by the agent.
- **HITL approvals**: Intercept `send_email_tool`, `mark_as_spam_tool`, `schedule_meeting_tool` calls as frontend actions. Provide approval modals mirroring Agent Inbox cards (showing to/from/subject, body, mark-as-spam confirmation). Use `useLangGraphInterrupt` to resume runs with operator responses.
- **Memory inspection/editing**: Mirror SqliteSaver state (conversation history, response preferences) through AG-UI state snapshots. Offer editable panels that issue controlled deltas or specialized frontend actions (e.g., "update_response_preferences") feeding back into the graph’s memory management nodes.
- **Testing & modes**: Respect environment toggles (`EMAIL_ASSISTANT_EVAL_MODE`, `HITL_AUTO_ACCEPT`, `EMAIL_ASSISTANT_SKIP_MARK_AS_READ`) by exposing UI switches or reading current settings from shared state so demos/tests stay deterministic.
- **Tool error surfacing**: Leverage AG-UI tool event stream to display calendar fallback errors or Gmail auth issues, matching existing agent safeguards (e.g., schedule_meeting_tool reporting missing credentials) and guiding manual follow-up.
- **Deployment shape**: Wrap existing FastAPI (or similar) runner with `LangGraphAGUIAgent`, enabling WebSocket/SSE streaming to the frontend. Reuse existing authentication/approval layers from Agent Inbox or extend CopilotKit’s actions with role-based gating before allowing tool completion.

## Integration Considerations & Next Steps
1. Prototype endpoint: compile `email_assistant_hitl_memory_gmail` with `LangGraphAGUIAgent` and AsyncSqliteSaver to verify AG-UI event fidelity (especially state delta size vs. memory payload).
2. Frontend shell: scaffold a React app (`npx create-ag-ui-app` or `npx copilotkit@latest init`) and connect via CopilotKit’s LangGraph bindings; define UI layouts for inbox, run timeline, and approval modals.
3. State schema audit: decide which SqliteSaver fields become part of AG-UI shared state; use input/output schema controls to avoid leaking sensitive artifacts (API keys, raw tool args).
4. HITL UX parity: replicate current Agent Inbox approval semantics (auto-accept flag, question/response handling) using `useLangGraphInterrupt` and `useCopilotAction` handlers.
5. Tool registry sync: ensure frontend tool descriptors mirror backend expectations (send email, check availability, mark spam) and include metadata for display (e.g., resolved recipients, normalized subject lines).
6. Auth & security: plan for OAuth-scoped Gmail credentials in the frontend (masking tokens, gating destructive tools) plus audit logging using AG-UI’s tool trace outputs.
7. Performance watchpoints: monitor JSON Patch frequency/size for long-running conversations; fall back to periodic snapshots if deltas drift.
8. Open questions: confirm whether existing Agent Inbox interrupt payloads align with CopilotKit resolver expectations; evaluate if additional LangGraph nodes need refactoring to emit richer state for UI rendering.

## Source References
- ag-ui-protocol/ag-ui: `README.md`, `docs/introduction.mdx`, `docs/concepts/events.mdx`, `docs/concepts/state.mdx`, `docs/concepts/tools.mdx`
- CopilotKit/CopilotKit docs: `docs/content/docs/langgraph/index.mdx`, `shared-state/index.mdx`, `shared-state/state-inputs-outputs.mdx`, `human-in-the-loop/interrupt-flow.mdx`, `frontend-actions.mdx`, `agent-app-context.mdx`, `generative-ui/index.mdx`, `persistence/message-persistence.mdx`
