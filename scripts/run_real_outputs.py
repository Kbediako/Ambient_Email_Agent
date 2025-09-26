#!/usr/bin/env python
"""Run the email assistant on test inputs and print real outputs.

Usage:
  python scripts/run_real_outputs.py \
      --agent-module email_assistant_hitl_memory \
      --max 5 \
      --respond-only \
      --auto-accept

Notes:
  - Requires GOOGLE_API_KEY and prefers GEMINI_MODEL=gemini-2.5-pro
  - For HITL flows, use --auto-accept or set HITL_AUTO_ACCEPT=1
  - Uses the non-Gmail dataset; no Gmail auth needed.
"""
from __future__ import annotations

import argparse
import logging
import os
import importlib
import sys
from typing import Any, Dict, List

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from email_assistant.tracing import (
    AGENT_PROJECT,
    init_project,
    invoke_with_root_run,
    summarize_email_for_grid,
)


logger = logging.getLogger(__name__)


def _load_dataset(agent_module: str):
    """
    Selects and loads the email dataset corresponding to the given agent module name.
    
    Parameters:
        agent_module (str): Agent module identifier used to choose the dataset (if the string "gmail" appears, the Gmail-specific dataset is selected).
    
    Returns:
        tuple: A 3-tuple (email_inputs, email_names, triage_outputs_list) where
            - email_inputs: list of email input objects,
            - email_names: list of names/identifiers for each input,
            - triage_outputs_list: list of triage actions for each input; if the dataset does not provide this, a list of "respond" entries is returned with one entry per input.
    """
    if "gmail" in agent_module:
        ds_mod = importlib.import_module("email_assistant.eval.email_gmail_dataset")
    else:
        ds_mod = importlib.import_module("email_assistant.eval.email_dataset")
    return (
        ds_mod.email_inputs,
        ds_mod.email_names,
        getattr(ds_mod, "triage_outputs_list", ["respond"] * len(ds_mod.email_inputs)),
    )


def _compile_agent(agent_module: str):
    """
    Compile the agent module's workflow into an executable graph, attaching a memory saver and an optional in-memory store.
    
    Parameters:
        agent_module (str): Name of the agent module under the `email_assistant` package to compile.
    
    Returns:
        tuple: A pair `(graph, store)` where `graph` is the compiled workflow graph and `store` is an `InMemoryStore` instance when supported by the agent or `None` if the agent does not accept a store.
    """
    module = importlib.import_module(f"email_assistant.{agent_module}")
    checkpointer = MemorySaver()
    store = InMemoryStore()
    try:
        graph = (
            module.overall_workflow
            .compile(checkpointer=checkpointer, store=store)
            .with_config(durability="sync")
        )
    except (TypeError, AttributeError, NotImplementedError) as exc:
        logger.warning(
            "Falling back to compile without store for %s (%s: %s)",
            agent_module,
            exc.__class__.__name__,
            exc,
        )
        # Fallback for agents that don't take store
        graph = (
            module.overall_workflow
            .compile(checkpointer=checkpointer)
            .with_config(durability="sync")
        )
        store = None
    return graph, store


def _extract_tool_calls(messages: List[Any]) -> List[str]:
    names: List[str] = []
    for m in messages:
        tc = getattr(m, "tool_calls", None)
        if tc:
            for t in tc:
                names.append(str(t.get("name")).lower())
    return names


def main():
    """
    Run the email assistant agent on a dataset of test inputs and print the agent's outputs, tool usages, and diagnostic summaries.
    
    This script initializes the project, optionally sets HITL auto-accept and normalizes the GEMINI_MODEL environment variable, loads and compiles the specified agent module, and then iterates dataset examples. It supports CLI flags to choose the agent module (--agent-module), limit the number of examples (--max), restrict to triage=="respond" (--respond-only), enable auto-accept for HITL (--auto-accept), and stream agent progress (--stream). For each example it prints a header and an email summary, invokes or streams the agent with per-run thread config, captures final state, prints detected tool calls and tool message outputs, and best-effort surfaces drafted email content when available. The function prints errors and warnings but does not raise them to the caller.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-module", default="email_assistant_hitl_memory", help="Agent module under email_assistant.*")
    ap.add_argument("--max", type=int, default=5, help="Max examples to run")
    ap.add_argument("--respond-only", action="store_true", help="Only run examples with triage=='respond'")
    ap.add_argument("--auto-accept", action="store_true", help="Auto-accept HITL prompts")
    ap.add_argument(
        "--stream",
        action="store_true",
        help="Stream agent progress (prints updates/messages as they arrive)",
    )
    args = ap.parse_args()

    init_project(AGENT_PROJECT)

    if args.auto_accept:
        os.environ.setdefault("HITL_AUTO_ACCEPT", "1")

    # Ensure model is 2.5 series with normalized name
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    if ":" in model:
        model = model.split(":", 1)[1]
    if model.startswith("models/"):
        model = model.split("/", 1)[1]
    os.environ["GEMINI_MODEL"] = model

    # Load dataset and compile agent
    email_inputs, email_names, triage_list = _load_dataset(args.agent_module)
    agent, store = _compile_agent(args.agent_module)

    # Thread config with runtime context metadata
    import uuid

    count = 0
    if sys.version_info >= (3, 10):
        example_iter = zip(email_inputs, email_names, triage_list, strict=True)
    else:
        example_iter = zip(email_inputs, email_names, triage_list)

    for i, (inp, name, triage) in enumerate(example_iter):
        if args.respond_only and str(triage).lower() != "respond":
            continue
        thread_id = f"script-{uuid.uuid4()}"
        thread_config = {
            "run_id": str(uuid.uuid4()),
            "configurable": {
                "thread_id": thread_id,
                "thread_metadata": {"thread_id": thread_id},
                "timezone": os.getenv("EMAIL_ASSISTANT_TIMEZONE", "Australia/Melbourne"),
                "eval_mode": os.getenv("EMAIL_ASSISTANT_EVAL_MODE", "").lower() in ("1", "true", "yes"),
            },
            "recursion_limit": 100,
        }
        print(f"\n=== [{i}] {name} | triage={triage} ===")
        payload = {"email_input": inp}
        summary = summarize_email_for_grid(inp)

        def _invoke_agent(
            payload=payload,
            thread_config=thread_config,
            agent=agent,
        ):
            """
            Invoke the agent with the provided payload and thread configuration.
            
            Parameters:
                payload: The input payload to pass to the agent (typically a dict with the email and metadata).
                thread_config: Per-invocation configuration (e.g., run/thread ids, timezone, eval_mode).
                agent: Agent instance exposing an `invoke(payload, config=...)` method.
            
            Returns:
                The agent's invocation result (final response/state returned by `agent.invoke`).
            """
            return agent.invoke(
                payload,
                config=thread_config,
            )

        def _stream_agent(
            payload=payload,
            thread_config=thread_config,
            agent=agent,
        ):
            """
            Stream an agent's output in real time and print concise previews of each chunk.
            
            Streams updates, messages, and custom chunks from the agent; prints custom chunks verbatim and prints a truncated preview (240 characters) for other modes. Designed to be used when real-time progress is desired.
            
            Parameters:
                payload: The input payload passed to the agent stream.
                thread_config: Configuration dict used for the agent run (e.g., thread and run identifiers, timezone, eval flags).
                agent: The agent instance providing a `stream(...)` method and `get_state(...)`.
            
            Returns:
                The agent's final state object retrieved via `agent.get_state(thread_config)`.
            """
            print("Streaming agent output (updates/messages)...")
            for mode, chunk in agent.stream(
                payload,
                config=thread_config,
                stream_mode=["updates", "messages", "custom"],
            ):
                if mode == "custom":
                    print(f"[custom] {chunk}")
                    continue
                preview = str(chunk)
                if len(preview) > 240:
                    preview = preview[:240] + "..."
                print(f"[{mode}] {preview}")
            # Return final state so invoke_with_root_run captures something meaningful
            return agent.get_state(thread_config)

        final_state = None
        try:
            runner = _stream_agent if args.stream else _invoke_agent
            final_state = invoke_with_root_run(
                runner,
                root_name=f"agent:{args.agent_module}",
                input_summary=summary,
            )
        except Exception as e:
            print(f"ERROR during invoke: {e}")
            continue

        # Fetch final state messages
        values: Dict[str, Any]
        if final_state is not None and not isinstance(final_state, dict) and hasattr(final_state, "values"):
            state_obj = final_state
        else:
            try:
                state_obj = agent.get_state(thread_config)
            except (AttributeError, KeyError, RuntimeError) as exc:
                logger.warning("Failed to fetch agent state (%s: %s)", exc.__class__.__name__, exc)
                state_obj = None

        if state_obj is not None:
            if hasattr(state_obj, "values") and not isinstance(state_obj, dict):
                values = state_obj.values
            elif isinstance(state_obj, dict):
                values = state_obj
            else:
                values = {}
        else:
            values = {}

        if final_state is not None and isinstance(final_state, dict):
            # Prefer explicit result dict from invoke() when available
            values = final_state
        messages = values.get("messages", [])
        tools = _extract_tool_calls(messages)
        print(f"Tool calls: {tools}")

        # Print last assistant message tool calls and tool observations
        for m in messages:
            role = getattr(m, "type", getattr(m, "role", ""))
            if role == "tool" or getattr(m, "role", "") == "tool":
                print(f"[tool] {getattr(m, 'content', '')}")

        # Best-effort: surface write_email content if present in tool call args
        try:
            from langchain_core.messages import AIMessage
            for m in reversed(messages):
                if isinstance(m, AIMessage) and m.tool_calls:
                    for tc in m.tool_calls:
                        if str(tc.get("name")).lower() in {"write_email", "send_email_tool"}:
                            args = tc.get("args", {})
                            to = args.get("to") or args.get("email_address")
                            subject = args.get("subject")
                            content = args.get("content") or args.get("response_text")
                            print("Draft summary:")
                            print(f"  To: {to}")
                            print(f"  Subject: {subject}")
                            if content:
                                preview = str(content).strip().splitlines()
                                preview = " ".join(preview[:5])
                                print(f"  Body (preview): {preview[:300]}")
                            break
                    break
        except Exception:
            pass

        count += 1
        if count >= args.max:
            break


if __name__ == "__main__":
    main()
