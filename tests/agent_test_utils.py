"""Shared helpers for agent integration tests.

These utilities centralize dataset loading and graph compilation so individual
tests do not need to duplicate the boilerplate that wires up the LangGraph
workflow. Keeping the helper isolated also makes it easier to evolve the
tests without touching production code.
"""

from __future__ import annotations

import importlib
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

# Ensure project root and src are importable when the helper runs standalone.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

DEFAULT_TEST_TIMEZONE = os.getenv("EMAIL_ASSISTANT_TIMEZONE", "Australia/Melbourne")


_GMAIL_TOOL_NAME_MAP = {
    "write_email": "send_email_tool",
    "schedule_meeting": "schedule_meeting_tool",
    "check_calendar_availability": "check_calendar_tool",
    "done": "done",
}


def is_eval_mode() -> bool:
    """Return True when EMAIL_ASSISTANT_EVAL_MODE requests deterministic mode."""

    return os.getenv("EMAIL_ASSISTANT_EVAL_MODE", "").lower() in ("1", "true", "yes")


def has_google_key() -> bool:
    """Check whether a live Gemini key is available."""

    return bool(os.getenv("GOOGLE_API_KEY"))


def _reload_if_present(module_name: str) -> None:
    """Reload a module if it was already imported during a prior test run."""

    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])


def load_dataset(agent_module_name: str) -> Tuple[List, List, List, List, List]:
    """Load dataset artifacts for the requested agent module.

    Gmail agents use the Gmail-specific dataset and normalise the expected tool
    names so assertions can compare directly against runtime tool traces.
    """

    if "gmail" in agent_module_name:
        dataset_module_name = "email_assistant.eval.email_gmail_dataset"
        _reload_if_present("email_assistant.eval.email_dataset")
    else:
        dataset_module_name = "email_assistant.eval.email_dataset"

    _reload_if_present(dataset_module_name)
    dataset_module = importlib.import_module(dataset_module_name)

    email_inputs = list(getattr(dataset_module, "email_inputs"))
    email_names = list(getattr(dataset_module, "email_names"))
    response_criteria_list = list(getattr(dataset_module, "response_criteria_list"))
    triage_outputs_list = list(getattr(dataset_module, "triage_outputs_list"))
    expected_tool_calls = [list(calls) for calls in getattr(dataset_module, "expected_tool_calls")]

    if "gmail" in agent_module_name:
        normalised_calls = []
        for calls in expected_tool_calls:
            normalised_calls.append([
                _GMAIL_TOOL_NAME_MAP.get(call.lower(), call.lower()) for call in calls
            ])
        expected_tool_calls = normalised_calls
        os.environ.setdefault("HITL_AUTO_ACCEPT", "1")
        os.environ.setdefault("EMAIL_ASSISTANT_SKIP_MARK_AS_READ", "1")
        os.environ.setdefault("EMAIL_ASSISTANT_EVAL_MODE", "0")

    return (
        email_inputs,
        email_names,
        response_criteria_list,
        triage_outputs_list,
        expected_tool_calls,
    )


def compile_agent(agent_module_name: str) -> Tuple[Any, Dict[str, Any], Optional[InMemoryStore], Any]:
    """
    Compile the agent module's workflow and prepare per-run runtime artifacts for testing.
    
    Returns:
        email_assistant: The compiled workflow callable configured with durability "sync".
        thread_config: Per-run configuration dict containing `run_id`, `configurable` (includes `thread_id`, `thread_metadata`, `timezone`, and `eval_mode`), and `recursion_limit`.
        store: An InMemoryStore instance for agents that use persisted memory, or `None` for stateless agents.
        module: The imported agent module object.
    """

    module_name = f"email_assistant.{agent_module_name}"
    _reload_if_present(module_name)
    module = importlib.import_module(module_name)

    checkpointer = MemorySaver()
    store: Optional[InMemoryStore] = InMemoryStore()
    run_id = str(uuid.uuid4())
    thread_id = f"thread-{uuid.uuid4()}"
    configurable_context: Dict[str, Any] = {
        "thread_id": thread_id,
        "thread_metadata": {"thread_id": thread_id},
        "timezone": DEFAULT_TEST_TIMEZONE,
        "eval_mode": is_eval_mode(),
    }
    thread_config = {
        "run_id": run_id,
        "configurable": configurable_context,
        "recursion_limit": 100,
    }

    if agent_module_name in {"email_assistant_hitl_memory", "email_assistant_hitl_memory_gmail"}:
        email_assistant = (
            module.overall_workflow
            .compile(checkpointer=checkpointer, store=store)
            .with_config(durability="sync")
        )
    else:
        email_assistant = (
            module.overall_workflow
            .compile(checkpointer=checkpointer)
            .with_config(durability="sync")
        )
        store = None

    return email_assistant, thread_config, store, module


def extract_tool_order(messages: Iterable[Any]) -> List[str]:
    """Return a lowercase list of tool names in the order they were invoked."""

    order: List[str] = []
    for message in messages:
        tool_calls = None
        if isinstance(message, dict):
            tool_calls = message.get("tool_calls")
        else:
            tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            continue
        for call in tool_calls:
            name = call.get("name")
            if name:
                order.append(name.lower())
    return order


def get_last_tool_args(messages: Iterable[Any], tool_name: str) -> Optional[Dict[str, Any]]:
    """Return the argument dict for the last call to the given tool name."""

    tool_name_lower = tool_name.lower()
    for message in reversed(list(messages)):
        tool_calls = None
        if isinstance(message, dict):
            tool_calls = message.get("tool_calls")
        else:
            tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            continue
        for call in reversed(tool_calls):
            name = (call.get("name") or "").lower()
            if name == tool_name_lower:
                return call.get("args") or {}
    return None
