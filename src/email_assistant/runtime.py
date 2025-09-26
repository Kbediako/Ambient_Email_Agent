"""Runtime context utilities shared across agents."""
from __future__ import annotations

import os
from typing import Any

from langgraph.runtime import Runtime

from email_assistant.schemas import AssistantContext

_DEFAULT_TIMEZONE = os.getenv("EMAIL_ASSISTANT_TIMEZONE", "Australia/Melbourne")


def extract_runtime_metadata(
    runtime: Runtime[AssistantContext] | None,
) -> tuple[str, bool, str | None, dict[str, Any]]:
    """
    Extract timezone, evaluation-mode flag, thread identifier, and a metadata dictionary from a Runtime's context.
    
    If `runtime` or its context is absent, returns sensible defaults: timezone falls back to the module default, `eval_mode` to False, and `thread_id` to None when not present; `thread_id` is taken from `context["thread_id"]` or `context["thread_metadata"]["thread_id"]` when available.
    
    Parameters:
        runtime (Runtime[AssistantContext] | None): LangGraph runtime containing the assistant context, or `None`.
    
    Returns:
        tuple[str, bool, str | None, dict[str, Any]]: A tuple of
            - `timezone`: timezone string,
            - `eval_mode`: `true` if evaluation mode is enabled, `false` otherwise,
            - `thread_id`: thread identifier string or `None`,
            - `metadata`: dictionary with keys `"timezone"` and `"eval_mode"`.
    """

    context = (runtime.context if runtime and runtime.context else {})  # type: ignore[attr-defined]
    timezone = context.get("timezone", _DEFAULT_TIMEZONE)
    eval_mode = bool(context.get("eval_mode", False))
    thread_metadata = context.get("thread_metadata") or {}
    thread_id = context.get("thread_id") or thread_metadata.get("thread_id")
    metadata = {"timezone": timezone, "eval_mode": eval_mode}
    return timezone, eval_mode, thread_id, metadata


def runtime_thread_id(runtime: Runtime[AssistantContext] | None) -> str | None:
    """
    Get the current thread id from the provided runtime context.
    
    Returns:
        `thread_id` string from `runtime.context["thread_id"]` or `runtime.context["thread_metadata"]["thread_id"]`, or `None` if no runtime is provided or no thread id is found.
    """

    if not runtime:
        return None
    context = runtime.context or {}
    thread_metadata = context.get("thread_metadata") or {}
    return context.get("thread_id") or thread_metadata.get("thread_id")
