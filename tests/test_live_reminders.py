import logging
import os
import warnings
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, cast

import pytest

from tests.agent_test_utils import compile_agent, has_google_key, is_eval_mode
from tests.trace_utils import configure_tracing_project, configure_judge_project
from email_assistant.tracing import (
    invoke_with_root_run,
    summarize_email_for_grid,
    trace_stage,
    current_root_run_id,
)
from email_assistant.utils import format_messages_string
from email_assistant.eval.judges import (
    run_correctness_judge,
    build_tool_call_context,
    serialise_messages,
    JudgeUnavailableError,
)
from email_assistant.eval.reminder_run_judge import run_reminder_run_judge
from email_assistant.eval.composite import run_composite_judge

try:  # LangSmith logging is optional in offline runs
    from langsmith import testing as t
except Exception:  # pragma: no cover - logging disabled when LangSmith SDK missing
    t = None


def _safe_log_inputs(payload, run_id):
    """
    Log the given input payload to the LangSmith test logger when the LangSmith client is available.
    
    If a `run_id` is provided it will be associated with the logged inputs. Any errors raised while attempting to log are ignored.
    Parameters:
        payload: The input data to record (typically dict or serializable object).
        run_id (optional): Identifier of the run to associate with the logged inputs.
    """
    if not t:
        return
    try:
        if run_id:
            t.log_inputs(payload, run_id=run_id)
        else:
            t.log_inputs(payload)
    except Exception:  # pragma: no cover
        pass


def _safe_log_outputs(payload, run_id):
    """
    Log an outputs payload to LangSmith if the LangSmith client is available.
    
    Parameters:
        payload: The output data to record (typically a dict or other serializable object).
        run_id (optional): Run identifier to associate the outputs with; if omitted the logging call is made without an explicit run ID.
    
    Notes:
        Errors raised by the LangSmith client are suppressed and the function is a no-op when the LangSmith client is unavailable.
    """
    if not t:
        return
    try:
        if run_id:
            t.log_outputs(payload, run_id=run_id)
        else:
            t.log_outputs(payload)
    except Exception:  # pragma: no cover
        pass


def _extract_values(state):
    """
    Return the underlying `values` attribute of a state-like object, or the state itself if no `values` attribute exists.
    
    Parameters:
        state: An object that may expose a `values` attribute (e.g., a state wrapper) or any other value.
    
    Returns:
        The `values` attribute of `state` when present; otherwise the original `state`.
    """
    if hasattr(state, "values"):
        return state.values
    return state

@pytest.fixture(autouse=True)
def _configure_live_reminder_env(monkeypatch):
    """
    Configure environment and tracing for live reminder end-to-end tests.
    
    Sets the reminder graph logger to INFO, configures tracing and judge projects for the live-reminders test run, and sets environment variables that enable the LLM judge, loosen judge strictness, enable HITL auto-accept, and skip marking messages as read. If no Google key is available and evaluation mode is not already enabled, also enables evaluation mode.
    
    Environment variables set:
    - EMAIL_ASSISTANT_LLM_JUDGE=1
    - EMAIL_ASSISTANT_JUDGE_STRICT=0
    - HITL_AUTO_ACCEPT=1
    - EMAIL_ASSISTANT_SKIP_MARK_AS_READ=1
    - EMAIL_ASSISTANT_EVAL_MODE=1 (only when no Google key and not already in eval mode)
    """
    logging.getLogger("email_assistant.graph.reminder_nodes").setLevel(logging.INFO)
    configure_tracing_project("email-assistant-live-reminders")
    configure_judge_project("email-assistant-judge-live-reminders")
    monkeypatch.setenv("EMAIL_ASSISTANT_LLM_JUDGE", "1")
    monkeypatch.setenv("EMAIL_ASSISTANT_JUDGE_STRICT", "0")
    monkeypatch.setenv("HITL_AUTO_ACCEPT", "1")
    monkeypatch.setenv("EMAIL_ASSISTANT_SKIP_MARK_AS_READ", "1")
    if not has_google_key() and not is_eval_mode():
        monkeypatch.setenv("EMAIL_ASSISTANT_EVAL_MODE", "1")


def test_live_reminder_create_and_cancel(agent_module_name, monkeypatch, gmail_service, tmp_path_factory):
    if "gmail" not in agent_module_name:
        pytest.skip("Live reminder flow is specific to the Gmail agent")

    reminder_db = tmp_path_factory.mktemp("reminder-db") / "reminders.sqlite"
    monkeypatch.setenv("REMINDER_DB_PATH", str(reminder_db))
    monkeypatch.setenv("REMINDER_NOTIFY_EMAIL", "assistant@example.com")
    monkeypatch.setenv("REMINDER_DEFAULT_HOURS", "24")
    monkeypatch.setenv("REMINDER_JUDGE_FORCE_DECISION", "hitl")

    email_assistant, thread_config, _, module = compile_agent(agent_module_name)
    run_id = thread_config.get("run_id")

    first_email = {
        "from": "Utility Billing <billing@example.com>",
        "to": "Assistant <assistant@example.com>",
        "subject": "Bill due October 1",
        "body": (
            "Hello,\n\nYour September electricity invoice ($120.45) is due on 1 October at 6:00 PM. "
            "Avoid late fees by paying before the deadline."
        ),
        "id": "thread-reminder-invoice",
    }

    payload = {"email_input": first_email}
    summary = summarize_email_for_grid(first_email)
    _safe_log_inputs({"case": "reminder_create", "email": first_email}, run_id)

    def _invoke_initial():
        """
        Invoke the assistant with the initial reminder payload using the test thread configuration.
        
        Returns:
            The assistant invocation result (response/state object).
        """
        return email_assistant.invoke(payload, config=thread_config, durability="sync")

    artifacts: dict[str, object] = {}

    def _snapshot(reminders):
        """
        Convert a sequence of reminder objects into serializable dictionary snapshots.
        
        Parameters:
            reminders (Iterable): Iterable of reminder-like objects with attributes
                `thread_id`, `subject`, `due_at`, `reason`, and `status`.
        
        Returns:
            list[dict]: A list of dictionaries each containing:
                - `thread_id`: the reminder's thread identifier
                - `subject`: the reminder's subject
                - `due_at`: ISO 8601 string of `due_at` if it has an `isoformat()` method, otherwise an empty string
                - `reason`: the reminder's reason
                - `status`: the reminder's status
        """
        return [
            {
                "thread_id": r.thread_id,
                "subject": r.subject,
                "due_at": getattr(r.due_at, "isoformat", lambda: "")(),
                "reason": r.reason,
                "status": r.status,
            }
            for r in reminders
        ]

    def _invoke_stage(stage_name: str, stage_payload: dict, stage_summary: str) -> dict:
        """
        Run a named tracing stage that invokes the email assistant with the provided payload and returns the assistant's state values.
        
        Parameters:
            stage_name (str): Name of the tracing stage.
            stage_payload (dict): Payload to send to the assistant for this stage.
            stage_summary (str): Short summary of the inputs for tracing.
        
        Returns:
            dict: The assistant's state values after invocation.
        """
        with trace_stage(stage_name, inputs_summary=stage_summary):
            email_assistant.invoke(stage_payload, config=thread_config, durability="sync")
        return _extract_values(email_assistant.get_state(thread_config))

    def _run_flow():
        """
        Execute an end-to-end reminder flow: create a reminder, simulate HITL approval, then cancel it, while collecting artifacts.
        
        The function runs three staged agent invocations (create, approve, cancel), logs inputs/outputs for each stage, manipulates test environment to force judge decisions, and captures reminder store snapshots.
        
        Returns:
            artifacts (dict): Mapping of collected run artifacts:
                - root_run_id (str): Root tracing run identifier.
                - initial_state (dict): Agent state after the initial create invocation.
                - initial_reminders (list): Active reminders before HITL approval.
                - followup_state (dict): Agent state after the approval invocation.
                - followup_reminders (list): Active reminders after approval.
                - reply_state (dict): Agent state after the cancellation invocation.
                - reminders_after (list): Active reminders remaining after cancellation.
                - Any additional intermediate entries produced during the flow.
        """
        root_run_id = current_root_run_id()
        artifacts["root_run_id"] = root_run_id
        initial_state = _invoke_stage("agent:reminder:create", payload, summary)
        artifacts["initial_state"] = initial_state
        reminders_initial = list(module.reminder_store.iter_active_reminders())
        artifacts["initial_reminders"] = reminders_initial
        _safe_log_outputs(
            {
                "case": "reminder_hitl",
                "assistant_reply": initial_state.get("assistant_reply"),
                "tool_trace": format_messages_string(initial_state.get("messages", [])),
                "reminders": _snapshot(reminders_initial),
            },
            root_run_id,
        )

        monkeypatch.setenv("REMINDER_JUDGE_FORCE_DECISION", "approve")

        followup_email = dict(first_email)
        followup_email["id"] = "thread-reminder-invoice-approval"
        followup_payload = {"email_input": followup_email}
        followup_summary = summarize_email_for_grid(followup_email)
        _safe_log_inputs({"case": "reminder_create", "email": followup_email}, root_run_id)

        followup_state = _invoke_stage("agent:reminder:create:approve", followup_payload, followup_summary)
        artifacts["followup_state"] = followup_state
        reminders_followup = list(module.reminder_store.iter_active_reminders())
        artifacts["followup_reminders"] = reminders_followup
        _safe_log_outputs(
            {
                "case": "reminder_create",
                "assistant_reply": followup_state.get("assistant_reply"),
                "tool_trace": format_messages_string(followup_state.get("messages", [])),
                "reminders": _snapshot(reminders_followup),
            },
            root_run_id,
        )

        monkeypatch.setenv("REMINDER_JUDGE_FORCE_DECISION", "approve")

        reply_email = {
            "from": "Assistant <assistant@example.com>",
            "to": "Utility Billing <billing@example.com>",
            "subject": "Re: Bill due October 1",
            "body": "Payment processed today. Thanks for the reminder!",
            "id": "thread-reminder-invoice-approval",
        }

        reply_payload = {"email_input": reply_email}
        reply_summary = summarize_email_for_grid(reply_email)
        _safe_log_inputs({"case": "reminder_cancel", "email": reply_email}, root_run_id)

        reply_state = _invoke_stage("agent:reminder:cancel", reply_payload, reply_summary)
        artifacts["reply_state"] = reply_state
        reminders_after = list(module.reminder_store.iter_active_reminders())
        artifacts["reminders_after"] = reminders_after
        _safe_log_outputs(
            {
                "case": "reminder_cancel",
                "assistant_reply": reply_state.get("assistant_reply"),
                "tool_trace": format_messages_string(reply_state.get("messages", [])),
                "reminders": _snapshot(reminders_after),
            },
            root_run_id,
        )

        return artifacts

    invoke_with_root_run(_run_flow, root_name="agent:reminder:create", input_summary=summary)

    root_run_id = cast(str | None, artifacts.get("root_run_id"))

    initial_state = cast(Dict[str, Any], artifacts["initial_state"])  # type: ignore[arg-type]
    reminders_initial = cast(List[Any], artifacts["initial_reminders"])  # type: ignore[arg-type]
    assert not reminders_initial, "High-risk reminder should defer to HITL reviewer"

    followup_state = cast(Dict[str, Any], artifacts["followup_state"])  # type: ignore[arg-type]
    reminders_followup = cast(List[Any], artifacts["followup_reminders"])  # type: ignore[arg-type]
    assert any(r.thread_id == "thread-reminder-invoice-approval" for r in reminders_followup)
    due_delta = reminders_followup[0].due_at - datetime.now(timezone.utc)
    assert timedelta(hours=23) <= due_delta <= timedelta(hours=25)

    followup_email = dict(first_email)
    followup_email["id"] = "thread-reminder-invoice-approval"
    creation_snapshot = [
        {
            "thread_id": r.thread_id,
            "subject": r.subject,
            "due_at": getattr(r.due_at, "isoformat", lambda: "")(),
            "reason": r.reason,
            "status": r.status,
        }
        for r in reminders_followup
    ]

    reply_state = cast(Dict[str, Any], artifacts["reply_state"])  # type: ignore[arg-type]
    reminders_after = cast(List[Any], artifacts["reminders_after"])  # type: ignore[arg-type]
    assert not any(r.thread_id == "thread-reminder-invoice-approval" for r in reminders_after)

    # Secondary correctness judge for the approval run
    messages = followup_state.get("messages", [])
    tool_trace = format_messages_string(messages)
    tool_calls_summary, tool_calls_json = build_tool_call_context(messages)
    raw_payload = serialise_messages(messages)
    reminder_cleared = [
        {
            "thread_id": r.thread_id,
            "subject": r.subject,
            "due_at": getattr(r.due_at, "isoformat", lambda: "")(),
            "reason": r.reason,
            "status": r.status,
        }
        for r in reminders_after
    ]

    judge_project_override = os.getenv("EMAIL_ASSISTANT_JUDGE_PROJECT_OVERRIDE")
    if judge_project_override:
        os.environ["EMAIL_ASSISTANT_JUDGE_PROJECT"] = judge_project_override

    reminder_judge_project_override = os.getenv("EMAIL_ASSISTANT_REMINDER_JUDGE_PROJECT_OVERRIDE")
    if reminder_judge_project_override:
        os.environ["EMAIL_ASSISTANT_REMINDER_JUDGE_PROJECT"] = reminder_judge_project_override

    judge_parent_run_id = root_run_id or run_id

    try:
        correctness_verdict = run_correctness_judge(
            email_markdown=followup_state.get("email_markdown", ""),
            assistant_reply=followup_state.get("assistant_reply", ""),
            tool_trace=tool_trace,
            tool_calls_summary=tool_calls_summary,
            tool_calls_json=tool_calls_json,
            raw_output_optional=raw_payload,
            parent_run_id=judge_parent_run_id,
        )
    except JudgeUnavailableError as exc:
        warnings.warn(f"Reminder judge unavailable: {exc}")
        monkeypatch.setenv("REMINDER_JUDGE_FORCE_DECISION", "")
        return

    try:
        reminder_verdict = run_reminder_run_judge(
            email_markdown=followup_state.get("email_markdown", ""),
            assistant_reply=followup_state.get("assistant_reply", ""),
            reminder_created=creation_snapshot,
            reminder_cleared=reminder_cleared,
            sender_email=followup_email.get("from", ""),
            parent_run_id=judge_parent_run_id,
        )
    except JudgeUnavailableError as exc:
        warnings.warn(f"Reminder judge unavailable: {exc}")
        monkeypatch.setenv("REMINDER_JUDGE_FORCE_DECISION", "")
        return

    composite = run_composite_judge(
        correctness=correctness_verdict,
        reminder=reminder_verdict,
        parent_run_id=judge_parent_run_id,
        sender_email=followup_email.get("from", ""),
        email_markdown=followup_state.get("email_markdown", ""),
        email_input=followup_email,
        reminder_created=creation_snapshot,
        reminder_cleared=reminder_cleared,
    )

    _safe_log_outputs(
        {
            "case": "reminder_judge",
            "correctness_judge": correctness_verdict.model_dump(),
            "reminder_judge": reminder_verdict.model_dump(),
            "composite": composite.model_dump(),
        },
        root_run_id,
    )

    monkeypatch.setenv("REMINDER_JUDGE_FORCE_DECISION", "")
