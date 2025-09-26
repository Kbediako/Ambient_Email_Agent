import json
import os
from pathlib import Path
from typing import Dict, List

import pytest

from email_assistant.utils import extract_tool_calls
from email_assistant.tracing import invoke_with_root_run, summarize_email_for_grid
from tests.agent_test_utils import (
    compile_agent,
    get_last_tool_args,
    has_google_key,
    is_eval_mode,
)
from tests.trace_utils import configure_tracing_project, configure_judge_project


SMOKE_CASES = {
    "reply_only_ack": ["send_email_tool", "done"],
    "scheduling_joint_presentation": [
        "check_calendar_tool",
        "schedule_meeting_tool",
        "send_email_tool",
        "done",
    ],
}

SMOKE_KEYWORD_GROUPS = {
    "reply_only_ack": [["thanks", "appreciate"], ["update", "submitted", "review", "follow up"]],
    "scheduling_joint_presentation": [["scheduled", "schedule"], ["invite"], ["slides", "presentation"]],
}

SNAPSHOT_PATH = Path("tests/snapshots/live_smoke.json")


def _load_experiment_inputs() -> Dict[str, Dict]:
    dataset = Path("datasets/experiment_gmail.jsonl")
    with dataset.open("r", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    result = {}
    for record in records:
        metadata = record.get("metadata", {})
        case_name = metadata.get("case")
        if case_name:
            result[case_name] = record["inputs"]["email_input"]
    return result


def _extract_values(state):
    if hasattr(state, "values"):
        return state.values
    return state


def _assert_keyword_coverage(case: str, text: str) -> None:
    groups = SMOKE_KEYWORD_GROUPS.get(case)
    if not groups:
        return
    lowered = (text or "").lower()
    missing = []
    for group in groups:
        if not any(keyword in lowered for keyword in group):
            missing.append(group)
    assert not missing, f"Live reply for {case} missing keywords: {missing}"


@pytest.fixture(autouse=True)
def _fallback_to_eval_when_key_missing(monkeypatch):
    configure_tracing_project("email-assistant-test-live-smoke")
    configure_judge_project("email-assistant-judge-test-live-smoke")
    if not has_google_key() and not is_eval_mode():
        monkeypatch.setenv("EMAIL_ASSISTANT_EVAL_MODE", "1")
    os.environ.setdefault("HITL_AUTO_ACCEPT", "1")
    os.environ.setdefault("EMAIL_ASSISTANT_SKIP_MARK_AS_READ", "1")


@pytest.mark.slow
def test_live_smoke_cases(agent_module_name, gmail_service):
    """
    Run live smoke tests for Gmail agent scenarios and validate their behavior.
    
    For each predefined smoke case this test compiles and invokes the specified Gmail agent, records the sequence of tool calls, captures the assistant's reply and a short excerpt of the drafted response, and verifies that the observed tool call order contains the expected sequence. The test skips if the agent module is not a Gmail agent or if required cases are missing. In non-eval mode it asserts that a drafted response exists and that required keyword coverage is present. In eval mode it either updates a snapshot file when EMAIL_ASSISTANT_UPDATE_SNAPSHOTS is set or compares the results against the existing snapshot.
    
    Parameters:
        agent_module_name (str): Module name of the agent under test (must include "gmail" to run).
        gmail_service: pytest fixture providing access to the Gmail test service (unused directly but ensures Gmail availability).
    """
    if "gmail" not in agent_module_name:
        pytest.skip("Live smoke test is only relevant for the Gmail agent")

    cases = _load_experiment_inputs()
    missing = [case for case in SMOKE_CASES if case not in cases]
    if missing:
        pytest.skip(f"Smoke dataset missing cases: {missing}")

    results: Dict[str, Dict[str, object]] = {}
    for case_name, expected_sequence in SMOKE_CASES.items():
        email_input = cases[case_name]
        email_assistant, thread_config, _, _ = compile_agent(agent_module_name)
        payload = {"email_input": email_input}
        summary = summarize_email_for_grid(email_input)

        def _invoke_agent(
            email_assistant=email_assistant,
            payload=payload,
            thread_config=thread_config,
        ):
            """
            Invoke the email assistant with the given payload and thread configuration.
            
            Parameters:
                email_assistant: Assistant instance exposing an `invoke(payload, config=...)` method.
                payload: The input payload to pass to the assistant (typically a dict describing the email and context).
                thread_config: Configuration object or dict to apply to this assistant invocation.
            
            Returns:
                The result returned by the assistant's `invoke` call (typically the assistant's run/response object).
            """
            return email_assistant.invoke(
                payload,
                config=thread_config,
            )

        invoke_with_root_run(
            _invoke_agent,
            root_name=f"agent:{case_name}",
            input_summary=summary,
        )
        state = email_assistant.get_state(thread_config)
        values = _extract_values(state)

        messages = values.get("messages", [])
        tool_order = extract_tool_calls(messages)
        send_args = get_last_tool_args(messages, "send_email_tool") or get_last_tool_args(messages, "write_email")
        response_text = ""
        if send_args:
            response_text = send_args.get("response_text") or send_args.get("content") or ""
        assistant_reply = values.get("assistant_reply", "")

        results[case_name] = {
            "tool_order": tool_order,
            "assistant_reply": assistant_reply,
            "response_excerpt": response_text[:200],
        }

        seq_index = 0
        for name in tool_order:
            if name == expected_sequence[seq_index]:
                seq_index += 1
                if seq_index == len(expected_sequence):
                    break
        assert seq_index == len(expected_sequence), (
            f"{case_name} tool order mismatch: expected {expected_sequence}, saw {tool_order}"
        )

        if not is_eval_mode():
            assert response_text, f"Expected a drafted email body for {case_name}"
            _assert_keyword_coverage(case_name, response_text)

    if is_eval_mode():
        if os.getenv("EMAIL_ASSISTANT_UPDATE_SNAPSHOTS", "") in ("1", "true", "yes"):
            SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with SNAPSHOT_PATH.open("w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2, ensure_ascii=False)
        else:
            if not SNAPSHOT_PATH.exists():
                pytest.fail(
                    "Snapshot missing. Re-run with EMAIL_ASSISTANT_UPDATE_SNAPSHOTS=1 to record baseline."
                )
            with SNAPSHOT_PATH.open("r", encoding="utf-8") as fh:
                expected = json.load(fh)
            assert results == expected, "Deterministic smoke output deviated from snapshot"
