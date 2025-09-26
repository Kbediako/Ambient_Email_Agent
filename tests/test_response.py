#!/usr/bin/env python

import os
from typing import Any, Dict, List, Optional, Tuple

import pytest
from dotenv import find_dotenv, load_dotenv
from langgraph.types import Command
from langgraph.store.memory import InMemoryStore

from email_assistant.utils import extract_tool_calls, format_messages_string
from email_assistant.tracing import invoke_with_root_run, summarize_email_for_grid
from tests.trace_utils import configure_tracing_project, configure_judge_project
from tests.agent_test_utils import (
    compile_agent,
    get_last_tool_args,
    is_eval_mode,
    load_dataset,
)
from email_assistant.eval.judges import (
    JudgeUnavailableError,
    build_tool_call_context,
    run_correctness_judge,
    serialise_messages,
)
import warnings
import json

load_dotenv(find_dotenv(), override=True)

configure_tracing_project("email-assistant-test-response")
configure_judge_project("email-assistant-judge-test-response")

EVAL_MODE_ENABLED = is_eval_mode()
HAS_GOOGLE_KEY = bool(os.getenv("GOOGLE_API_KEY"))
if not (EVAL_MODE_ENABLED or HAS_GOOGLE_KEY):
    pytest.skip(
        "Live model testing requires GOOGLE_API_KEY; set EMAIL_ASSISTANT_EVAL_MODE=1 for offline runs.",
        allow_module_level=True,
    )

from email_assistant.tools.gmail import gmail_tools  # noqa: E402  (import after env handling)

# Route Gmail interactions through the session's gmail_service fixture
@pytest.fixture(autouse=True)
def _route_gmail_through_fixture(gmail_service, monkeypatch):
    monkeypatch.setattr(gmail_tools, "mark_as_read", gmail_service.mark_as_read, raising=True)

from langsmith import testing as t


# Removed: local LLM-as-Judge setup. Use LangStudio/LangSmith UI judge for qualitative evaluation.

# Global variable for module name (used by setup helper)
AGENT_MODULE = None


@pytest.fixture(autouse=True, scope="function")
def set_agent_module(agent_module_name):
    """Set the global AGENT_MODULE for each test function.
    Using scope="function" ensures we get a fresh import for each test."""
    global AGENT_MODULE
    AGENT_MODULE = agent_module_name
    print(f"Using agent module: {AGENT_MODULE}")

    return AGENT_MODULE


def setup_assistant() -> Tuple[Any, Dict[str, Any], InMemoryStore]:
    """
    Setup the email assistant and create thread configuration.
    Returns the assistant, thread config, and store.
    """
    email_assistant, thread_config, store, _ = compile_agent(AGENT_MODULE)
    return email_assistant, thread_config, store


def _safe_log_inputs(payload: Dict[str, Any], run_id: Optional[str]) -> None:
    try:
        if run_id:
            t.log_inputs(payload, run_id=run_id)
        else:
            t.log_inputs(payload)
    except TypeError:
        if run_id:
            try:
                t.log_inputs(payload)
            except Exception:
                pass
    except Exception:
        pass


def _safe_log_outputs(payload: Dict[str, Any], run_id: Optional[str]) -> None:
    try:
        if run_id:
            t.log_outputs(payload, run_id=run_id)
        else:
            t.log_outputs(payload)
    except TypeError:
        if run_id:
            try:
                t.log_outputs(payload)
            except Exception:
                pass
    except Exception:
        pass


def extract_values(state: Any) -> Dict[str, Any]:
    """Extract values from state object regardless of type."""
    if hasattr(state, "values"):
        return state.values
    else:
        return state


# Helper previously used by judge evaluation removed.


# Key phrase groups the drafted email should address per dataset case.
# Each inner list represents interchangeable keywords; at least one keyword
# from every group must appear in the email body when running against the live
# model. The words are deliberately narrow so we can catch regressions without
# relying on an LLM judge locally.
RESPONSE_KEYWORD_GROUPS: Dict[str, List[List[str]]] = {
    "email_input_1": [["/auth/refresh"], ["/auth/validate"]],
    "email_input_4": [["tax", "planning"], ["45-minute", "45 minute"], ["tuesday", "thursday"]],
    "email_input_6": [["techconf"], ["ai/ml", "ai", "ml"], ["group discount", "discount"]],
    "email_input_7": [["review", "reviewing"], ["technical"], ["friday", "deadline"]],
    "email_input_8": [["reserve", "register"], ["daughter"], ["swimming", "class", "intermediate"]],
    "email_input_10": [["90-minute", "90 minute"], ["monday", "wednesday"], ["10 am", "3 pm"]],
    "email_input_13": [["reminder"], ["schedule", "appointment", "call"]],
    "email_input_15": [["scheduled", "schedule"], ["60-minute", "60 minute"], ["slides", "collaborate"]],
}


def assert_response_matches_criteria(email_name: str, response_text: str) -> None:
    """Ensure the drafted email covers the dataset's core requirements."""

    groups = RESPONSE_KEYWORD_GROUPS.get(email_name)
    if not groups:
        return

    text = (response_text or "").lower()
    missing_groups: List[List[str]] = []
    for group in groups:
        if not any(keyword in text for keyword in group):
            missing_groups.append(group)

    assert not missing_groups, (
        f"Draft for {email_name} is missing required phrases: {missing_groups}."
    )


def _build_raw_output_payload(values: Dict[str, Any]) -> str:
    try:
        payload = {
            "assistant_reply": values.get("assistant_reply"),
            "tool_trace": values.get("tool_trace"),
            "email_markdown": values.get("email_markdown"),
            "messages": serialise_messages(values.get("messages", [])),
        }
        return json.dumps(payload, ensure_ascii=False)
    except Exception:
        return ""


def maybe_invoke_llm_judge(
    email_name: str, values: Dict[str, Any], parent_run_id: Optional[str]
) -> None:
    if not ENABLE_LLM_JUDGE:
        return
    if not HAS_GOOGLE_KEY:
        warnings.warn("Skipping LLM judge: GOOGLE_API_KEY not configured.")
        return

    email_markdown = values.get("email_markdown", "")
    assistant_reply = values.get("assistant_reply", "")
    tool_trace = values.get("tool_trace") or format_messages_string(values.get("messages", []))
    tool_calls_summary, tool_calls_json = build_tool_call_context(values.get("messages", []))
    raw_output = _build_raw_output_payload(values)

    try:
        verdict = run_correctness_judge(
            email_markdown=email_markdown,
            assistant_reply=assistant_reply,
            tool_trace=tool_trace,
            tool_calls_summary=tool_calls_summary,
            tool_calls_json=tool_calls_json,
            raw_output_optional=raw_output,
            parent_run_id=parent_run_id,
        )
    except JudgeUnavailableError as exc:
        warnings.warn(f"LLM judge unavailable: {exc}")
        return

    try:
        _safe_log_outputs(
            {
                "judge": verdict.model_dump(),
                "judge_summary": verdict.short_summary(),
            },
            parent_run_id,
        )
    except Exception:
        pass

    if verdict.verdict == "fail":
        message = (
            f"LLM judge flagged {email_name}: {verdict.short_summary()} | notes={verdict.notes}"
        )
        if STRICT_LLM_JUDGE:
            pytest.fail(message)
        else:
            warnings.warn(message)


# Heuristic/structured judge removed from local tests.


def run_initial_stream(email_assistant: Any, email_input: Dict, thread_config: Dict) -> List[Dict]:
    """
    Collect initial non-custom stream chunks produced by the assistant for a single email input.
    
    Parameters:
        email_input (dict): The input email payload passed to the assistant's stream.
        thread_config (dict): Configuration used for the streaming call (thread-level settings).
    
    Returns:
        List[dict]: Collected stream chunk dictionaries from modes other than `"custom"`.
    """
    messages = []
    for mode, chunk in email_assistant.stream(
        {"email_input": email_input},
        config=thread_config,
        stream_mode=["updates", "messages", "custom"],
        durability="sync",
    ):
        if mode == "custom":
            continue
        messages.append(chunk)
    return messages


def run_stream_with_command(email_assistant: Any, command: Command, thread_config: Dict) -> List[Dict]:
    """
    Stream the assistant with a command and collect non-custom message chunks.
    
    Iterates the assistant's stream using the provided command and thread configuration, skipping chunks emitted with mode "custom".
    
    Parameters:
        command (Command): The command to send to the assistant.
        thread_config (Dict): Configuration for the streaming thread passed to the assistant.
    
    Returns:
        List[Dict]: Collected message chunks from the stream in order, excluding "custom" chunks.
    """
    messages = []
    for mode, chunk in email_assistant.stream(
        command,
        config=thread_config,
        stream_mode=["updates", "messages", "custom"],
        durability="sync",
    ):
        if mode == "custom":
            continue
        messages.append(chunk)
    return messages


def is_module_compatible(required_modules: List[str]) -> bool:
    """Check if current module is compatible with test.

    Returns:
        bool: True if module is compatible, False otherwise
    """
    return AGENT_MODULE in required_modules


def create_response_test_cases(dataset: Tuple[List, List, List, List, List]):
    """Create test cases for parametrized criteria evaluation with LangSmith.
    Only includes emails that require a response (triage_output == "respond").
    These are more relevant / interesting for testing tool calling / response quality.
    """

    (
        email_inputs,
        email_names,
        response_criteria_list,
        triage_outputs_list,
        expected_tool_calls,
    ) = dataset

    test_cases = []
    for email_input, email_name, criteria, triage_output, expected_calls in zip(
        email_inputs, email_names, response_criteria_list, triage_outputs_list, expected_tool_calls
    ):
        if triage_output == "respond":
            test_cases.append((email_input, email_name, criteria, expected_calls))

    print(f"Created {len(test_cases)} test cases for emails requiring responses")
    return test_cases


def pytest_generate_tests(metafunc):
    """Parametrize tests dynamically based on the selected agent module."""

    required = {"email_input", "email_name", "criteria", "expected_calls"}
    if required.issubset(set(metafunc.fixturenames)):
        agent_module_name = metafunc.config.getoption("--agent-module")
        dataset = load_dataset(agent_module_name)
        test_cases = create_response_test_cases(dataset)
        metafunc.parametrize(
            "email_input,email_name,criteria,expected_calls",
            test_cases,
        )


# Reference output key
def test_email_dataset_tool_calls(email_input, email_name, criteria, expected_calls, gmail_service):
    """Test if email processing contains expected tool calls."""
    print(f"Processing {email_name}...")

    # Set up the assistant
    email_assistant, thread_config, _ = setup_assistant()
    run_id = thread_config.get("run_id")

    # Log minimal inputs for LangSmith (safe noop if plugin disabled)
    _safe_log_inputs(
        {"module": AGENT_MODULE, "test": "test_email_dataset_tool_calls"},
        run_id,
    )

    # Run the agent
    if AGENT_MODULE in ["email_assistant", "email_assistant_hitl_memory_gmail"]:
        # Workflow agent takes email_input directly
        payload = {"email_input": email_input}
        summary = summarize_email_for_grid(email_input)

        def _invoke_agent():
            """
            Invoke the email assistant with the prepared payload and thread configuration.
            
            Returns:
                The assistant's invocation result object.
            """
            return email_assistant.invoke(
                payload,
                config=thread_config,
                durability="sync",
            )

        invoke_with_root_run(
            _invoke_agent,
            root_name=f"agent:{AGENT_MODULE}",
            input_summary=summary,
        )
    else:
        raise ValueError(f"Unsupported agent module: {AGENT_MODULE}. Only 'email_assistant' is supported in automated testing.")

    # Get the final state
    state = email_assistant.get_state(thread_config)
    values = extract_values(state)

    # Extract tool calls from messages
    extracted_tool_calls = extract_tool_calls(values["messages"])

    # Check if all expected tool calls are in the extracted ones
    missing_calls = [call for call in expected_calls if call.lower() not in extracted_tool_calls]
    # Extra calls are allowed (we only fail if expected calls are missing)
    extra_calls = [call for call in extracted_tool_calls if call.lower() not in [c.lower() for c in expected_calls]]

    if not EVAL_MODE_ENABLED:
        expected_lower = [call.lower() for call in expected_calls]
        if expected_lower:
            seq_index = 0
            for name in extracted_tool_calls:
                if name == expected_lower[seq_index]:
                    seq_index += 1
                    if seq_index == len(expected_lower):
                        break
            assert (
                seq_index == len(expected_lower)
            ), f"Tool call order mismatch for {email_name}: expected {expected_lower}, saw {extracted_tool_calls}"

        if any(call in (expected_lower or []) for call in ("send_email_tool", "write_email")):
            args = get_last_tool_args(values["messages"], "send_email_tool")
            if args is None:
                args = get_last_tool_args(values["messages"], "write_email")
            response_text = ""
            if args:
                response_text = args.get("response_text") or args.get("content") or ""
            assert response_text, f"Expected a drafted email body for {email_name} but none was found."
            assert_response_matches_criteria(email_name, response_text)

    # Log
    all_messages_str = format_messages_string(values["messages"])
    _safe_log_outputs(
        {
            "extracted_tool_calls": extracted_tool_calls,
            "missing_calls": missing_calls,
            "extra_calls": extra_calls,
            "response": all_messages_str,
        },
        run_id,
    )

    # Pass feedback key
    assert len(missing_calls) == 0

    if ENABLE_LLM_JUDGE and AGENT_MODULE == "email_assistant_hitl_memory_gmail":
        maybe_invoke_llm_judge(email_name, values, run_id)


# Reference output key
# Each test case is (email_input, email_name, criteria, expected_calls)
# Qualitative judge-based test removed. Use the LangStudio UI judge instead.
ENABLE_LLM_JUDGE = os.getenv("EMAIL_ASSISTANT_LLM_JUDGE", "").lower() in ("1", "true", "yes")
STRICT_LLM_JUDGE = os.getenv("EMAIL_ASSISTANT_JUDGE_STRICT", "").lower() in ("1", "true", "yes")
