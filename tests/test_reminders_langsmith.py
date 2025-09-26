import os
import sys
import json
import uuid
import glob

import pytest
from langsmith import Client
from langsmith.schemas import Dataset, Example

from tests.trace_utils import configure_tracing_project, configure_judge_project

# Add src to path to allow for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from email_assistant.email_assistant_hitl_memory_gmail import overall_workflow
from email_assistant.tools.gmail import gmail_tools
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from email_assistant.tracing import invoke_with_root_run, summarize_email_for_grid

EVAL_MODE_ENABLED = os.getenv("EMAIL_ASSISTANT_EVAL_MODE", "").lower() in ("1", "true", "yes")
HAS_GOOGLE_KEY = bool(os.getenv("GOOGLE_API_KEY"))
LANGSMITH_CONFIGURED = bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))

if not LANGSMITH_CONFIGURED:
    pytest.skip("LangSmith credentials are required for reminder LangSmith tests", allow_module_level=True)

if not (EVAL_MODE_ENABLED or HAS_GOOGLE_KEY):
    pytest.skip(
        "Reminders LangSmith tests require GOOGLE_API_KEY for live runs; set EMAIL_ASSISTANT_EVAL_MODE=1 to run offline.",
        allow_module_level=True,
    )

configure_tracing_project("email-assistant-test-reminders")
configure_judge_project("email-assistant-judge-test-reminders")

# --- 1. Setup LangSmith Client and Dataset ---
DATASET_NAME = "Reminder Scenarios Evaluation v1"


@pytest.fixture(autouse=True)
def _route_gmail_through_fixture(gmail_service, monkeypatch):
    """
    Patch the gmail_tools.mark_as_read function to use the provided test Gmail service's method.
    
    This replaces the module-level mark_as_read with gmail_service.mark_as_read so tests route mark-as-read operations through the fixture rather than using live credentials.
    
    Parameters:
        gmail_service: Test Gmail service fixture exposing a `mark_as_read` callable.
        monkeypatch: pytest MonkeyPatch used to set the attribute on the gmail_tools module.
    """

    monkeypatch.setattr(
        gmail_tools,
        "mark_as_read",
        gmail_service.mark_as_read,
        raising=True,
    )


def _normalize_email_input(d: dict) -> dict:
    """
    Normalize an email-like input dict so it contains an 'id' field when only 'thread_id' is present.
    
    Parameters:
        d (dict): Input mapping representing an email or thread. If `d` has no `id` but contains `thread_id`, the returned dict will include `id` set to the `thread_id` value.
    
    Returns:
        dict: The original mapping or a shallow copy with `id` populated from `thread_id` when applicable.
    """
    if isinstance(d, dict) and d.get("id") in (None, "") and d.get("thread_id"):
        d = dict(d)
        d["id"] = d["thread_id"]
    return d


def get_or_create_dataset() -> Dataset:
    """Creates a new dataset or returns an existing one."""
    client = Client()
    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        print(f"Found existing dataset: '{DATASET_NAME}'")
    except Exception:
        print(f"Creating new dataset: '{DATASET_NAME}'")
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="End-to-end tests for the reminder feature.",
        )
        # Upload examples from our evaluation files
        eval_files = glob.glob("tests/evaluation_data/reminders/*.txt")
        print(f"Found {len(eval_files)} examples to upload.")
        for file_path in sorted(eval_files):
            with open(file_path, "r") as f:
                data = json.load(f)
            inputs = {"email_input": _normalize_email_input(data)}
            # We don't have a simple expected output, so we'll leave it empty.
            # The value is in observing the trace in the experiment.
            client.create_example(inputs=inputs, outputs=None, dataset_id=dataset.id)
        print("Examples uploaded successfully.")
    return dataset


# --- 2. Get Examples and Define Test ---
dataset = get_or_create_dataset()
client = Client()
examples = list(client.list_examples(dataset_name=DATASET_NAME))


@pytest.mark.parametrize("example", examples)
def test_reminder_scenarios_on_langsmith(example: Example, gmail_service):
    """
    Execute the reminders agent on a LangSmith dataset example and assert the run completes.
    
    This test sets up in-memory persistence, compiles the agent with synchronous durability, configures a unique run/thread context (including timezone and eval mode), invokes the agent with the example's inputs, and asserts that the resulting run did not crash.
    
    Parameters:
        example (Example): A LangSmith Example whose inputs contain the email under the "email_input" key.
        gmail_service: Test Gmail service fixture used to route Gmail actions for the agent.
    """
    os.environ["HITL_AUTO_ACCEPT"] = "1"

    checkpointer = MemorySaver()
    store = InMemoryStore()
    agent = (
        overall_workflow
        .compile(checkpointer=checkpointer, store=store)
        .with_config(durability="sync")
    )

    thread_id = f"reminders-{uuid.uuid4()}"
    config = {
        "run_id": str(uuid.uuid4()),
        "configurable": {
            "thread_id": thread_id,
            "thread_metadata": {"thread_id": thread_id},
            "timezone": os.getenv("EMAIL_ASSISTANT_TIMEZONE", "Australia/Melbourne"),
            "eval_mode": EVAL_MODE_ENABLED,
        },
        "recursion_limit": 100,
    }
    summary = summarize_email_for_grid(example.inputs.get("email_input", {}))

    def _invoke_agent():
        """
        Invoke the compiled agent using the current example inputs and test configuration.
        
        Returns:
            The agent invocation result object produced by agent.invoke (the run output).
        """
        return agent.invoke(
            example.inputs,
            config,
            durability="sync",
        )

    result = invoke_with_root_run(
        _invoke_agent,
        root_name="agent:reminders",
        input_summary=summary,
    )

    # Basic assertion to ensure the run didn't crash
    assert result is not None
    print(f"Successfully ran agent for example ID: {example.id}")
