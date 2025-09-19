import os
import sys
import json
import uuid
import glob

import pytest
from langsmith import Client
from langsmith.schemas import Dataset, Example

from tests.trace_utils import configure_tracing_project

# Add src to path to allow for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from email_assistant.email_assistant_hitl_memory_gmail import overall_workflow
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

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

# --- 1. Setup LangSmith Client and Dataset ---
DATASET_NAME = "Reminder Scenarios Evaluation v1"


def _normalize_email_input(d: dict) -> dict:
    """Ensure keys match what the Gmail graph expects."""
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
    """Runs the agent on an example from the LangSmith dataset."""
    os.environ["HITL_AUTO_ACCEPT"] = "1"

    checkpointer = MemorySaver()
    store = InMemoryStore()
    agent = overall_workflow.compile(checkpointer=checkpointer, store=store)

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = agent.invoke(example.inputs, config)

    # Basic assertion to ensure the run didn't crash
    assert result is not None
    print(f"Successfully ran agent for example ID: {example.id}")
