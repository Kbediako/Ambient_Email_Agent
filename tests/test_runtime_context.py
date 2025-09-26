from types import SimpleNamespace

from email_assistant import email_assistant
from email_assistant.runtime import extract_runtime_metadata, runtime_thread_id


def test_extract_runtime_metadata_defaults():
    timezone, eval_mode, thread_id, metadata = extract_runtime_metadata(None)
    assert timezone == "Australia/Melbourne"
    assert eval_mode is False
    assert thread_id is None
    assert metadata == {"timezone": timezone, "eval_mode": False}


def test_triage_router_uses_runtime_context(monkeypatch):
    captured: dict[str, object] = {}

    def fake_prime_parent_run(**kwargs):  # type: ignore[no-untyped-def]
        """
        Capture `metadata_update` and `thread_id` from keyword arguments into the outer `captured` dict for testing.
        
        Parameters:
            **kwargs: Keyword arguments expected to include:
                metadata_update (dict): Runtime metadata to capture and store under `captured["metadata"]`.
                thread_id (str | None): Thread identifier to capture and store under `captured["thread_id"]`.
        """
        captured["metadata"] = kwargs.get("metadata_update")
        captured["thread_id"] = kwargs.get("thread_id")

    monkeypatch.setattr(email_assistant, "prime_parent_run", fake_prime_parent_run)

    class Result(SimpleNamespace):
        classification: str

    monkeypatch.setattr(
        email_assistant,
        "llm_router",
        SimpleNamespace(invoke=lambda *_args, **_kwargs: Result(classification="respond")),
    )

    state = {
        "email_input": {
            "author": "Alice <alice@example.com>",
            "to": "Bob <bob@example.com>",
            "subject": "Hello",
            "email_thread": "Body",
        }
    }
    runtime = SimpleNamespace(
        context={
            "timezone": "America/New_York",
            "eval_mode": True,
            "thread_id": "thread-ctx",
        }
    )

    command = email_assistant.triage_router_task.func(state, runtime)

    assert captured["metadata"] == {"timezone": "America/New_York", "eval_mode": True}
    assert captured["thread_id"] == "thread-ctx"
    assert command.goto == "response_agent"
    assert command.update["classification_decision"] == "respond"


def test_runtime_thread_id_prefers_context_thread():
    runtime = SimpleNamespace(context={"thread_id": "primary", "thread_metadata": {"thread_id": "secondary"}})
    assert runtime_thread_id(runtime) == "primary"
