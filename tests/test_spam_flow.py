import os
import types
import pytest

from langgraph.store.memory import InMemoryStore
from langchain_core.messages import AIMessage


@pytest.fixture(autouse=True)
def ensure_no_auto_accept(monkeypatch):
    # Disable global auto-accept unless test sets behavior explicitly
    monkeypatch.setenv("HITL_AUTO_ACCEPT", "0")


def _make_state_with_question(email_id: str = "abc123"):
    # Minimal state with last AI message proposing a Question tool
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "Question", "args": {"content": "Is this spam?"}, "id": "q1"}],
    )
    state = {
        "email_input": {
            "from": "sender@example.com",
            "to": "You <me@example.com>",
            "subject": "Test message",
            "body": "Suspicious content...",
            "id": email_id,
        },
        "messages": [ai],
        "classification_decision": "respond",
    }
    return state


def test_gmail_spam_flow_moves_to_spam(monkeypatch, gmail_service):
    import email_assistant.email_assistant_hitl_memory_gmail as mod

    store = InMemoryStore()
    state = _make_state_with_question("abc123")

    # Monkeypatch _maybe_interrupt to first return a spam response, then accept confirmation
    calls = {"n": 0}

    def fake_interrupt(requests):
        calls["n"] += 1
        if calls["n"] == 1:
            return [{"type": "response", "args": "This is spam"}]
        return [{"type": "accept", "args": {}}]

    monkeypatch.setattr(mod, "_maybe_interrupt", fake_interrupt)

    cmd = mod.interrupt_handler(state, store)
    # Should end the workflow after moving to spam
    assert cmd.goto == mod.END
    # And include a tool message mentioning moved to Spam
    tool_msgs = [m for m in cmd.update["messages"] if (isinstance(m, dict) and m.get("role") == "tool")]
    assert any("Moved message abc123 to Spam" in (m.get("content") or "") for m in tool_msgs)


def test_gmail_spam_flow_non_spam_feedback_no_spam_action(monkeypatch, gmail_service):
    import email_assistant.email_assistant_hitl_memory_gmail as mod

    store = InMemoryStore()
    state = _make_state_with_question("xyz789")

    # First interrupt returns general feedback; no spam keywords
    def fake_interrupt(requests):
        return [{"type": "response", "args": "Not spam, please proceed."}]

    monkeypatch.setattr(mod, "_maybe_interrupt", fake_interrupt)

    # Spy by wrapping service method to ensure it's not called
    called = {"v": False}
    original = gmail_service.mark_as_spam
    
    def wrapper(msg_id):
        called["v"] = True
        return original(msg_id)
    
    # Patch the agent module entry point for mark_as_spam to our wrapper
    monkeypatch.setattr(mod, "mark_as_spam", wrapper)

    cmd = mod.interrupt_handler(state, store)
    # Should not call mark_as_spam
    assert called["v"] is False
    # Should continue the workflow (back to llm_call)
    assert cmd.goto == "llm_call"

