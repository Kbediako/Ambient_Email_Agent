import types

from email_assistant.tools.gmail.gmail_tools import fetch_group_emails


def _make_service(messages_pages):
    # Minimal fake for gmail service: users().messages().list().execute()
    svc = types.SimpleNamespace()
    def list_fn(userId, q, pageToken=None):
        idx = int(pageToken) if pageToken else 0
        page = messages_pages[idx]
        next_token = str(idx + 1) if idx + 1 < len(messages_pages) else None
        return types.SimpleNamespace(execute=lambda: {"messages": page, "nextPageToken": next_token})
    def get_fn(userId, id):
        msg = {
            "id": id,
            "threadId": f"t-{id}",
            "payload": {"headers": [{"name": "From", "value": "other@example.com"}, {"name": "Subject", "value": "Hello"}]},
        }
        return types.SimpleNamespace(execute=lambda: msg)
    def thread_get_fn(userId, id):
        # two messages in thread, with second the latest
        m1 = {"id": id + "-1", "payload": {"headers": [{"name": "From", "value": "me@example.com"}]} }
        m2 = {"id": id + "-2", "payload": {"headers": [{"name": "From", "value": "other@example.com"}], "headers": [{"name": "From", "value": "other@example.com"}]}}
        return types.SimpleNamespace(execute=lambda: {"messages": [m1, m2]})
    svc.users = lambda: types.SimpleNamespace(
        messages=lambda: types.SimpleNamespace(list=list_fn, get=get_fn),
        threads=lambda: types.SimpleNamespace(get=thread_get_fn),
    )
    return svc


def test_skip_filters_true_processes_latest(monkeypatch):
    monkeypatch.setenv("GMAIL_TOKEN", "{}")
    # Patch credentials to bypass real API
    monkeypatch.setattr("email_assistant.tools.gmail.gmail_tools.get_credentials", lambda *a, **k: types.SimpleNamespace())
    # Patch build to our fake service
    monkeypatch.setattr("email_assistant.tools.gmail.gmail_tools.build", lambda *a, **k: _make_service([[{"id": "m1"}]]) )

    emails = list(fetch_group_emails("me@example.com", minutes_since=1, include_read=True, skip_filters=True))
    assert emails, "Expected at least one processed email"

