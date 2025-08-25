#!/usr/bin/env python

import os
import sys
from pathlib import Path
from typing import List, Optional

import pytest

project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))


def pytest_addoption(parser):
    """Add command-line options to pytest."""
    parser.addoption(
        "--agent-module",
        action="store",
        default="email_assistant",
        help="Specify which email assistant module to test",
    )


@pytest.fixture(scope="session")
def agent_module_name(request):
    """Return the agent module name from command line."""
    return request.config.getoption("--agent-module")


# ---------------- Gmail service fixture ---------------- #
class FakeGmailClient:
    """In-memory fake Gmail client for tests without real API access."""

    def __init__(self):
        self.created_message_ids: List[str] = []
        self.created_thread_ids: List[str] = []
        self.actions: List[object] = []

    def fetch_inbox(self, query: Optional[str] = None):
        return []

    def mark_as_read(self, message_id: str, gmail_token: Optional[str] = None, gmail_secret: Optional[str] = None):
        self.actions.append(("mark_as_read", message_id))
        return None

    def mark_as_spam(self, message_id: str, gmail_token: Optional[str] = None, gmail_secret: Optional[str] = None) -> str:
        self.actions.append(("mark_as_spam", message_id))
        return f"Moved message {message_id} to Spam."

    def send_email(
        self,
        email_id: str,
        response_text: str,
        email_address: str,
        addn_receipients: Optional[List[str]] = None,
    ) -> bool:
        new_id = f"fake-msg-{len(self.created_message_ids) + 1}"
        thread_id = (
            f"fake-thread-{len(self.created_thread_ids) + 1}"
            if not email_id or email_id == "NEW_EMAIL"
            else f"fake-thread-{email_id}"
        )
        self.created_message_ids.append(new_id)
        if thread_id not in self.created_thread_ids:
            self.created_thread_ids.append(thread_id)
        self.actions.append(("send_email", email_id, new_id))
        return True

    def teardown(self):
        # Nothing to clean up for fake client
        return None


class RealGmailClient:
    """Thin wrapper around the Gmail API with bookkeeping and teardown."""

    def __init__(self):
        from googleapiclient.discovery import build
        from email_assistant.tools.gmail.gmail_tools import get_credentials

        creds = get_credentials(os.getenv("GMAIL_TOKEN"), os.getenv("GMAIL_SECRET"))
        if creds is None:
            raise RuntimeError("Gmail credentials not available. Set GMAIL_TOKEN or create .secrets/token.json")
        self._service = build("gmail", "v1", credentials=creds)
        self.created_message_ids: List[str] = []
        self.created_thread_ids: List[str] = []

    # Minimal inbox fetch for completeness
    def fetch_inbox(self, query: Optional[str] = None):
        result = self._service.users().messages().list(userId="me", q=query or "").execute()
        return result.get("messages", [])

    def mark_as_read(self, message_id: str, gmail_token: Optional[str] = None, gmail_secret: Optional[str] = None):
        self._service.users().messages().modify(
            userId="me", id=message_id, body={"removeLabelIds": ["UNREAD"]}
        ).execute()

    def mark_as_spam(self, message_id: str, gmail_token: Optional[str] = None, gmail_secret: Optional[str] = None) -> str:
        self._service.users().messages().modify(
            userId="me",
            id=message_id,
            body={"addLabelIds": ["SPAM"], "removeLabelIds": ["INBOX"]},
        ).execute()
        return f"Moved message {message_id} to Spam."

    def send_email(
        self,
        email_id: str,
        response_text: str,
        email_address: str,
        addn_receipients: Optional[List[str]] = None,
    ) -> bool:
        import base64
        from email.mime.text import MIMEText

        # Try to reply to an existing message; otherwise create a new one
        thread_id = None
        to_addr = "recipient@example.com"
        subject = "Response"
        try:
            if email_id and email_id != "NEW_EMAIL":
                msg = self._service.users().messages().get(userId="me", id=email_id).execute()
                headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
                subject = headers.get("Subject", subject)
                if not subject.startswith("Re:"):
                    subject = f"Re: {subject}"
                to_addr = headers.get("From", to_addr)
                thread_id = msg.get("threadId")
        except Exception:
            # Fallback to a new message
            thread_id = None

        mime = MIMEText(response_text)
        mime["to"] = to_addr
        mime["from"] = email_address
        mime["subject"] = subject
        if addn_receipients:
            mime["cc"] = ", ".join(addn_receipients)
        raw = base64.urlsafe_b64encode(mime.as_bytes()).decode("utf-8")
        body = {"raw": raw}
        if thread_id:
            body["threadId"] = thread_id
        sent = (
            self._service.users().messages().send(userId="me", body=body).execute()
        )
        msg_id = sent.get("id")
        thr_id = sent.get("threadId")
        if msg_id:
            self.created_message_ids.append(msg_id)
        if thr_id:
            self.created_thread_ids.append(thr_id)
        return True

    def teardown(self):
        # Best-effort cleanup of created resources (permanently delete)
        try:
            for mid in list(set(self.created_message_ids)):
                try:
                    self._service.users().messages().delete(userId="me", id=mid).execute()
                except Exception:
                    pass
            for tid in list(set(self.created_thread_ids)):
                try:
                    self._service.users().threads().delete(userId="me", id=tid).execute()
                except Exception:
                    pass
        except Exception:
            pass


@pytest.fixture(scope="function")
def gmail_service(monkeypatch):
    """Provide a Gmail service (fake unless GMAIL_API_KEY is set).

    - When GMAIL_API_KEY is absent: returns FakeGmailClient and patches Gmail ops.
    - When present: builds a real Gmail API client and patches Gmail ops; cleans up created messages/threads on teardown.
    """
    from email_assistant.tools.gmail import gmail_tools

    use_real = bool(os.environ.get("GMAIL_API_KEY"))
    try:
        service = RealGmailClient() if use_real else FakeGmailClient()
    except Exception:
        # Fall back to fake if real client fails to initialize
        service = FakeGmailClient()

    # Route gmail_tools functions through the service
    monkeypatch.setattr(gmail_tools, "mark_as_read", service.mark_as_read, raising=True)
    monkeypatch.setattr(gmail_tools, "mark_as_spam", service.mark_as_spam, raising=True)
    monkeypatch.setattr(gmail_tools, "send_email", service.send_email, raising=True)

    # Also patch imported symbols in the Gmail agent module if available
    try:
        import email_assistant.email_assistant_hitl_memory_gmail as gmail_agent_mod
        monkeypatch.setattr(gmail_agent_mod, "mark_as_read", service.mark_as_read, raising=False)
        monkeypatch.setattr(gmail_agent_mod, "mark_as_spam", service.mark_as_spam, raising=False)
    except Exception:
        pass

    try:
        yield service
    finally:
        # Teardown for real client to clean up created resources
        try:
            service.teardown()
        except Exception:
            pass