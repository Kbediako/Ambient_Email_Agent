import pytest
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

# Ensure imports from src work by adjusting the path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from email_assistant.tools.reminders import SqliteReminderStore
from scripts.reminder_worker import check_reminders


@pytest.fixture
def memory_store() -> SqliteReminderStore:
    """Provides a clean, in-memory SQLite database for each test."""
    # Use the special :memory: path for an in-memory DB
    store = SqliteReminderStore(db_path=":memory:")
    store.setup()
    return store


def test_add_and_get_reminder(memory_store: SqliteReminderStore):
    """Tests that a reminder can be added and retrieved."""
    thread_id = "thread_123"
    due_at = datetime.now(timezone.utc) + timedelta(days=1)
    
    # Add a reminder
    reminder_id = memory_store.add_reminder(thread_id, "Test Subject", due_at, "test reason")
    assert reminder_id is not None

    # To test retrieval, we'll check the due reminders logic.
    # Since it's not due yet, it shouldn't be returned.
    assert not memory_store.get_due_reminders()


def test_add_reminder_idempotent(memory_store: SqliteReminderStore):
    """Tests that adding the same reminder twice doesn't create a duplicate."""
    thread_id = "thread_idempotent"
    due_at = datetime.now(timezone.utc) + timedelta(days=1)

    # Add the reminder twice
    id1 = memory_store.add_reminder(thread_id, "Sub", due_at, "reason")
    id2 = memory_store.add_reminder(thread_id, "Sub", due_at, "reason")

    # Assert that the returned ID is the same and no new reminder was created
    assert id1 == id2


def test_cancel_reminder(memory_store: SqliteReminderStore):
    """Tests that a reminder can be successfully canceled."""
    thread_id = "thread_to_cancel"
    due_at = datetime.now(timezone.utc) + timedelta(days=1)
    memory_store.add_reminder(thread_id, "Sub", due_at, "reason")

    # Cancel the reminder
    cancelled_count = memory_store.cancel_reminder(thread_id)
    assert cancelled_count == 1

    # Verify it is no longer considered active/due
    assert not memory_store.get_due_reminders()

    # Verify that trying to cancel again does nothing
    cancelled_again_count = memory_store.cancel_reminder(thread_id)
    assert cancelled_again_count == 0


def test_worker_finds_due_reminders(memory_store: SqliteReminderStore):
    """Tests that the worker's check_reminders function finds and processes due reminders."""
    thread_id_due = "thread_due"
    thread_id_not_due = "thread_not_due"

    # Create one reminder that is due and one that is not
    due_time = datetime.now(timezone.utc) - timedelta(minutes=1)
    not_due_time = datetime.now(timezone.utc) + timedelta(days=1)
    memory_store.add_reminder(thread_id_due, "Due Subject", due_time, "is due")
    memory_store.add_reminder(thread_id_not_due, "Not Due Subject", not_due_time, "is not due")

    # Mock the delivery service to see if it gets called
    mock_delivery = MagicMock()

    # Use patch to replace the factory functions within the scope of this test
    with patch('scripts.reminder_worker.get_default_store', return_value=memory_store), \
         patch('scripts.reminder_worker.get_default_delivery', return_value=mock_delivery):
        
        # Run the worker logic
        check_reminders(memory_store)

    # Assert that send_notification was called exactly once
    mock_delivery.send_notification.assert_called_once()
    
    # Assert that it was called with the correct reminder
    call_args, _ = mock_delivery.send_notification.call_args
    called_with_reminder = call_args[0]
    assert called_with_reminder.thread_id == thread_id_due

    # Assert that the due reminder is no longer due after being processed
    assert not memory_store.get_due_reminders()
