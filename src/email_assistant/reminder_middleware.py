"""Sender reputation and reminder risk helpers."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Tuple


PROFILE_NAMESPACE = ("email_assistant", "sender_reputation")
PROFILE_KEY = "profile"

_MONEY_KEYWORDS = [
    "invoice",
    "payment",
    "wire",
    "transfer",
    "bank",
    "crypto",
    "bitcoin",
    "paypal",
    "urgent",
    "overdue",
    "bill",
    "due",
]


@dataclass
class SenderAssessment:
    email: str
    status: str
    risk_level: str
    reason: str


def _load_profile(store) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Load the sender profile from the provided store, falling back to a default empty profile if unavailable.
    
    Attempts to read PROFILE_NAMESPACE/PROFILE_KEY from the store and parse its JSON value. If the entry is missing or cannot be parsed, returns a default profile with empty "known" and "flagged" mappings.
    
    Returns:
        profile (Dict[str, Dict[str, Dict[str, str]]]): A profile object containing "known" and "flagged" dictionaries that map email addresses to metadata dictionaries (for example "status", "reason", "updated_at").
    """
    try:
        entry = store.get(PROFILE_NAMESPACE, PROFILE_KEY)
        if entry and getattr(entry, "value", None):
            return json.loads(entry.value)
    except Exception:
        pass
    return {"known": {}, "flagged": {}}


def _save_profile(store, profile: Dict[str, Dict[str, Dict[str, str]]]) -> None:
    """
    Persist the sender profile to the given store under the module's PROFILE_NAMESPACE/PROFILE_KEY.
    
    Parameters:
        store: A storage-like object exposing a `put(namespace, key, value)` method.
        profile (Dict[str, Dict[str, Dict[str, str]]]): Profile data (e.g., keys "known" and "flagged") that will be serialized as JSON and saved.
    
    Notes:
        Any exception raised by the store is swallowed; failures to persist are silent.
    """
    try:
        store.put(PROFILE_NAMESPACE, PROFILE_KEY, json.dumps(profile))
    except Exception:
        pass


def _extract_email(address: str | None) -> str:
    """
    Extracts and normalizes an email address from a header-like string.
    
    Parameters:
        address (str | None): A raw address string (e.g., "Name <user@example.com>" or "user@example.com"); may be None.
    
    Returns:
        str: The email address in lowercase with surrounding whitespace removed, or an empty string if `address` is falsy.
    """
    if not address:
        return ""
    match = re.search(r"<([^>]+)>", address)
    if match:
        return match.group(1).strip().lower()
    return address.strip().lower()


def assess_sender(store, author: str | None, subject: str, body: str) -> SenderAssessment:
    """
    Evaluate sender reputation and assign a reminder risk level based on stored sender profiles and message content.
    
    If the author address is missing or cannot be extracted, the function returns a high-risk assessment. The function also updates the profile's last-seen timestamp for the assessed email and persists the profile.
    
    Parameters:
        author (str | None): Raw author/address string (may include angle-bracketed address).
        subject (str): Email subject text.
        body (str): Email body text.
    
    Returns:
        SenderAssessment: Assessment containing `email`, `status`, `risk_level`, and `reason`. `email` will be an empty string and `risk_level` will be `high` when the sender address is missing.
    """
    profile = _load_profile(store)
    email = _extract_email(author)
    if not email:
        return SenderAssessment(email="", status="unknown", risk_level="high", reason="Missing sender address")

    now_iso = datetime.now(timezone.utc).isoformat()
    record = profile["known"].get(email) or profile["flagged"].get(email)
    status = record.get("status") if record else "new"

    lower_subject = (subject or "").lower()
    lower_body = (body or "").lower()
    text = f"{lower_subject}\n{lower_body}"

    risk_level = "low"
    reason = "Known sender"

    money_hit = any(keyword in text for keyword in _MONEY_KEYWORDS)
    if status == "new":
        if money_hit:
            risk_level = "high"
            reason = "New sender requesting financial action"
        else:
            risk_level = "medium"
            reason = "New sender"
    elif status == "flagged":
        risk_level = "high"
        reason = record.get("reason", "Previously flagged sender")
    else:
        reason = record.get("reason", "Known sender")

    profile.setdefault("last_seen", {})[email] = now_iso
    _save_profile(store, profile)

    return SenderAssessment(email=email, status=status, risk_level=risk_level, reason=reason)


def note_sender(store, email: str, status: str, reason: str | None = None) -> None:
    """
    Record or update a sender's status in the persistent profile store.
    
    Updates the profile stored in `store` to mark `email` with the provided `status` and optional `reason`, sets an `updated_at` timestamp, and ensures the email appears only in the appropriate category (`known` for "trusted"/"known", `flagged` for "flagged"). If `email` is falsy the function does nothing.
    
    Parameters:
        store: Storage backend used by _load_profile/_save_profile.
        email (str): The sender email address to record.
        status (str): New status for the sender (e.g., "trusted", "known", "flagged").
        reason (str | None): Optional human-readable reason for the status change.
    """
    if not email:
        return
    profile = _load_profile(store)
    entry = {"status": status, "updated_at": datetime.now(timezone.utc).isoformat()}
    if reason:
        entry["reason"] = reason
    if status in {"trusted", "known"}:
        profile.setdefault("known", {})[email] = entry
        profile.get("flagged", {}).pop(email, None)
    elif status == "flagged":
        profile.setdefault("flagged", {})[email] = entry
        profile.get("known", {}).pop(email, None)
    _save_profile(store, profile)


def sender_exists(store, email: str) -> bool:
    """
    Check whether an email address is present in the stored sender profile.
    
    Returns:
        `True` if the email exists in either the profile's "known" or "flagged" entries, `False` otherwise.
    """
    profile = _load_profile(store)
    return email in profile.get("known", {}) or email in profile.get("flagged", {})


def sender_profile_snapshot(store) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Return a snapshot of the sender profile.
    
    Returns:
        dict: The full profile with keys "known" and "flagged", each mapping sender email strings to metadata dictionaries (e.g., "status", "updated_at", "reason").
    """
    return _load_profile(store)


def judge_disabled() -> bool:
    """
    Determine whether the reminder judge should be disabled based on environment variables.
    
    Checks REMINDER_JUDGE_FORCE_DECISION and EMAIL_ASSISTANT_LLM_JUDGE to decide behavior:
    - If REMINDER_JUDGE_FORCE_DECISION is set, the judge is not disabled.
    - Otherwise the judge is disabled when EMAIL_ASSISTANT_LLM_JUDGE is not one of "1", "true", or "yes" (case-insensitive).
    
    Returns:
        bool: `True` if the reminder judge should be disabled (no LLM judge in use and no forced decision override), `False` otherwise.
    """
    forced = os.getenv("REMINDER_JUDGE_FORCE_DECISION")
    if forced:
        return False
    use_llm = os.getenv("EMAIL_ASSISTANT_LLM_JUDGE", "").lower() in ("1", "true", "yes")
    return not use_llm
