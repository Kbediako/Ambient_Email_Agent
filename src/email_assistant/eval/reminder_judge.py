"""LLM-backed reminder risk evaluation."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field

from email_assistant.configuration import get_llm


class ReminderJudgeResult(BaseModel):
    decision: Literal["approve", "hitl", "reject"] = Field(..., description="Decision outcome")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk score between 0-1")
    rationale: str = Field(..., description="Short explanation of decision")


DEFAULT_APPROVAL = ReminderJudgeResult(decision="approve", risk_score=0.4, rationale="Default approval")
DEFAULT_HITL = ReminderJudgeResult(decision="hitl", risk_score=0.8, rationale="Escalate to human review")


PROMPT_TEMPLATE = """
You are a security reviewer protecting the user's interests. A Gmail assistant wants to set a follow-up reminder.

Decide whether to:
- "approve" the reminder automatically (safe, expected, trusted sender),
- "hitl" to escalate to human review (suspicious or unverified but not clearly malicious), or
- "reject" when the reminder should not be created (phishing, malicious, or violates policy).

Consider sender reputation, financial requests, urgency, links, and whether the assistant already responded.

Return EXACTLY one JSON object with keys decision (approve/hitl/reject), risk_score (0-1 float), rationale (<=200 chars).

Context:
- Sender email: {sender_email}
- Sender status: {sender_status}
- Current risk level: {risk_level}
- Risk reason: {risk_reason}
- Email summary: {email_summary}
- Proposed reminder reason: {reminder_reason}
"""


def _forced_decision(default_result: ReminderJudgeResult) -> ReminderJudgeResult:
    forced = os.getenv("REMINDER_JUDGE_FORCE_DECISION", "").lower()
    if forced == "approve":
        return ReminderJudgeResult(decision="approve", risk_score=0.2, rationale="Forced approval via env")
    if forced == "hitl":
        return ReminderJudgeResult(decision="hitl", risk_score=0.8, rationale="Forced HITL via env")
    if forced == "reject":
        return ReminderJudgeResult(decision="reject", risk_score=0.95, rationale="Forced rejection via env")
    return default_result


def evaluate_reminder_risk(
    *,
    sender_email: str,
    sender_status: str,
    risk_level: str,
    risk_reason: str,
    email_summary: str,
    reminder_reason: str,
) -> ReminderJudgeResult:
    """Evaluate high-risk reminders and determine approval or escalation."""

    default = DEFAULT_HITL if risk_level.lower() == "high" else DEFAULT_APPROVAL

    if os.getenv("EMAIL_ASSISTANT_EVAL_MODE", "").lower() in ("1", "true", "yes"):
        return _forced_decision(default)

    result = _forced_decision(default)
    if result is not default:
        return result

    if os.getenv("EMAIL_ASSISTANT_LLM_JUDGE", "").lower() not in ("1", "true", "yes"):
        return default

    prompt = PROMPT_TEMPLATE.format(
        sender_email=sender_email or "(unknown)",
        sender_status=sender_status,
        risk_level=risk_level,
        risk_reason=risk_reason,
        email_summary=email_summary,
        reminder_reason=reminder_reason,
    )

    try:
        llm = get_llm(model=os.getenv("EMAIL_ASSISTANT_JUDGE_MODEL") or None)
        structured_llm = llm.with_structured_output(ReminderJudgeResult)
        decision = structured_llm.invoke([
            {"role": "system", "content": "Return only the JSON object."},
            {"role": "user", "content": prompt},
        ])
        return decision
    except Exception:
        return default
