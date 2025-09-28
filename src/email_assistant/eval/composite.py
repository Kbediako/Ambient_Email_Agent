"""Composite evaluator that combines judge scores."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from email_assistant.eval.judges import JudgeResult, resolve_feedback_targets
from email_assistant.eval.reminder_run_judge import (
    ReminderRunJudgeVerdict,
    _primary_thread_id,
    _reminder_input_payload,
)
from email_assistant.tracing import (
    AGENT_PROJECT,
    JUDGE_PROJECT,
    invoke_with_root_run,
    prime_parent_run,
)
class CompositeJudgeResult(BaseModel):
    overall_score: float = Field(..., ge=0.0, le=1.0)
    verdict: str = Field(..., description="pass/fail based on weighted score")
    component_scores: Dict[str, float] = Field(default_factory=dict)
    notes: str = Field(default="", description="Summary of component contributions")


def combine_judge_scores(
    correctness: JudgeResult,
    reminder: ReminderRunJudgeVerdict,
    *,
    weights: Optional[Dict[str, float]] = None,
    threshold: float = 0.70,
) -> CompositeJudgeResult:
    """
    Produce a weighted composite judge result from correctness and reminder scores.
    
    Parameters:
        correctness (JudgeResult): JudgeResult containing `overall_correctness` in [0.0, 1.0].
        reminder (ReminderRunJudgeVerdict): Verdict containing `reminder_score` in [0.0, 1.0].
        weights (Optional[Dict[str, float]]): Optional mapping with keys "correctness" and "reminder"
            that specify relative weights for each component. Defaults to {"correctness": 0.6, "reminder": 0.4}.
        threshold (float): Score threshold used to assign the verdict "pass" when the composite score
            is greater than or equal to this value. Default is 0.70.
    
    Returns:
        CompositeJudgeResult: Contains `overall_score` (the weighted composite score), `verdict` ("pass" or "fail"),
        `component_scores` mapping the individual component scores, and a `notes` string summarizing values and weights.
    
    Raises:
        ValueError: If the sum of provided weights is less than or equal to zero.
    """

    weights = weights or {"correctness": 0.6, "reminder": 0.4}
    correct_w = weights.get("correctness", 0.6)
    reminder_w = weights.get("reminder", 0.4)
    total = correct_w + reminder_w
    if total <= 0:
        raise ValueError("Composite weights must be positive")

    composite = (
        correctness.overall_correctness * correct_w + reminder.reminder_score * reminder_w
    ) / total

    verdict = "pass" if composite >= threshold else "fail"
    notes = (
        f"Correctness={correctness.overall_correctness:.2f} (w={correct_w:.2f}); "
        f"Reminder={reminder.reminder_score:.2f} (w={reminder_w:.2f})"
    )

    return CompositeJudgeResult(
        overall_score=composite,
        verdict=verdict,
        component_scores={
            "correctness": correctness.overall_correctness,
            "reminder": reminder.reminder_score,
        },
        notes=notes,
    )


def _resolve_judge_project() -> str:
    """
    Resolve the judge project identifier using environment-variable overrides with fallbacks.
    
    Checks the following environment variables in order and returns the first non-empty value found:
    EMAIL_ASSISTANT_REMINDER_JUDGE_PROJECT_OVERRIDE, EMAIL_ASSISTANT_REMINDER_JUDGE_PROJECT,
    EMAIL_ASSISTANT_JUDGE_PROJECT_OVERRIDE, EMAIL_ASSISTANT_JUDGE_PROJECT. If none are set,
    returns the module-level default JUDGE_PROJECT.
    
    Returns:
        str: The resolved judge project name.
    """
    for key in (
        "EMAIL_ASSISTANT_REMINDER_JUDGE_PROJECT_OVERRIDE",
        "EMAIL_ASSISTANT_REMINDER_JUDGE_PROJECT",
        "EMAIL_ASSISTANT_JUDGE_PROJECT_OVERRIDE",
        "EMAIL_ASSISTANT_JUDGE_PROJECT",
    ):
        candidate = os.getenv(key)
        if candidate:
            return candidate
    return JUDGE_PROJECT


def _resolve_agent_project() -> str:
    """
    Determine the agent project name, preferring an environment-variable override.
    
    Returns:
        str: The agent project name — the value of `EMAIL_ASSISTANT_REMINDER_AGENT_PROJECT` if set, otherwise the default `AGENT_PROJECT`.
    """
    override = os.getenv("EMAIL_ASSISTANT_REMINDER_AGENT_PROJECT")
    if override:
        return override
    return AGENT_PROJECT


def _attach_composite_feedback(
    run_id: Optional[str],
    result: CompositeJudgeResult,
    *,
    email_markdown: Optional[str],
) -> None:
    """
    Attach the composite judge result as feedback to LangSmith runs when an API key and target runs are available.
    
    This attempts to resolve feedback targets (optionally using the provided run_id and email_markdown) and create a feedback entry named "reminder_composite" for each target run. If the LANGSMITH_API_KEY is not set, or no client/targets are resolved, the function returns without action. Individual feedback creation errors are ignored so a failure for one target does not prevent attempts for others.
    
    Parameters:
        run_id (Optional[str]): Optional run identifier to scope feedback target resolution.
        result (CompositeJudgeResult): The composite evaluation to attach as feedback; its serialized payload, overall score, verdict, and notes are included in the feedback.
        email_markdown (Optional[str]): Optional markdown string used when resolving feedback targets or providing context.
    """
    if not os.getenv("LANGSMITH_API_KEY"):
        return

    project = _resolve_agent_project()
    if project:
        os.environ.setdefault("LANGSMITH_PROJECT", project)
        os.environ.setdefault("LANGCHAIN_PROJECT", project)

    try:
        client, run_ids = resolve_feedback_targets(
            run_id, email_markdown=email_markdown
        )
    except Exception:
        return
    if not client or not run_ids:
        return

    payload = result.model_dump()
    comp = result.component_scores or {}
    correctness = comp.get("correctness")
    reminder_component = comp.get("reminder")
    parts = [
        f"score={result.overall_score:.2f}",
        f"verdict={result.verdict}",
    ]
    if correctness is not None:
        parts.append(f"correctness={correctness:.2f}")
    if reminder_component is not None:
        parts.append(f"reminder={reminder_component:.2f}")
    summary = "; ".join(parts)

    for target in run_ids:
        try:
            client.create_feedback(
                run_id=target,
                key="reminder_composite",
                score=result.overall_score,
                value=summary,
                comment=result.notes,
                extra=payload,
            )
        except Exception:
            continue


def run_composite_judge(
    *,
    correctness: JudgeResult,
    reminder: ReminderRunJudgeVerdict,
    parent_run_id: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
    threshold: float = 0.70,
    sender_email: Optional[str] = None,
    email_markdown: Optional[str] = None,
    email_input: Optional[Dict[str, Any]] = None,
    reminder_created: Optional[List[dict]] = None,
    reminder_cleared: Optional[List[dict]] = None,
) -> CompositeJudgeResult:
    """
    Compute a weighted composite reminder judgment from correctness and reminder judgments, record the result as a traced run with metadata, and attach LangSmith feedback when configured.
    
    Parameters:
        correctness: JudgeResult containing the correctness evaluation for the run.
        reminder: ReminderRunJudgeVerdict containing the reminder-specific evaluation for the run.
        parent_run_id: Optional run identifier to attach LangSmith feedback to.
        weights: Optional mapping of component weights (keys "correctness" and "reminder"); defaults to {"correctness": 0.6, "reminder": 0.4}.
        threshold: Score threshold used to determine the composite verdict; defaults to 0.70.
        sender_email: Optional sender email used to construct or enrich the recorded email payload.
        email_markdown: Optional markdown content of the email used as feedback context.
        email_input: Optional prebuilt email payload to record instead of deriving one from other inputs.
        reminder_created: Optional list of dicts describing reminders created during processing; included in run metadata.
        reminder_cleared: Optional list of dicts describing reminders cleared during processing; included in run metadata.
    
    Returns:
        CompositeJudgeResult: The finalized composite evaluation containing overall_score, verdict, component_scores, and notes.
    """

    result = combine_judge_scores(
        correctness,
        reminder,
        weights=weights,
        threshold=threshold,
    )

    project = _resolve_judge_project()
    weights_payload = dict(weights or {"correctness": 0.6, "reminder": 0.4})

    created_records = reminder_created or []
    cleared_records = reminder_cleared or []

    if email_input is not None:
        email_payload = dict(email_input)
    else:
        email_payload = _reminder_input_payload(
            sender_email or "",
            created_records,
            cleared_records,
            email_markdown or "",
            "",
        )

    def _log_result() -> CompositeJudgeResult:
        """
        Log the composite judge result as a parent run artifact and return it.
        
        Returns:
            result (CompositeJudgeResult): The composite judge result that was recorded and is returned unchanged.
        """
        prime_parent_run(
            email_input=email_payload,
            email_markdown=email_markdown or "",
            outputs=result.model_dump(),
            agent_label="judge:reminder:composite",
            tags=["reminder_judge"],
            metadata_update={
                "weights": weights_payload,
                "threshold": threshold,
                "component_scores": result.component_scores,
                "sender_email": sender_email,
                "reminder_created_count": len(created_records),
                "reminder_cleared_count": len(cleared_records),
            },
            thread_id=_primary_thread_id(created_records, cleared_records),
        )
        return result

    def _summary(value: CompositeJudgeResult) -> str:
        """
        Create a compact one-line summary of a CompositeJudgeResult for logging or display.
        
        Parameters:
            value (CompositeJudgeResult): The composite judge result to summarize.
        
        Returns:
            summary (str): Single-line string containing the verdict and overall score formatted to two decimal places (e.g., "[reminder_composite] verdict=pass score=0.85").
        """
        return (
            f"[reminder_composite] verdict={value.verdict} "
            f"score={value.overall_score:.2f}"
        )

    invoke_with_root_run(
        _log_result,
        root_name="judge:reminder:composite",
        input_summary="composite reminder score",
        metadata={
            "weights": weights_payload,
            "threshold": threshold,
            "component_scores": result.component_scores,
        },
        output_transform=_summary,
        project_name=project,
    )

    _attach_composite_feedback(
        parent_run_id,
        result,
        email_markdown=email_markdown,
    )

    return result
