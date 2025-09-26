"""LLM-based evaluators for the email assistant.

This module exposes a Gemini 2.5 Flash powered "LLM-as-judge" utility that
scores a run's final reply and tool usage. The implementation mirrors the
LangSmith prompt shared in the project documentation so it can be reused both
locally (during pytest runs) and in LangSmith datasets/experiments.
"""

from __future__ import annotations

import json
import contextvars
import os
import time
from functools import lru_cache
from typing import Any, Dict, Iterable, Iterator, List, Literal, Optional, Tuple
from uuid import UUID

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
from langsmith import Client
from langsmith.evaluation import EvaluationResult, LangChainStringEvaluator, StringEvaluator
from langsmith.run_helpers import get_current_run_tree, traceable
from langsmith.schemas import Example, Run

from email_assistant.configuration import get_llm
from email_assistant.utils import extract_message_content, format_messages_string
from email_assistant.tracing import (
    email_fingerprint,
    invoke_with_root_run,
    strip_markdown_to_text,
    truncate_markdown,
)

_FEEDBACK_PRIORITY_DELAY = float(os.getenv("EMAIL_ASSISTANT_FEEDBACK_DELAY", "0.05"))

__all__ = [
    "JudgeResult",
    "JudgeUnavailableError",
    "build_correctness_judge_chain",
    "build_tool_call_context",
    "run_correctness_judge",
    "serialise_messages",
    "create_langsmith_correctness_evaluator",
    "iter_experiment_runs",
]


class JudgeUnavailableError(RuntimeError):
    """Raised when the judge cannot be executed (e.g., config missing)."""


class ToolIssue(BaseModel):
    tool: str = Field(default="", description="Name of the tool that was misused")
    why: str = Field(default="", description="Explanation of what went wrong")


class JudgeResult(BaseModel):
    overall_correctness: float
    verdict: Literal["pass", "fail"]
    content_alignment: int
    tool_usage: int
    missing_tools: List[str]
    incorrect_tool_uses: List[ToolIssue]
    evidence: List[str]
    notes: str

    def short_summary(self) -> str:
        return (
            f"verdict={self.verdict} overall={self.overall_correctness:.2f} "
            f"content={self.content_alignment}/5 tools={self.tool_usage}/5"
        )


SYSTEM_PROMPT = """You are “LLM Reviewer.” Return ONE valid JSON object only.\n\nJudge CORRECTNESS of an email agent’s final reply and its tool usage.\n\nAxes (0–5 each):\n• Content Alignment — Does the drafted reply resolve the sender’s request with correct commitments, deadlines, tone, and follow-through?\n• Tool Usage — Were the right tools used with correct arguments, in a valid order?\n\nPolicies:\n\n- Scheduling request → check_calendar_tool → schedule_meeting_tool → send_email_tool → Done. Confirm meeting duration matches the ask (allow ±10 minutes for “about” phrasing).\n- 90-minute planning (availability-only) → check_calendar_tool → send_email_tool → Done (NO scheduling). Ensure reply offers availability covering the requested window.\n- Conference invitations that only require interest/questions (e.g., TechConf) → send_email_tool → Done. Do NOT expect calendar tools unless the sender explicitly asks to schedule.\n- send_email_tool must include email_id and email_address, then Done.\n- Never call Done before drafting the reply.\n- Spam requires explicit confirmation before mark_as_spam_tool.\n- Tool-name normalization: treat as equivalent\n    - send_email_tool == write_email\n    - check_calendar_tool == check_calendar_availability\n    - schedule_meeting_tool == schedule_meeting\n- Exceptions:\n    - No‑reply/system “do not reply”: may end without send_email_tool (Done allowed).\n    - Spam (after explicit HITL confirmation): may end via mark_as_spam_tool without Done and without drafting a reply.\n- Question is correct when information is missing or consent is ambiguous.\n- For schedule_meeting_tool: start_time/end_time must be ISO; organizer_email required; timezone present or defaulted; duration should respect the request.\n- Tool-name/arg normalization: send_email_tool == write_email. When the name is write_email (normalized), accept the fields “to”, “subject”, and “content” and do not require Gmail-only args. Require email_id and email_address only when the raw name is send_email_tool.\n- When evaluating send_email_tool/write_email content, ensure the reply acknowledges key user constraints (deadlines, commitments, next steps) reflected in the email context and any executed tools.\n\nFinding & scoring rules:\n- Always explain tool issues via missing_tools / incorrect_tool_uses instead of only lowering scores.\n- missing_tools must list every expected tool from Policies that never ran (use normalized names).\n- incorrect_tool_uses must contain {{"tool": name, "why": explanation}} for wrong args, wrong order, premature Done, duration mismatches, wrong recipients, or other policy violations.\n- - If either list is non-empty, tool_usage must be ≤2 (use 1 for severely broken flows). Scores 3–5 are only allowed when both lists are empty.\n- Reduce content_alignment when the reply text fails to acknowledge commitments/tools outcomes the user needs.\n\nRubric:\n• Content Alignment: 5 exact/complete/tone-appropriate; 3 minor gaps; 1 misses/incorrect.\n• Tool Usage: 5 correct tools/args/order (+ terminal Done when applicable); 3 minor issues; 1 wrong/missing tools or invalid sequence.\n\nExamples:\nPass — all tools correct:\n{{"overall_correctness": 0.92, "verdict": "pass", "content_alignment": 5, "tool_usage": 5, "missing_tools": [], "incorrect_tool_uses": [], "evidence": ["Schedule shows 60 min meeting", "Reply confirms time and attendees"], "notes": "Handled scheduling request end-to-end."}}\nFail — missing schedule_meeting_tool:\n{{"overall_correctness": 0.40, "verdict": "fail", "content_alignment": 2, "tool_usage": 2, "missing_tools": ["schedule_meeting_tool"], "incorrect_tool_uses": [], "evidence": ["Email asked to schedule 60 min", "Only check_calendar_tool was called"], "notes": "Agent never scheduled the requested meeting."}}\n\nEvaluation source of truth:\n\n- Prefer final-state fields if present: output.assistant_reply, output.tool_trace, output.email_markdown.\n- Otherwise, derive from output.messages (e.g., last assistant content; AI tool_calls).\n- Validate arguments using raw tool calls from output.messages (AI messages with tool_calls). Use output.tool_trace for sequence evidence only; it is lossy/normalized.\n- Do not penalize the literal tool name when an equivalent normalized name is used.\n- Treat each field as populated unless it literally reads "(no … provided)" or "(empty)". Truncated strings ending with an ellipsis (… ) still contain evidence—use them. NEVER claim "information insufficient" when substantive content is present.\n\nCompute:\noverall_correctness = 0.6*(content_alignment/5) + 0.4*(tool_usage/5)\nverdict = "pass" if overall_correctness ≥ 0.70 else "fail".\n\nSTRICT OUTPUT CONTRACT: return valid JSON with these exact keys (empty arrays allowed): overall_correctness, verdict, content_alignment, tool_usage, missing_tools, incorrect_tool_uses, evidence, notes. Use double quotes and no trailing commas.\n\nValidity rules: standard JSON only (no markdown or free-text outside the JSON). Provide 2–4 evidence strings (≤120 chars each). Notes must always contain a concise (≤300 chars) insight: for passes, highlight strengths; for fails, state the key fix. If information is insufficient, still return the full object with zeros and empty arrays and put the reason in notes."""

HUMAN_PROMPT = """Evaluate this run per the rubric and return ONLY the JSON object.\n\n<email_markdown>\n{email_markdown}\n</email_markdown>\n\n<assistant_reply>\n{assistant_reply}\n</assistant_reply>\n\n<tool_trace>\n{tool_trace}\n</tool_trace>\n\n<tool_calls_summary>\n{tool_calls_summary}\n</tool_calls_summary>\n\n<tool_calls_json>\n{tool_calls_json}\n</tool_calls_json>\n\n<raw_output_optional>\n{raw_output_optional}\n</raw_output_optional>\n"""


def _default_model_name() -> str:
    """
    Return the configured model name for the correctness judge.
    
    Reads the EMAIL_ASSISTANT_JUDGE_MODEL environment variable and returns its value if set; otherwise returns "gemini-2.5-flash".
    """
    return os.getenv("EMAIL_ASSISTANT_JUDGE_MODEL", "gemini-2.5-flash")


@lru_cache(maxsize=4)
def _build_base_chain(model_name: str, temperature: float) -> Runnable:
    """
    Builds a Runnable LangChain chain that prompts an LLM and parses its JSON output into a JudgeResult.
    
    Parameters:
        model_name (str): Name of the LLM model to instantiate.
        temperature (float): Sampling temperature for the LLM.
    
    Returns:
        Runnable: A runnable pipeline combining the system and human prompts, the configured LLM, and a JSON output parser targeting `JudgeResult`.
    """
    llm = get_llm(
        model=model_name,
        temperature=temperature,
        max_output_tokens=4096,
    )
    parser = JsonOutputParser(pydantic_object=JudgeResult)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )
    return prompt | llm | parser


def build_correctness_judge_chain(
    *, model_name: Optional[str] = None, temperature: float = 0.1
) -> Runnable:
    """Return a cached runnable that executes the correctness judge."""

    name = model_name or _default_model_name()
    return _build_base_chain(name, temperature)


def _coalesce_placeholder(value: Optional[str], placeholder: str) -> str:
    """Return the original string when it has content, otherwise the placeholder."""

    if value is None:
        return placeholder
    if isinstance(value, str):
        if value.strip():
            return value
        return placeholder
    return str(value)


def run_correctness_judge(
    *,
    email_markdown: str,
    assistant_reply: str,
    tool_trace: str,
    tool_calls_summary: str = "",
    tool_calls_json: str = "",
    raw_output_optional: str = "",
    parent_run_id: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.1,
) -> JudgeResult:
    """
    Run the Gemini-based correctness judge on an assistant reply and return its structured verdict.
    
    Executes the configured LLM judge with provided email content, assistant reply, and tool trace/context, normalizes the judge output into a JudgeResult, and attaches feedback to LangSmith runs when available.
    
    Parameters:
        email_markdown (str): The email content in markdown used as the judged context.
        assistant_reply (str): The assistant's final reply to be evaluated.
        tool_trace (str): A textual trace of tool usage and results.
        tool_calls_summary (str): Short human-readable summary of tool calls (used for prompt context).
        tool_calls_json (str): JSON payload describing tool call entries (defaults to "[]").
        raw_output_optional (str): Optional raw model or run output to include for context.
        parent_run_id (Optional[str]): LangSmith run id used to attach feedback; if omitted an attempt is made to locate a matching run.
        model_name (Optional[str]): Override for the judge model name; defaults to configured model.
        temperature (float): Sampling temperature for the judge LLM.
    
    Returns:
        JudgeResult: Structured evaluation including overall_correctness, verdict, content_alignment, tool_usage, missing_tools, incorrect_tool_uses, evidence, and notes.
    
    Raises:
        JudgeUnavailableError: If required configuration (e.g., GOOGLE_API_KEY) is missing, if the judge produces unparsable output, or if invocation otherwise fails.
    """

    if not os.getenv("GOOGLE_API_KEY"):
        raise JudgeUnavailableError(
            "GOOGLE_API_KEY missing – cannot call Gemini-based judge."
        )

    chain = build_correctness_judge_chain(model_name=model_name, temperature=temperature)

    payload = {
        "email_markdown": _coalesce_placeholder(email_markdown, "(no email_markdown provided)"),
        "assistant_reply": _coalesce_placeholder(assistant_reply, "(assistant_reply empty)"),
        "tool_trace": _coalesce_placeholder(tool_trace, "(tool_trace empty)"),
        "tool_calls_summary": _coalesce_placeholder(tool_calls_summary, "(no tool calls)"),
        "tool_calls_json": tool_calls_json or "[]",
        "raw_output_optional": _coalesce_placeholder(raw_output_optional, "(raw output omitted)"),
    }

    trace_project = os.getenv("EMAIL_ASSISTANT_JUDGE_PROJECT", "email-assistant-judge")
    should_trace = bool(os.getenv("LANGSMITH_API_KEY") and trace_project)

    def _invoke(data):
        return chain.invoke(data)

    if should_trace:
        traced = traceable(
            run_type="chain",
            name="gemini_correctness_judge",
            project_name=trace_project,
        )(_invoke)
        invoke_fn = traced
    else:
        invoke_fn = _invoke

    try:
        judge_summary = "Judge review"
        reply_snippet = _truncate_text(strip_markdown_to_text(assistant_reply), limit=120)
        if reply_snippet:
            judge_summary = f"{judge_summary} | reply: {reply_snippet}"
        tools_snippet = (
            _truncate_text(strip_markdown_to_text(tool_calls_summary), limit=80)
            if tool_calls_summary
            else ""
        )
        if tools_snippet:
            judge_summary = f"{judge_summary} | tools: {tools_snippet}"

        def _summarize_output(raw_result: object) -> str | None:
            verdict = None
            score = None
            if isinstance(raw_result, JudgeResult):
                verdict = raw_result.verdict
                score = raw_result.overall_correctness
            elif isinstance(raw_result, dict):
                verdict = raw_result.get("verdict")
                score = raw_result.get("overall_correctness")
            if verdict is None and score is None:
                return "[judge]"
            details: list[str] = ["[judge]"]
            if verdict:
                details.append(f"verdict={verdict}")
            if score is not None:
                try:
                    details.append(f"score={float(score):.2f}")
                except Exception:
                    details.append(f"score={score}")
            return " ".join(details)

        root_token = None
        try:
            active = False
            try:
                active = _JUDGE_ROOT_ACTIVE.get()
            except LookupError:
                active = False

            if not active and not _inside_judge_root():
                root_token = _JUDGE_ROOT_ACTIVE.set(True)
                raw = invoke_with_root_run(
                    lambda: invoke_fn(payload),
                    root_name="judge:gemini_correctness",
                    input_summary=judge_summary,
                    metadata={"payload_keys": list(payload.keys())},
                    output_transform=_summarize_output,
                    project_name=trace_project,
                )
            else:
                raw = invoke_fn(payload)
        except OutputParserException as exc:
            llm_output = getattr(exc, "llm_output", None)
            snippet = (llm_output or "").strip().splitlines()[0][:160] if llm_output else ""
            message = (
                "Judge produced non-JSON output, skipping evaluation."
                if snippet
                else "Judge returned an empty response; treating evaluation as unavailable."
            )
            if snippet:
                message += f" First line: {snippet}"
            raise JudgeUnavailableError(message) from exc
        finally:
            if root_token is not None:
                _JUDGE_ROOT_ACTIVE.reset(root_token)

        if isinstance(raw, JudgeResult):
            result = raw
        elif isinstance(raw, dict):
            normalised = _normalise_result_dict(raw)
            result = JudgeResult(**normalised)
        else:
            raise TypeError(f"Unexpected judge payload type: {type(raw)}")

        _record_feedback(
            result,
            parent_run_id=parent_run_id,
            email_markdown=email_markdown,
        )
        return result
    except Exception as exc:  # pragma: no cover - defensive logging
        raise JudgeUnavailableError(f"Judge invocation failed: {exc}") from exc


def serialise_messages(messages: Iterable[object]) -> str:
    """Serialize LangChain/primitive messages to JSON for judge context."""

    serialisable = []
    for message in messages or []:
        if hasattr(message, "model_dump"):
            try:
                serialisable.append(message.model_dump())
                continue
            except Exception:  # pragma: no cover - fall back
                pass
        if hasattr(message, "dict"):
            try:
                serialisable.append(message.dict())  # type: ignore[attr-defined]
                continue
            except Exception:  # pragma: no cover - fall back
                pass
        serialisable.append(str(message))

    try:
        return json.dumps(serialisable, ensure_ascii=False)
    except TypeError:
        return json.dumps([str(item) for item in serialisable], ensure_ascii=False)


def _truncate_text(text: str, limit: int = 200) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "…"


def _compact_json(value: Any, limit: int = 200) -> str:
    if isinstance(value, str):
        return _truncate_text(value, limit)
    try:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        payload = str(value)
    return _truncate_text(payload, limit)


def _message_role(message: object) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or message.get("type") or "").lower()
    return str(
        getattr(message, "role", None)
        or getattr(message, "type", None)
        or ""
    ).lower()


def _tool_calls_from_message(message: object) -> Optional[List[Dict[str, Any]]]:
    if isinstance(message, dict):
        calls = message.get("tool_calls")
    else:
        calls = getattr(message, "tool_calls", None)
    if not calls:
        return None
    return list(calls)


def _tool_result_payload(message: object) -> Tuple[str, str]:
    role = _message_role(message)
    if role != "tool":
        return "", ""
    if isinstance(message, dict):
        tool_call_id = str(message.get("tool_call_id") or message.get("id") or "")
        raw_content = message.get("content", "")
        if isinstance(raw_content, (list, dict)):
            content = _compact_json(raw_content)
        else:
            content = _truncate_text(raw_content)
    else:
        tool_call_id = str(
            getattr(message, "tool_call_id", None)
            or getattr(message, "id", None)
            or ""
        )
        raw_content = extract_message_content(message)
        content = _truncate_text(raw_content)
    return tool_call_id, content


def _build_tool_call_context(messages: Iterable[object]) -> Tuple[str, str]:
    """
    Builds a human-readable summary and a compact JSON payload describing tool calls found in the provided messages.
    
    Scans each message for tool call entries and collects for each call: a step index, the tool name, a compacted `args` representation, and an optional `result` (when a corresponding tool result payload is available). If no messages or no tool calls are found, returns the sentinel summary "(no tool calls)" and the JSON string "[]".
    
    Parameters:
        messages (Iterable[object]): An iterable of message-like objects to inspect for tool calls. Messages may be dicts or objects with tool call structures; the function tolerates varied shapes and will coerce non-dict call entries to strings.
    
    Returns:
        tuple:
            summary (str): A newline-separated human-readable list of tool calls with step numbers, names, args, and optional results (or "(no tool calls)" when none).
            tool_calls_json (str): A compact JSON string representing an array of entries, where each entry is an object with keys:
                - "step" (int): sequential index of the call in the scanned messages
                - "name" (str): tool name
                - "args" (str): compacted JSON/text representation of arguments
                - "result" (str, optional): truncated/compacted result payload when available
    """
    messages = list(messages or [])
    if not messages:
        return "(no tool calls)", "[]"

    tool_results: Dict[str, str] = {}
    for message in messages:
        tool_call_id, content = _tool_result_payload(message)
        if tool_call_id and content:
            tool_results[tool_call_id] = content

    entries: List[Dict[str, Any]] = []
    summary_lines: List[str] = []
    index = 1
    for message in messages:
        tool_calls = _tool_calls_from_message(message)
        if not tool_calls:
            continue
        for call in tool_calls:
            if not isinstance(call, dict):
                call = {"name": str(call), "args": str(call)}
            name = str(call.get("name", ""))
            args = call.get("args", {})
            args_text = _compact_json(args, limit=600)
            call_id = str(call.get("id") or call.get("tool_call_id") or "")
            result_text = tool_results.get(call_id, "")
            entry: Dict[str, Any] = {"step": index, "name": name, "args": args_text}
            if result_text:
                entry["result"] = result_text
                summary_lines.append(
                    f"{index}. {name} args={args_text} → result={result_text}"
                )
            else:
                summary_lines.append(f"{index}. {name} args={args_text}")
            entries.append(entry)
            index += 1

    if not entries:
        return "(no tool calls)", "[]"

    summary = "\n".join(summary_lines)
    tool_calls_json = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
    return summary, tool_calls_json


def build_tool_call_context(messages: Iterable[object]) -> Tuple[str, str]:
    """Public wrapper that returns (summary_text, json_payload) for tool calls."""

    return _build_tool_call_context(messages)


def _normalise_result_dict(data: dict) -> dict:
    """
    Normalize a raw judge result dictionary into the canonical structure expected by the evaluator.
    
    This enforces types and bounds, sanitizes list entries, trims long text fields, and computes missing aggregate fields so the returned dict conforms to the JudgeResult contract used elsewhere in the module.
    
    Parameters:
        data (dict): Raw result data from the LLM judge/parser. May include keys such as
            "incorrect_tool_uses", "missing_tools", "evidence", "content_alignment",
            "tool_usage", "overall_correctness", "verdict", and "notes".
    
    Returns:
        dict: A normalized copy of `data` where:
          - "incorrect_tool_uses" entries are guaranteed to be dicts with at least "tool" and "why".
          - "missing_tools" is a list.
          - "evidence" contains up to four trimmed string entries (120 chars max each).
          - "content_alignment" and "tool_usage" are coerced to integers when present.
          - "tool_usage" is set or capped appropriately when there are missing or incorrect tool uses.
          - "overall_correctness" is recomputed from content_alignment (60%) and tool_usage (40%) when both are available, otherwise coerced to float if present.
          - "verdict" is lowercased and forced to "fail" when tool usage indicates an issue.
          - "notes" is populated with a sensible default message when empty and trimmed to 300 characters.
    """
    result = data.copy()
    incorrect = result.get("incorrect_tool_uses") or []
    normalised_incorrect = []
    for item in incorrect:
        if isinstance(item, dict):
            normalised_incorrect.append(item)
        else:
            normalised_incorrect.append({"tool": "", "why": str(item)})
    result["incorrect_tool_uses"] = normalised_incorrect
    missing_tools = list(result.get("missing_tools") or [])
    result["missing_tools"] = missing_tools
    evidence = [str(e) for e in (result.get("evidence") or [])]
    trimmed = [e[:120] for e in evidence][:4]
    if len(trimmed) < 2 and evidence:
        trimmed = trimmed[: max(1, len(trimmed))]
    result["evidence"] = trimmed
    content_alignment_val = result.get("content_alignment")
    tool_usage_val = result.get("tool_usage")
    if content_alignment_val is not None:
        content_alignment_val = int(content_alignment_val)
        result["content_alignment"] = content_alignment_val
    if tool_usage_val is not None:
        tool_usage_val = int(tool_usage_val)
    if tool_usage_val is None and (missing_tools or normalised_incorrect):
        tool_usage_val = 1
    if (missing_tools or normalised_incorrect) and tool_usage_val is not None and tool_usage_val > 2:
        tool_usage_val = 2
    if tool_usage_val is not None:
        result["tool_usage"] = tool_usage_val
    if content_alignment_val is not None and tool_usage_val is not None:
        recalculated = 0.6 * (content_alignment_val / 5) + 0.4 * (tool_usage_val / 5)
        result["overall_correctness"] = float(recalculated)
    elif "overall_correctness" in result:
        result["overall_correctness"] = float(result["overall_correctness"])
    if "content_alignment" in result:
        result["content_alignment"] = int(result["content_alignment"])
    if "verdict" in result:
        result["verdict"] = str(result["verdict"]).lower()
    if tool_usage_val is not None and tool_usage_val < 3:
        result["verdict"] = "fail"
    notes = str(result.get("notes") or "").strip()
    verdict = str(result.get("verdict", "")).lower()
    if not notes:
        if verdict == "pass":
            notes = "Pass: reply and tool usage met the request; keep matching user constraints."
        elif verdict == "fail":
            notes = "Fail: review evidence and address the highlighted tool/content issues."
        else:
            notes = "No additional notes provided."
    result["notes"] = notes[:300]
    return result


def _record_feedback(
    result: JudgeResult,
    *,
    parent_run_id: Optional[str] = None,
    email_markdown: Optional[str] = None,
) -> None:
    """
    Attach judge feedback to a LangSmith run when available.
    
    Locate an appropriate run (using parent_run_id or the current run tree), attempt to resolve the most relevant agent run by matching email content or fingerprint, and create feedback entries describing missing tools, incorrect tool uses, numeric metrics (tool_usage, content_alignment, overall_correctness), evidence, notes, and the final verdict. Attachment is best-effort: it communicates with the LangSmith API, may delay briefly to influence feedback ordering, and silently ignores any errors so callers are not disrupted.
    
    Parameters:
        result (JudgeResult): Normalized judge result containing scores, verdict, evidence, notes, and tool issue details.
        parent_run_id (Optional[str]): Optional fallback run id used as the root for resolving the target agent run. If omitted, the current run tree is used.
        email_markdown (Optional[str]): Optional email markdown used to match runs by fingerprint or content when resolving the correct agent run to attach feedback to.
    """

    try:
        base_run_id = parent_run_id
        if not base_run_id:
            run_tree = get_current_run_tree()
            if not run_tree:
                return
            base_run_id = run_tree.id
        base_run_id = str(base_run_id)
        client = Client()

        def _ancestor_chain(start_id: str, max_depth: int = 6) -> List[Tuple[str, str]]:
            """
            Builds an ancestor chain of run IDs and names starting from `start_id` up through parent runs up to `max_depth`.
            
            Parameters:
                start_id (str): The starting run ID to trace upward.
                max_depth (int): Maximum number of ancestry steps to follow (default 6).
            
            Returns:
                List[Tuple[str, str]]: Ordered list of `(run_id, name)` tuples beginning with `start_id` and moving to its parents. If a run cannot be read, its ID is included with an empty name and traversal stops. Cycles or missing IDs terminate the chain early.
            """
            chain: List[Tuple[str, str]] = []
            current_id = start_id
            visited: set[str] = set()
            for _ in range(max_depth):
                if not current_id or current_id in visited:
                    break
                visited.add(current_id)
                try:
                    current = client.read_run(current_id)
                except Exception:
                    chain.append((current_id, ""))
                    break
                name = str(getattr(current, "name", "") or "")
                chain.append((str(current.id), name))
                parent_id = getattr(current, "parent_run_id", None)
                if not parent_id:
                    break
                current_id = str(parent_id)
            return chain

        target_markdown = truncate_markdown(email_markdown) if email_markdown else None
        target_fingerprint = email_fingerprint(email_markdown)

        def _matches_markdown(run: Run) -> bool:
            """
            Checks whether a LangSmith run matches the target email content by fingerprint or markdown.
            
            First, if a target fingerprint is available, compares it to the run's `metadata.email_fingerprint` or `extra.email_fingerprint` and returns `True` on match. If no fingerprint match and a target markdown is provided, compares the run's `metadata.email_markdown` or `extra.email_markdown` for exact equality or substring containment in either direction.
            
            Returns:
                bool: `True` if the run matches by fingerprint or markdown, `False` otherwise.
            """
            candidate_fp = None
            if target_fingerprint:
                candidate_fp = (
                    getattr(run, "metadata", None) or {}
                ).get("email_fingerprint")
                if not candidate_fp:
                    candidate_fp = (
                        getattr(run, "extra", None) or {}
                    ).get("email_fingerprint")
                if candidate_fp and candidate_fp == target_fingerprint:
                    return True

            if not target_markdown:
                return False
            metadata = getattr(run, "metadata", None) or {}
            candidate_md = metadata.get("email_markdown")
            if not candidate_md:
                extra = getattr(run, "extra", None) or {}
                candidate_md = extra.get("email_markdown")
            if not candidate_md:
                return False
            candidate_md_str = str(candidate_md)
            return (
                candidate_md_str == target_markdown
                or target_markdown in candidate_md_str
                or candidate_md_str in target_markdown
            )

        def _resolve_agent_run(start_id: str, depth: int = 0) -> str:
            """
            Resolve the most appropriate agent-related run id for a given starting run by searching children and related session runs.
            
            Searches downward from the provided start_id (up to a recursion depth of 3) to locate a run whose name indicates an agent/email_assistant run and that best matches the target content fingerprint or markdown when available. The resolution prioritizes:
            - direct child runs with agent-like names (most recent first),
            - recursively resolved children,
            - candidate runs within the same session/project matched by content fingerprint or markdown, reference_example_id, trace_id, or closest start time within 180 seconds (preferring runs that started at or after the start run).
            If a matching agent run cannot be found, the function may return a pending agent child id (if one exists and no explicit target markdown/fingerprint is provided) or the original start_id.
            
            Parameters:
                start_id (str): The run id from which to begin resolution.
                depth (int): Current recursion depth (used internally); recursion stops and returns start_id when greater than 3.
            
            Returns:
                str: The resolved agent run id, or the original start_id if no better candidate is found.
            """
            if depth > 3:
                return start_id
            try:
                current = client.read_run(start_id)
            except Exception:
                return start_id

            name = str(getattr(current, "name", "") or "")
            normalised = name.lower()
            if normalised.startswith("agent:") or normalised.startswith("email_assistant"):
                if _matches_markdown(current):
                    return start_id
                return start_id

            reference_example_id = getattr(current, "reference_example_id", None)
            trace_id = getattr(current, "trace_id", None)
            start_time = getattr(current, "start_time", None)

            try:
                children = list(
                    client.list_runs(parent_run_id=start_id, limit=50, order="asc")
                )
            except Exception:
                children = []

            def _sorted_children(runs: List[Run]) -> List[Run]:
                # Prioritise the most recent children so feedback targets the
                # run that just completed rather than earlier siblings rolled
                # up under the same parent (e.g., parametrised pytest case).
                """
                Return the given runs sorted with the most recently started first.
                
                Parameters:
                    runs (List[Run]): Iterable of Run objects; each may have a `start_time` attribute.
                
                Returns:
                    List[Run]: The input runs sorted by `start_time` in descending order. Runs with missing or unparsable `start_time` are treated as oldest.
                """
                def _start_time(run: Run) -> float:
                    ts = getattr(run, "start_time", None)
                    if ts is None:
                        return float("-inf")
                    try:
                        return ts.timestamp()
                    except Exception:
                        return float("-inf")

                return sorted(runs, key=_start_time, reverse=True)

            pending_agent_id: Optional[str] = None
            for child in _sorted_children(children):
                child_name = str(getattr(child, "name", "") or "")
                child_norm = child_name.lower()
                if child_norm.startswith("agent:") or child_norm.startswith("email_assistant"):
                    if not _matches_markdown(child):
                        try:
                            child = client.read_run(str(child.id))
                        except Exception:
                            pass
                    if _matches_markdown(child):
                        return str(child.id)
                    if pending_agent_id is None:
                        pending_agent_id = str(child.id)

            for child in _sorted_children(children):
                resolved = _resolve_agent_run(str(child.id), depth + 1)
                if resolved != str(child.id):
                    return resolved

            session_id = getattr(current, "session_id", None)
            if session_id:
                try:
                    best_match: Optional[Run] = None
                    best_delta: float = float("inf")
                    for candidate_run in iter_experiment_runs(
                        client=client,
                        project_id=session_id,
                        preview=True,
                        limit=200,
                    ):
                        candidate_name = str(getattr(candidate_run, "name", "") or "").lower()
                        if candidate_name.startswith("agent:") or candidate_name.startswith(
                            "email_assistant"
                        ):
                            if not _matches_markdown(candidate_run):
                                try:
                                    candidate_run = client.read_run(candidate_run.id)
                                except Exception:
                                    pass
                            if _matches_markdown(candidate_run):
                                return str(candidate_run.id)
                            candidate_ref = getattr(candidate_run, "reference_example_id", None)
                            if reference_example_id and candidate_ref == reference_example_id:
                                return str(candidate_run.id)
                            candidate_trace = getattr(candidate_run, "trace_id", None)
                            if trace_id and candidate_trace == trace_id:
                                return str(candidate_run.id)
                            candidate_start = getattr(candidate_run, "start_time", None)
                            if start_time and candidate_start:
                                try:
                                    delta = abs((candidate_start - start_time).total_seconds())
                                except Exception:
                                    delta = float("inf")
                                else:
                                    # Prefer runs that started at or after the
                                    # current run; older runs are likely
                                    # unrelated siblings.
                                    if (
                                        candidate_start >= start_time
                                        and delta < best_delta
                                        and delta <= 180
                                    ):
                                        best_delta = delta
                                        best_match = candidate_run
                    if best_match is not None:
                        return str(best_match.id)
                except Exception:
                    pass

            if (
                pending_agent_id is not None
                and not target_fingerprint
                and not target_markdown
            ):
                return pending_agent_id

            return start_id

        target_run_id = _resolve_agent_run(base_run_id)
        ancestors = _ancestor_chain(base_run_id)
        agent_ancestor_id: Optional[str] = None
        for ancestor_id, name in ancestors:
            if name.lower().startswith("agent:") or name.lower().startswith(
                "email_assistant"
            ):
                agent_ancestor_id = ancestor_id
                break
        run_ids: list[str] = []
        for candidate in (base_run_id, target_run_id, agent_ancestor_id):
            if candidate and candidate not in run_ids:
                run_ids.append(candidate)

        def _attach_feedback(run_id: str) -> None:
            """
            Attach judge feedback entries to the LangSmith run identified by `run_id`.
            
            Emits structured feedback for missing tools, incorrect tool uses, metric scores (tool_usage, content_alignment, overall_correctness), evidence snippets, notes, and the final verdict. Exceptions during feedback submission are suppressed to avoid interrupting the caller; a small configurable delay may be applied between submissions to influence UI ordering.
            
            Parameters:
                run_id (str): LangSmith run identifier to which feedback entries will be attached.
            """
            def _safe_feedback(**kwargs: Any) -> None:
                try:
                    client.create_feedback(run_id=run_id, **kwargs)
                    if _FEEDBACK_PRIORITY_DELAY > 0:
                        time.sleep(_FEEDBACK_PRIORITY_DELAY)
                except Exception:
                    pass

            # Diagnostics lowest priority so they appear at the bottom of the UI.
            if result.missing_tools:
                _safe_feedback(
                    key="missing_tools",
                    value=", ".join(result.missing_tools),
                    comment="Tools the agent failed to invoke",
                )

            if result.incorrect_tool_uses:
                for issue in result.incorrect_tool_uses:
                    # Emit the explanation first so the follow-up tool label
                    # appears above it in the LangSmith UI (newest-first).
                    _safe_feedback(
                        key="why",
                        value=issue.why or "See comment",
                        comment=(
                            f"Tool '{issue.tool}' flagged by judge"
                            if issue.tool
                            else "Incorrect tool usage details"
                        ),
                    )
                    _safe_feedback(
                        key="tool",
                        value=issue.tool or "(unknown)",
                        comment=issue.why or "Incorrect tool usage detected",
                    )

            # Metrics next.
            metric_entries = [
                (
                    "tool_usage",
                    result.tool_usage,
                    "Tool usage score (0-5)",
                ),
                (
                    "content_alignment",
                    result.content_alignment,
                    "Content alignment score (0-5)",
                ),
                (
                    "overall_correctness",
                    result.overall_correctness,
                    "Weighted correctness (0..1)",
                ),
            ]
            for key, value, comment in metric_entries:
                if value is None:
                    continue
                _safe_feedback(
                    key=key,
                    score=float(value),
                    value=str(value),
                    comment=comment,
                )

            # Evidence bundle.
            if result.evidence:
                _safe_feedback(
                    key="evidence",
                    value=" | ".join(result.evidence),
                    comment="Judge evidence snippets",
                )

            # Notes ahead of verdict.
            if result.notes:
                _safe_feedback(
                    key="notes",
                    value=result.notes,
                    comment="Judge guidance",
                )

            # Verdict last so newest-first ordering shows it at the top.
            if result.verdict:
                _safe_feedback(
                    key="verdict",
                    value=result.verdict,
                    comment=f"Pass when overall_correctness ≥ 0.70 (scored {result.overall_correctness:.2f})",
                    extra=result.model_dump(),
                )

        for rid in run_ids:
            _attach_feedback(rid)

    except Exception:
        # Feedback attachment is best-effort; ignore failures so tests keep running.
        return


class GeminiJudgeStringEvaluator(StringEvaluator):
    """LangSmith-compatible string evaluator that uses the Gemini judge."""

    evaluation_name: str = "gemini_correctness_judge"
    input_key: str = "email_markdown"
    prediction_key: str = "assistant_reply"
    answer_key: Optional[str] = None
    grading_function = staticmethod(lambda *_: {})  # Unused; required by base class
    model_name: Optional[str] = None
    temperature: float = 0.1

    def evaluate_run(
        self,
        run: Run,
        example: Optional[Example] = None,
        evaluator_run_id: Optional[str] = None,
    ) -> EvaluationResult:
        if run.outputs is None:
            raise ValueError("Run outputs cannot be None for LLM judge.")

        outputs = run.outputs
        email_markdown = (
            outputs.get("email_markdown")
            or run.inputs.get("email_markdown")
            or ""
        )
        assistant_reply = (
            outputs.get("assistant_reply")
            or outputs.get("output")
            or ""
        )
        tool_trace = outputs.get("tool_trace") or ""
        messages = outputs.get("messages") or []
        if not tool_trace and messages:
            try:
                tool_trace = format_messages_string(messages)
            except Exception:  # pragma: no cover
                tool_trace = ""

        tool_calls_summary, tool_calls_json = _build_tool_call_context(messages)
        raw_messages = serialise_messages(messages)
        raw_output = _compact_json(
            {
                "message_count": len(messages),
                "messages_preview": raw_messages[:400],
            },
            limit=400,
        )

        parent_run_id = getattr(run, "id", None)

        result = run_correctness_judge(
            email_markdown=email_markdown,
            assistant_reply=assistant_reply,
            tool_trace=tool_trace,
            tool_calls_summary=tool_calls_summary,
            tool_calls_json=tool_calls_json,
            raw_output_optional=raw_output,
            parent_run_id=parent_run_id,
            model_name=self.model_name,
            temperature=self.temperature,
        )

        return EvaluationResult(
            key=self.evaluation_name,
            score=result.overall_correctness,
            value=result.verdict,
            comment=result.notes,
            evaluator_info={"judge": result.model_dump()},
        )


def create_langsmith_correctness_evaluator(
    *, model_name: Optional[str] = None, temperature: float = 0.1
) -> LangChainStringEvaluator:
    """Return a LangChainStringEvaluator wrapping the Gemini judge."""

    evaluator = GeminiJudgeStringEvaluator(
        model_name=model_name, temperature=temperature
    )
    return LangChainStringEvaluator(evaluator=evaluator)

_JUDGE_ROOT_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "email_assistant_judge_root_active", default=False
)


def iter_experiment_runs(
    *,
    experiment_name: Optional[str] = None,
    project_id: Optional[UUID | str] = None,
    client: Optional[Client] = None,
    preview: bool = True,
    limit: Optional[int] = None,
) -> Iterator[Run]:
    """Yield runs from a LangSmith experiment using the streaming results API.

    Args:
        experiment_name: Optional experiment name when you prefer lookup by name.
        project_id: Experiment session ID (UUID or string form).
        client: Existing LangSmith client (defaults to constructing a new one).
        preview: When True, request lightweight previews instead of full payloads.
        limit: Optional cap on total runs to yield.

    Returns:
        Iterator of LangSmith ``Run`` objects in dataset order.
    """

    resolved_client = client or Client()
    kwargs: Dict[str, Any] = {"preview": preview}
    if experiment_name:
        kwargs["name"] = experiment_name
    if limit is not None:
        kwargs["limit"] = limit
    if project_id:
        try:
            project_uuid = project_id if isinstance(project_id, UUID) else UUID(str(project_id))
        except Exception as exc:  # pragma: no cover - defensive conversion guard
            raise ValueError(f"Invalid project_id supplied: {project_id}") from exc
        kwargs["project_id"] = project_uuid

    try:
        results = resolved_client.get_experiment_results(**kwargs)
    except Exception:
        return

    examples = results.get("examples_with_runs") if isinstance(results, dict) else getattr(results, "examples_with_runs", None)
    if examples is None:
        return

    total_yielded = 0
    for example in examples:
        runs = getattr(example, "runs", None) or []
        for run in runs:
            yield run
            total_yielded += 1
            if limit is not None and total_yielded >= limit:
                return


def _inside_judge_root() -> bool:
    """Return True when the current LangSmith root run is already the judge."""

    try:
        tree = get_current_run_tree()
    except Exception:
        return False
    if not tree:
        return False

    root = tree
    while getattr(root, "parent_run", None) is not None:
        root = getattr(root, "parent_run")

    name = str(getattr(root, "name", "")).strip().lower()
    return name == "judge:gemini_correctness"
