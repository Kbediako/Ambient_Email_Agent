"""Utilities for consistent LangSmith tracing output."""
from __future__ import annotations

import contextvars
import logging
import os
import re
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence as SeqType

from datetime import datetime, timezone

try:  # pragma: no cover - zoneinfo only present on py3.9+
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - fallback when tzdata missing
    ZoneInfo = None  # type: ignore

from email_assistant.utils import extract_message_content, parse_gmail
from email_assistant import version as EMAIL_ASSISTANT_VERSION

# LangSmith recently refactored the tracing helpers. Import order keeps backward
# compatibility with older packages while preferring the newer ``trace`` API.
RunTree = None  # type: ignore[assignment]
get_current_run_tree = None  # type: ignore[assignment]
_TRACE_CONTEXT = None

try:  # pragma: no cover - optional dependency
    from langsmith.run_helpers import trace as _TRACE_CONTEXT  # type: ignore
except Exception:  # pragma: no cover - older versions may not expose ``trace``
    _TRACE_CONTEXT = None

try:  # pragma: no cover - optional dependency
    from langsmith.run_helpers import get_current_run_tree as _GET_CURRENT  # type: ignore
except Exception:  # pragma: no cover
    _GET_CURRENT = None

if _GET_CURRENT is not None:  # pragma: no cover - ensure shared reference
    get_current_run_tree = _GET_CURRENT

if _TRACE_CONTEXT is None:  # Fall back to legacy RunTree context manager
    try:  # pragma: no cover - legacy behaviour
        from langsmith.run_helpers import RunTree as _LEGACY_RUN_TREE  # type: ignore
    except Exception:  # pragma: no cover
        _LEGACY_RUN_TREE = None
    else:
        if hasattr(_LEGACY_RUN_TREE, "__enter__"):
            RunTree = _LEGACY_RUN_TREE
else:
    try:  # pragma: no cover - new API exposes RunTree from run_trees
        from langsmith.run_trees import RunTree as _RUN_TREE  # type: ignore
    except Exception:  # pragma: no cover - older versions still provide RunTree in run_helpers
        try:
            from langsmith.run_helpers import RunTree as _LEGACY_RUN_TREE  # type: ignore
        except Exception:  # pragma: no cover
            _RUN_TREE = None
        else:
            _RUN_TREE = _LEGACY_RUN_TREE
    if _RUN_TREE is not None:
        RunTree = _RUN_TREE


logger = logging.getLogger(__name__)


_TRACE_DEBUG = os.getenv("EMAIL_ASSISTANT_TRACE_DEBUG", "").lower() in ("1", "true", "yes")
_TRACE_TIMEZONE_NAME = os.getenv("EMAIL_ASSISTANT_TRACE_TIMEZONE", "Australia/Sydney")

_ROOT_RUN_TREE: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "email_assistant_root_run_tree", default=None
)


def _debug_log(message: str) -> None:
    if _TRACE_DEBUG:
        print(f"[trace-debug] {message}")

def _project_with_date(base: str) -> str:
    tzinfo = timezone.utc
    if ZoneInfo is not None:
        try:
            tzinfo = ZoneInfo(_TRACE_TIMEZONE_NAME)
        except Exception:
            logger.warning(
                "Unknown EMAIL_ASSISTANT_TRACE_TIMEZONE=%r; falling back to UTC",
                _TRACE_TIMEZONE_NAME,
            )
        else:
            # Ensure tzdata is available even if underlying OS lacks zoneinfo.
            if tzinfo is None:
                tzinfo = timezone.utc

    today = datetime.now(tzinfo).strftime("%Y%m%d")
    prefix, sep, suffix = base.partition(":")
    if sep:
        return f"{prefix}-{suffix.upper()}-{today}"
    return f"{base}-{today}"


AGENT_PROJECT = _project_with_date("email-assistant:agent")
JUDGE_PROJECT = _project_with_date("email-assistant:judge")

_HIDDEN_FLAGS = (
    "LANGSMITH_HIDE_INPUTS",
    "LANGSMITH_HIDE_OUTPUTS",
    "LANGCHAIN_HIDE_INPUTS",
    "LANGCHAIN_HIDE_OUTPUTS",
)


def init_project(project: str | None) -> None:
    """Initialise stable LangSmith/LangChain project settings."""

    if not project:
        return

    os.environ.setdefault("LANGSMITH_PROJECT", project)
    os.environ.setdefault("LANGCHAIN_PROJECT", project)
    # Quiet Gemini gRPC client noise (ALTS creds warnings) unless user overrides.
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

    for flag in _HIDDEN_FLAGS:
        if os.environ.get(flag):
            del os.environ[flag]

    try:
        from langsmith import utils as langsmith_utils  # type: ignore
    except Exception:
        langsmith_utils = None

    if langsmith_utils is not None and hasattr(langsmith_utils.get_env_var, "cache_clear"):
        langsmith_utils.get_env_var.cache_clear()


def _grid_text(text: str | None) -> str:
    """Return a safe, non-empty string for LangSmith grid cells."""

    value = (text or "").strip()
    return value if value else "[n/a]"


def _start_langsmith_run(
    name: str,
    *,
    run_type: str,
    inputs_summary: str | None,
    metadata: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    tags: SeqType[str] | None = None,
    parent: Any | None = None,
    project_name: str | None = None,
):
    """Create a LangSmith run using the available SDK helpers."""

    active_tags = list(tags or [])

    if _TRACE_CONTEXT is not None:
        payload_inputs: Mapping[str, Any] | None = None
        if inputs_summary is not None:
            payload_inputs = {"summary": _grid_text(inputs_summary)}

        ctx = _TRACE_CONTEXT(
            name,
            run_type=run_type,
            inputs=payload_inputs,
            metadata=dict(metadata or {}),
            extra=dict(extra or {}),
            tags=active_tags or None,
            parent=parent,
            project_name=project_name,
        )
        run = ctx.__enter__()
        return ctx, run

    if RunTree is not None and hasattr(RunTree, "__enter__"):
        inputs_payload: Mapping[str, Any] = {}
        if inputs_summary is not None:
            inputs_payload = {"summary": _grid_text(inputs_summary)}

        kwargs: dict[str, Any]
        if parent is not None:
            kwargs = {
                "name": name,
                "run_type": run_type,
                "inputs": inputs_payload,
                "metadata": dict(metadata or {}),
                "extra": dict(extra or {}),
                "tags": active_tags or None,
            }
            child = parent.create_child(**{k: v for k, v in kwargs.items() if v is not None})  # type: ignore[arg-type]
            run = child.__enter__()
            return child, run

        kwargs = {
            "name": name,
            "run_type": run_type,
            "inputs": inputs_payload,
            "metadata": dict(metadata or {}),
            "extra": dict(extra or {}),
        }

        if active_tags:
            kwargs["tags"] = active_tags
        if project_name:
            kwargs["project_name"] = project_name

        try:
            ctx = RunTree(**kwargs)
        except TypeError:
            kwargs.pop("tags", None)
            ctx = RunTree(**kwargs)

        run = ctx.__enter__()
        return ctx, run

    return None, None


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").lower() in ("1", "true", "yes")


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def default_trace_tags(extra: SeqType[str] | None = None) -> list[str]:
    """Return default tags describing the runtime environment."""

    tags: list[str] = []
    tags.append("eval" if _env_flag("EMAIL_ASSISTANT_EVAL_MODE") else "live")

    model = (
        os.getenv("EMAIL_ASSISTANT_MODEL")
        or os.getenv("GEMINI_MODEL")
        or os.getenv("GEMINI_MODEL_AGENT")
    )
    if model:
        tags.append(model)

    stage = os.getenv("EMAIL_ASSISTANT_TRACE_STAGE")
    if stage:
        tags.append(stage)

    extra_tokens = []
    if extra:
        extra_tokens.extend(extra)
    env_extra = os.getenv("EMAIL_ASSISTANT_TRACE_TAGS")
    if env_extra:
        extra_tokens.extend(part.strip() for part in env_extra.split(","))

    if extra_tokens:
        tags.extend(extra_tokens)

    return _dedupe_preserve_order(str(tag) for tag in tags if tag)


def default_root_metadata(
    *,
    agent_label: str | None = None,
    thread_id: str | None = None,
    run_label: str | None = None,
) -> dict[str, Any]:
    """Base metadata emitted on the root run for filtering/search."""

    metadata: dict[str, Any] = {
        "agent_version": EMAIL_ASSISTANT_VERSION,
        "eval_mode": _env_flag("EMAIL_ASSISTANT_EVAL_MODE"),
        "hitl_auto_accept": _env_flag("HITL_AUTO_ACCEPT"),
    }

    if agent_label:
        metadata["agent_label"] = agent_label
    if thread_id:
        metadata["thread_id"] = thread_id
    if run_label:
        metadata["run_label"] = run_label

    model = (
        os.getenv("EMAIL_ASSISTANT_MODEL")
        or os.getenv("GEMINI_MODEL")
        or os.getenv("GEMINI_MODEL_AGENT")
    )
    if model:
        metadata.setdefault("default_model", model)

    experiment = os.getenv("EMAIL_ASSISTANT_EXPERIMENT")
    if experiment:
        metadata["experiment"] = experiment

    return metadata


_markdown_tokens = re.compile(r"[`*_]{1,}|__")
_heading_tokens = re.compile(r"^\s*#+\s*", re.MULTILINE)
_quote_tokens = re.compile(r"^\s*>+\s*", re.MULTILINE)
_list_tokens = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+)", re.MULTILINE)
_link_pattern = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
_image_pattern = re.compile(r"!\[[^\]]*\]\([^\)]+\)")
_whitespace_collapse = re.compile(r"\s+")


def strip_markdown_to_text(markdown: Any) -> str:
    """Convert markdown-like content to plain text while preserving line breaks."""

    if markdown is None:
        return ""

    text = str(markdown).replace("\r\n", "\n")
    text = _image_pattern.sub("", text)
    text = _link_pattern.sub(r"\1", text)
    text = _heading_tokens.sub("", text)
    text = _quote_tokens.sub("", text)
    text = _list_tokens.sub("", text)
    text = _markdown_tokens.sub("", text)

    lines: list[str] = []
    for raw_line in text.split("\n"):
        collapsed = _whitespace_collapse.sub(" ", raw_line).strip()
        lines.append(collapsed)

    # Trim leading/trailing blank lines while keeping intentional spacing
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()

    return "\n".join(lines)


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    snippet = text[: max(limit - 3, 0)].rstrip()
    return f"{snippet}..."


def summarize_email_for_grid(email: Any) -> str:
    """Return compact email summary suitable for LangSmith Inputs column."""

    author, to_addr, subject, body, _email_id = parse_gmail(email or {})
    subject_text = strip_markdown_to_text(subject) or "(no subject)"
    from_text = strip_markdown_to_text(author) or "(unknown sender)"
    to_text = strip_markdown_to_text(to_addr) or "(unknown recipient)"

    header = f"{subject_text} - {from_text} -> {to_text}"
    body_text = strip_markdown_to_text(body)
    if not body_text:
        return header

    snippet = _shorten(body_text.replace("\n", " "), 800)
    return f"{header}\n{snippet}"


def _normalise_tool_name(name: str) -> str:
    return str(name or "unknown").strip()


def _format_email_like(value: str | None) -> str:
    if not value:
        return ""
    value = value.strip()
    if "@" not in value:
        return _shorten(value, 48)
    local, _, domain = value.partition("@")
    domain_display = domain.split(",", 1)[0]
    local_display = local[:32]
    if len(local) > 32:
        local_display = f"{local_display}..."
    return f"{local_display}@{domain_display}"


def _count_items(obj: Any) -> int | None:
    try:
        if isinstance(obj, (str, bytes)):
            return None
        if isinstance(obj, Mapping):
            return len(obj)
        return len(list(obj))  # type: ignore[arg-type]
    except Exception:
        return None


def summarize_tool_call_for_grid(name: str, args: Any) -> str:
    """Produce a terse tool summary for tracing grids."""

    display_name = _normalise_tool_name(name)
    lower_name = display_name.lower()

    if not isinstance(args, Mapping):
        arg_repr = strip_markdown_to_text(args)
        arg_repr = _shorten(arg_repr, 160) if arg_repr else ""
        if arg_repr:
            return f"[tool] {display_name} {arg_repr}"
        return f"[tool] {display_name}"

    items: list[str] = []

    if lower_name in {"send_email_tool", "write_email"}:
        to_value = args.get("to") or args.get("email_address")
        if to_value:
            items.append(f"to={_format_email_like(str(to_value))}")
        subject = args.get("subject")
        if subject:
            items.append(f"subject=\"{_shorten(strip_markdown_to_text(subject), 32)}\"")
        body = args.get("content") or args.get("response_text")
        if body:
            items.append(f"body_len={len(strip_markdown_to_text(body))}")
        attachments = args.get("attachments")
        count = _count_items(attachments)
        if count:
            items.append(f"attachments={count}")
    elif lower_name in {"schedule_meeting_tool", "schedule_meeting"}:
        start = args.get("start_time") or args.get("start")
        end = args.get("end_time") or args.get("end")
        duration = args.get("duration_minutes") or args.get("duration")
        attendees = _count_items(args.get("attendees"))
        if start:
            items.append(f"start={_shorten(str(start), 24)}")
        if end:
            items.append(f"end={_shorten(str(end), 24)}")
        if duration:
            items.append(f"duration={duration}")
        if attendees:
            items.append(f"attendees={attendees}")
    elif lower_name in {"check_calendar_tool", "check_calendar_availability"}:
        day = args.get("day") or args.get("range") or args.get("window")
        if day:
            items.append(f"window=\"{_shorten(strip_markdown_to_text(day), 40)}\"")
        tz = args.get("timezone")
        if tz:
            items.append(f"tz={_shorten(str(tz), 20)}")
    elif lower_name in {"mark_as_spam_tool", "mark_as_spam"}:
        message_id = args.get("message_id") or args.get("id")
        if message_id:
            items.append(f"id={_shorten(str(message_id), 32)}")
    elif lower_name in {"question", "ask"}:
        prompt = args.get("content") or args.get("question")
        if prompt:
            items.append(f"prompt=\"{_shorten(strip_markdown_to_text(prompt), 64)}\"")
    elif lower_name in {"done"}:
        outcome = args.get("outcome") or args.get("status")
        if outcome:
            items.append(f"status={_shorten(strip_markdown_to_text(outcome), 40)}")
    else:
        # Generic mapping: take up to three simple scalars
        for key, value in list(args.items())[:3]:
            if isinstance(value, (str, int, float)):
                cleaned = strip_markdown_to_text(value)
                items.append(f"{key}={_shorten(cleaned, 40)}")
            else:
                count = _count_items(value)
                if count is not None:
                    items.append(f"{key}_count={count}")

    arg_str = ", ".join(items)
    return f"[tool] {display_name}{(' ' + arg_str) if arg_str else ''}"


def _iter_messages(payload: Any) -> Iterable[Any]:
    if payload is None:
        return []
    if isinstance(payload, Mapping) and "messages" in payload:
        value = payload.get("messages")
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return value
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return payload
    return [payload]


def summarize_llm_for_grid(payload: Any) -> str:
    """Summarise LLM prompt/response payload for LangSmith grids."""

    messages = list(_iter_messages(payload))
    count = len(messages)
    if count == 0:
        return "0 msgs | no content"

    last_user = None
    last_role = None

    for message in reversed(messages):
        role = None
        if isinstance(message, Mapping):
            role = str(message.get("role") or message.get("type") or "").lower()
            content = message.get("content")
        else:
            role = str(getattr(message, "type", "") or getattr(message, "role", "")).lower()
            content = extract_message_content(message)

        text = strip_markdown_to_text(content)
        if text:
            if role in {"human", "user"} and last_user is None:
                last_user = text
                break
            if last_role is None:
                last_role = (role or "assistant", text)

    if last_user:
        return f"{count} msgs | last user: \"{_shorten(last_user, 96)}\""
    if last_role:
        return f"{count} msgs | last {last_role[0]}: \"{_shorten(last_role[1], 96)}\""
    return f"{count} msgs | no text"


def truncate_markdown(markdown: Any, max_chars: int = 4096) -> str:
    """Trim markdown to the LangSmith metadata safe length."""

    if markdown is None:
        return ""
    text = str(markdown)
    if len(text) <= max_chars:
        return text
    snippet = text[: max_chars - 3].rstrip()
    return f"{snippet}..."


def maybe_update_run_io(
    *,
    run_id: str | None,
    email_input: Any | None = None,
    llm_payload: Any | None = None,
    tool_name: str | None = None,
    tool_args: Any | None = None,
    outputs: Any | None = None,
    force_inputs: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    name: str | None = None,
    tags: SeqType[str] | None = None,
    append_tags: bool = False,
    update_metadata: bool = False,
    update_extra: bool = False,
    retries: int = 3,
    jitter: float = 0.25,
) -> bool:
    """Best-effort patch to ensure LangSmith grid cells contain readable text."""

    if not run_id:
        _debug_log("maybe_update_run_io: no run_id available")
        return False

    def _find_run_tree(start: Any, target: str | None) -> Any | None:
        if start is None:
            return None
        target_str = target or str(getattr(start, "id", ""))
        stack = [start]
        seen: set[int] = set()
        while stack:
            node = stack.pop()
            node_id = getattr(node, "id", None)
            if node_id is not None and str(node_id) == target_str:
                return node
            children = getattr(node, "child_runs", None) or []
            for child in children:
                if id(child) in seen:
                    continue
                seen.add(id(child))
                stack.append(child)
        return None

    run_tree_obj = None
    if get_current_run_tree is not None:
        try:
            current_tree = get_current_run_tree()
        except Exception:
            current_tree = None
        if current_tree is not None:
            target_str = str(run_id)
            current_id = getattr(current_tree, "id", None)
            if current_id is not None and str(current_id) == target_str:
                run_tree_obj = current_tree
            else:
                root = current_tree
                while getattr(root, "parent_run", None) is not None:
                    root = getattr(root, "parent_run")
                run_tree_obj = _find_run_tree(root, target_str)

    if run_tree_obj is None:
        try:
            cached_root = _ROOT_RUN_TREE.get()
        except LookupError:
            cached_root = None
        if cached_root is not None:
            run_tree_obj = _find_run_tree(cached_root, str(run_id))

    if run_tree_obj is None:
        _debug_log(f"maybe_update_run_io: run tree not found for {run_id}")
        return False

    def _as_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        return {}

    inputs_payload = _as_dict(getattr(run_tree_obj, "inputs", {}))
    outputs_payload = _as_dict(getattr(run_tree_obj, "outputs", {}))
    extra_payload = _as_dict(getattr(run_tree_obj, "extra", {}))
    metadata_payload = _as_dict(extra_payload.get("metadata", {}))

    _debug_log(
        "maybe_update_run_io: preparing update for "
        f"{getattr(run_tree_obj, 'name', '<unknown>')} ({getattr(run_tree_obj, 'id', '<no-id>')})"
    )

    changed = False

    if force_inputs is not None:
        inputs_payload["summary"] = _grid_text(force_inputs)
        changed = True
    else:
        summary_source: str | None = None
        if tool_name:
            summary_source = summarize_tool_call_for_grid(tool_name, tool_args or {})
        elif llm_payload is not None:
            summary_source = summarize_llm_for_grid(llm_payload)
        elif email_input is not None:
            summary_source = summarize_email_for_grid(email_input)
        if summary_source:
            summary_text = _grid_text(summary_source)
            existing = str(inputs_payload.get("summary", ""))
            if summary_text != existing:
                inputs_payload["summary"] = summary_text
                changed = True

    if outputs is not None:
        if isinstance(outputs, Mapping):
            outputs_payload.update(dict(outputs))
        elif isinstance(outputs, str):
            outputs_payload["summary"] = _grid_text(outputs)
        else:
            outputs_payload["summary"] = _grid_text(strip_markdown_to_text(outputs))
        changed = True
    elif llm_payload is not None and "summary" not in outputs_payload:
        summary_text = _grid_text(summarize_llm_for_grid(llm_payload))
        existing = str(outputs_payload.get("summary", ""))
        if summary_text != existing:
            outputs_payload["summary"] = summary_text
            changed = True

    if metadata and update_metadata:
        metadata_payload.update(metadata)
        changed = True

    if extra and update_extra:
        extra_payload.update(extra)
        changed = True

    if tags is not None:
        base_tags = list(getattr(run_tree_obj, "tags", []) or [])
        candidate = list(tags)
        if append_tags:
            candidate = base_tags + candidate
        deduped = _dedupe_preserve_order(str(tag) for tag in candidate if tag)
        if deduped != base_tags:
            run_tree_obj.tags = deduped
            changed = True

    if metadata_payload:
        extra_payload["metadata"] = metadata_payload

    if outputs_payload:
        run_tree_obj.outputs = outputs_payload

    if inputs_payload:
        run_tree_obj.inputs = inputs_payload

    if extra_payload:
        run_tree_obj.extra = extra_payload

    if name:
        run_tree_obj.name = str(name)

    return changed


def prime_parent_run(
    *,
    email_input: Any,
    email_markdown: str | None = None,
    metadata_update: Mapping[str, Any] | None = None,
    extra_update: Mapping[str, Any] | None = None,
    outputs: Any | None = None,
    agent_label: str | None = None,
    tags: SeqType[str] | None = None,
    thread_id: str | None = None,
    run_label: str | None = None,
) -> bool:
    """Prime the root LangSmith run with readable inputs/metadata."""

    if get_current_run_tree is None:
        return False

    try:
        run_tree = get_current_run_tree()
    except Exception:
        return False

    if not run_tree:
        return False

    parent = getattr(run_tree, "parent_run", None)
    _debug_log(
        "prime_parent_run: current run="
        f"{getattr(run_tree, 'name', '<unknown>')} ({getattr(run_tree, 'id', '<no-id>')})"
        f" parent={getattr(parent, 'name', None)}"
    )

    root_tree = run_tree
    while getattr(root_tree, "parent_run", None) is not None:
        root_tree = getattr(root_tree, "parent_run")

    if getattr(root_tree, "parent_run", None) is None:
        try:
            cached_root = _ROOT_RUN_TREE.get()
        except LookupError:
            cached_root = None
        else:
            if cached_root is not None:
                root_tree = cached_root

    run_id = getattr(root_tree, "id", None)
    if not run_id:
        return False

    base_metadata = default_root_metadata(
        agent_label=agent_label,
        thread_id=thread_id,
        run_label=run_label,
    )
    metadata_payload = dict(base_metadata)
    if metadata_update:
        metadata_payload.update(metadata_update)
    if email_markdown is not None:
        metadata_payload.setdefault("email_markdown", truncate_markdown(email_markdown))

    metadata_payload.setdefault("example_raw", email_input)

    extra_payload = dict(extra_update or {})
    extra_payload.setdefault("raw_input", email_input)

    summary = summarize_email_for_grid(email_input)

    updated = maybe_update_run_io(
        run_id=str(run_id),
        email_input=email_input,
        outputs=outputs,
        force_inputs=summary,
        metadata=metadata_payload,
        extra=extra_payload,
        name=agent_label,
        tags=default_trace_tags(tags),
        append_tags=True,
        update_metadata=True,
        update_extra=True,
    )
    if not updated:
        _debug_log("prime_parent_run: maybe_update_run_io returned False")
    return updated


@dataclass
class TraceRunHandle:
    """Helper allowing nodes to summarise outputs and attach artefacts."""

    _tree: Any
    _run: Any
    _outputs_summary: str | None = None
    _attachments: list[tuple[str, Any]] = field(default_factory=list)
    _closed: bool = False

    def set_outputs(self, summary: str | None) -> None:
        self._outputs_summary = summary

    def add_attachment(self, name: str, data: Any) -> None:
        self._attachments.append((str(name), data))

    def _finish(self, *, error: str | None = None) -> None:
        if self._closed:
            return
        try:
            if error:
                self._run.end(error=_grid_text(error))
            else:
                kwargs: dict[str, Any] = {}
                if self._outputs_summary is not None:
                    kwargs["outputs"] = {"summary": _grid_text(self._outputs_summary)}
                self._run.end(**kwargs)
        except Exception:
            pass

        for name, data in self._attachments:
            try:
                add_attachment = getattr(self._tree, "add_attachment", None)
                if callable(add_attachment):
                    add_attachment(name=name, data=data)
            except Exception:
                pass

        try:
            self._tree.__exit__(None, None, None)
        except Exception:
            pass
        self._closed = True


@contextmanager
def trace_stage(
    name: str,
    *,
    run_type: str = "chain",
    inputs_summary: str | None = None,
    tags: SeqType[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
):
    """Create a LangSmith child run for a logical stage.

    Yields a :class:`TraceRunHandle` when tracing is active; otherwise ``None``.
    """

    if (_TRACE_CONTEXT is None and RunTree is None) or get_current_run_tree is None:
        yield None
        return

    try:
        parent = get_current_run_tree()
    except Exception:
        parent = None

    if not parent:
        yield None
        return

    stage_tags = default_trace_tags(tags)

    ctx, run = _start_langsmith_run(
        name,
        run_type=run_type,
        inputs_summary=inputs_summary,
        metadata=metadata,
        extra=extra,
        tags=stage_tags,
        parent=parent,
    )

    if ctx is None or run is None:
        yield None
        return

    handle = TraceRunHandle(ctx, run)
    try:
        yield handle
    except Exception as exc:
        handle._finish(error=str(exc))
        raise
    else:
        handle._finish()


def _get_latest_child_run(run_type: str | None = None):
    if get_current_run_tree is None:
        return None
    try:
        tree = get_current_run_tree()
    except Exception:
        return None
    if not tree:
        return None
    root = tree
    while getattr(root, "parent_run", None) is not None:
        root = getattr(root, "parent_run")

    children = getattr(root, "children", None)
    if not children:
        children = getattr(root, "child_runs", None)
    if not children:
        return None
    for child in reversed(children):
        child_type = str(
            getattr(child, "run_type", "")
            or getattr(child, "type", "")
        ).lower()
        if run_type is None or child_type == run_type.lower():
            return child
    return None


def log_tool_child_run(
    *,
    name: str,
    args: Any,
    result: Any | None = None,
    metadata_update: Mapping[str, Any] | None = None,
) -> bool:
    """Ensure the latest tool child run has readable grid fields."""

    child = _get_latest_child_run("tool")
    if child is None:
        fallback = None
        if get_current_run_tree is not None:
            try:
                fallback = get_current_run_tree()
            except Exception:
                fallback = None
        if fallback is None:
            return False
        run_type = str(getattr(fallback, "run_type", "")).lower()
        if run_type != "tool":
            return False
        child = fallback

    run_id = getattr(child, "id", None)
    if not run_id:
        return False

    summary = summarize_tool_call_for_grid(name, args or {})

    metadata_payload = {"tool_raw": args}
    if metadata_update:
        metadata_payload.update(metadata_update)

    return maybe_update_run_io(
        run_id=str(run_id),
        tool_name=name,
        tool_args=args,
        outputs=result,
        force_inputs=summary,
        metadata=metadata_payload,
        update_metadata=True,
    )


def log_llm_child_run(
    *,
    prompt: Any,
    response: Any | None = None,
    metadata_update: Mapping[str, Any] | None = None,
) -> bool:
    """Ensure the latest LLM child run lists concise prompt text."""

    child = _get_latest_child_run("llm")
    if child is None:
        return False

    run_id = getattr(child, "id", None)
    if not run_id:
        return False

    summary = summarize_llm_for_grid(prompt)

    metadata_payload = {}
    if metadata_update:
        metadata_payload.update(metadata_update)

    return maybe_update_run_io(
        run_id=str(run_id),
        llm_payload=prompt,
        outputs=response,
        force_inputs=summary,
        metadata=metadata_payload,
        update_metadata=bool(metadata_payload),
    )


def invoke_with_root_run(
    func: Callable[[], Any],
    *,
    root_name: str,
    input_summary: str,
    metadata: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    output_transform: Callable[[Any], str | None] | None = None,
    project_name: str | None = None,
) -> Any:
    """Execute ``func`` inside a manually managed LangSmith root run."""

    tags = default_trace_tags()

    ctx, run = _start_langsmith_run(
        root_name,
        run_type="chain",
        inputs_summary=input_summary,
        metadata=metadata,
        extra=extra,
        tags=tags,
        project_name=project_name,
    )

    if ctx is None or run is None:
        return func()

    root_token = _ROOT_RUN_TREE.set(run)

    try:
        result = func()
        outputs_summary = None
        if output_transform is not None:
            try:
                outputs_summary = output_transform(result)
            except Exception:
                outputs_summary = None
        if outputs_summary is not None:
            run.end(outputs={"summary": _grid_text(outputs_summary)})
        else:
            run.end()
        return result
    except Exception as exc:
        run.end(error=_grid_text(str(exc)))
        raise
    finally:
        _ROOT_RUN_TREE.reset(root_token)
        ctx.__exit__(None, None, None)


def format_final_output(state: Mapping[str, Any]) -> str:
    """Return a two-line plain-text summary for LangSmith Outputs."""

    classification = str(state.get("classification_decision", "")).lower()
    messages = list(state.get("messages", []))

    last_tool = None
    for message in reversed(messages):
        tool_calls = None
        if isinstance(message, Mapping):
            tool_calls = message.get("tool_calls")
        else:
            tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in reversed(tool_calls):
            name = str(tc.get("name", "")).lower()
            if name in {"done", "finalize", "finish"}:
                continue
            last_tool = tc
            break
        if last_tool:
            break
    if last_tool:
        name = str(last_tool.get("name", "")).lower()
        args = last_tool.get("args") or {}
        if name in {"send_email_tool", "write_email"}:
            body = args.get("response_text") or args.get("content") or ""
            snippet = strip_markdown_to_text(body).replace("\n", " ")
            snippet = _shorten(snippet, 220)
            return f"[reply]\n{_grid_text(snippet)}"
        if name in {"mark_as_spam_tool", "mark_as_spam"}:
            reason = args.get("reason") or "Moved thread to Spam."
            reason_text = strip_markdown_to_text(reason) or "Moved thread to Spam."
            return f"[tool_call] mark_as_spam\n{_grid_text(reason_text)}"
        if name in {"schedule_meeting_tool", "schedule_meeting"}:
            info = args.get("subject") or args.get("start_time") or "Scheduled meeting."
            info_text = strip_markdown_to_text(info)
            return f"[tool_call] schedule_meeting\n{_grid_text(info_text)}"
        if name in {"check_calendar_tool", "check_calendar_availability"}:
            window = args.get("day") or args.get("range") or "Checked availability."
            window_text = strip_markdown_to_text(window)
            return f"[tool_call] check_calendar\n{_grid_text(window_text)}"
    if classification == "ignore":
        return "[no_action]\nIgnored after triage."
    if classification == "notify":
        return "[no_action]\nNotified user; no reply sent."
    if classification == "respond":
        return "[no_action]\nWorkflow ended without drafting a reply."
    return "[no_action]\nWorkflow completed."


__all__ = [
    "AGENT_PROJECT",
    "JUDGE_PROJECT",
    "init_project",
    "default_trace_tags",
    "default_root_metadata",
    "strip_markdown_to_text",
    "summarize_email_for_grid",
    "summarize_tool_call_for_grid",
    "summarize_llm_for_grid",
    "truncate_markdown",
    "maybe_update_run_io",
    "prime_parent_run",
    "log_tool_child_run",
    "log_llm_child_run",
    "invoke_with_root_run",
    "format_final_output",
    "trace_stage",
    "TraceRunHandle",
]
