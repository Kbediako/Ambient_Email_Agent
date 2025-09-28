from typing import List, Any
import json
import logging
import re

try:
    import html2text  # type: ignore
except ImportError:
    html2text = None


logger = logging.getLogger(__name__)

def format_email_markdown(subject, author, to, email_thread, email_id=None):
    """
    Create a markdown-formatted representation of an email.
    
    Includes Subject, From, To, an optional **ID** line when `email_id` is provided, the email thread/body, and a trailing separator.
    
    Parameters:
        subject: Email subject line.
        author: Email sender.
        to: Email recipient(s).
        email_thread: Body or thread content of the email.
        email_id: Optional identifier to include as an **ID** field.
    
    Returns:
        A markdown string containing the formatted email.
    """
    id_section = f"\n**ID**: {email_id}" if email_id else ""
    
    return f"""

**Subject**: {subject}
**From**: {author}
**To**: {to}{id_section}

{email_thread}

---
"""

def format_gmail_markdown(subject, author, to, email_thread, email_id=None):
    """
    Builds a markdown representation of a Gmail message, converting HTML bodies to plain text when detected.
    
    Parameters:
        subject (str): Email subject line.
        author (str): Email sender.
        to (str): Email recipient(s).
        email_thread (str): Message body; if it appears to contain HTML, it will be converted to plain text.
        email_id (Optional[str]): Optional message identifier to include in the output.
    
    Returns:
        str: A markdown-formatted string containing Subject, From, To, optional ID, the message body, and a trailing separator.
    """
    id_section = f"\n**ID**: {email_id}" if email_id else ""
    
    # Check if email_thread is HTML content and convert to text if needed
    if email_thread and (email_thread.strip().startswith("<!DOCTYPE") or 
                          email_thread.strip().startswith("<html") or
                          "<body" in email_thread):
        if html2text is not None:
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = True
            h.body_width = 0  # Don't wrap text
            email_thread = h.handle(email_thread)
        else:
            email_thread = re.sub(r"<[^>]+>", " ", email_thread)
    
    return f"""

**Subject**: {subject}
**From**: {author}
**To**: {to}{id_section}

{email_thread}

---
"""

def _normalise_tool_args(name: str, raw_args):
    """
    Normalize tool arguments for display and indicate any discarded input.
    
    Returns a tuple of (args_dict, residual). If `raw_args` is already a dict, it is returned as `args_dict` and `residual` is `None`. If `raw_args` is a non-empty list whose first element is a dict, the first element is returned and `residual` is `{"discarded": "<preview>"}`. If `raw_args` is None or any other non-dict shape, an empty dict is returned and `residual` contains a `{"discarded": "<preview>"}` string showing a truncated preview of the original value. The function logs a warning when the input shape is unexpected.
    """

    if isinstance(raw_args, dict):
        return raw_args, None

    preview = repr(raw_args)
    truncated_preview = preview[:200] if preview else preview

    if isinstance(raw_args, list) and raw_args and isinstance(raw_args[0], dict):
        logger.warning(
            "Agent Inbox: tool %s emitted list args; using first element.",
            name,
        )
        return raw_args[0], {"discarded": truncated_preview}

    if raw_args is None:
        logger.warning(
            "Agent Inbox: tool %s emitted None args; defaulting to empty mapping.",
            name,
        )
        return {}, {"discarded": "None"}

    logger.warning(
        "Agent Inbox: tool %s emitted non-dict args of type %s; discarding (preview=%s)",
        name,
        type(raw_args).__name__,
        truncated_preview,
    )
    return {}, {"discarded": truncated_preview}


def _clean_text(value, default="") -> str:
    """
    Produce a string representation of `value`, using `default` when `value` is `None` or an empty string.
    
    Parameters:
        value: The value to convert to a string. If `value` is `None` or `""`, `default` is used.
        default (str): Fallback string returned when `value` is `None` or an empty string.
    
    Returns:
        str: `str(value)` when `value` is not `None` or empty, otherwise `default`.
    """
    if value in (None, ""):
        return default
    return str(value)


def _clean_list(values) -> List[str]:
    """
    Normalize a sequence of values into a list of non-empty strings.
    
    Parameters:
        values: A list of items to normalize; non-list inputs produce an empty list.
    
    Returns:
        A list of strings containing the truthy items from `values` converted with `str()`. Falsy items are omitted.
    """
    if not isinstance(values, list):
        return []
    cleaned: List[str] = []
    for item in values:
        if not item:
            continue
        cleaned.append(str(item))
    return cleaned


def format_for_display(tool_call):
    """
    Render a tool call into a human-readable Markdown string for display in the Agent Inbox.
    
    The output adapts to common tool shapes:
    - For "write_email" or "send_email_tool": returns an Email Draft block with To, Subject, and body.
    - For "schedule_meeting" or "schedule_meeting_tool": returns a Calendar Invite block with meeting title, attendees, duration, and day/start.
    - For a tool named "question" (case-insensitive): returns a Question for User block containing the content.
    - For other tools: returns a Tool Call block showing the tool name and its arguments. If the original arguments had an unexpected shape, a note is appended indicating that some original args were discarded.
    
    Parameters:
        tool_call (dict-like): Mapping-like object representing the tool call; expected to contain at least a `name` and optionally `args`.
    
    Returns:
        str: A Markdown-formatted string representing the tool call for display.
    """
    # Initialize empty display
    display = ""
    
    name = tool_call.get("name", "")
    raw_args = tool_call.get("args")
    args, residual = _normalise_tool_args(name, raw_args)

    # Add tool call information
    if name in {"write_email", "send_email_tool"}:
        to_value = _clean_text(args.get("to") or args.get("email_address"), "")
        subject_value = _clean_text(args.get("subject"), "")
        body = args.get("content") or args.get("response_text") or ""
        display += f"""# Email Draft

**To**: {to_value}
**Subject**: {subject_value}

{body}
"""
    elif name in {"schedule_meeting", "schedule_meeting_tool"}:
        attendees = _clean_list(args.get("attendees"))
        attendees_display = ", ".join(attendees) if attendees else "N/A"
        subject_value = _clean_text(args.get("subject") or args.get("title"), "N/A")
        duration_value = _clean_text(args.get("duration_minutes"), "N/A")
        preferred_day = _clean_text(args.get("preferred_day") or args.get("start_time"), "N/A")
        display += f"""# Calendar Invite

**Meeting**: {subject_value}
**Attendees**: {attendees_display}
**Duration**: {duration_value} minutes
**Day**: {preferred_day}
"""
    elif str(name).lower() == "question":
        # Special formatting for questions to make them clear
        content = _clean_text(args.get("content"), "")
        display += f"""# Question for User

{content}
"""
    else:
        # Generic format for other tools
        display += f"""# Tool Call: {name}

Arguments:"""
        display_args = args if args else residual.get("discarded") if residual else raw_args
        if isinstance(display_args, dict):
            display += f"\n{json.dumps(display_args, indent=2)}\n"
        else:
            display += f"\n{display_args}\n"

    if residual and residual.get("discarded") and name not in {"Question"}:
        display += "\n_Note: Original tool args were discarded due to unexpected shape._\n"

    return display

def parse_email(email_input: dict) -> tuple[str, str, str, str]:
    """
    Parse an email-like dictionary and extract the author, recipient, subject, and message thread using common schema variants.
    
    Supports the repository dataset schema (author, to, subject, email_thread) and Gmail-like/alternate-capitalization schemas (from/From, To, Subject, body/Body/page_content).
    
    Returns:
        tuple[str, str, str, str]: (author, to, subject, email_thread) where missing values are returned as empty strings.
    """
    if not isinstance(email_input, dict):
        return ("", "", "", "")

    # Primary (dataset) keys
    author = email_input.get("author")
    to = email_input.get("to")
    subject = email_input.get("subject")
    thread = email_input.get("email_thread")

    # Fallbacks (Gmail-like / alternate capitalizations)
    if author is None:
        author = email_input.get("from") or email_input.get("From")
    if subject is None:
        subject = email_input.get("Subject")
    if to is None:
        to = email_input.get("To")
    if thread is None:
        thread = (
            email_input.get("body")
            or email_input.get("Body")
            or email_input.get("page_content")
        )

    return (author or "", to or "", subject or "", thread or "")

def extract_message_content(message) -> str:
    """Extract content from different message types as clean string.
    
    Args:
        message: A message object (HumanMessage, AIMessage, ToolMessage)
        
    Returns:
        str: Extracted content as clean string
    """
    content = message.content
    
    # Check for recursion marker in string
    if isinstance(content, str) and '<Recursion on AIMessage with id=' in content:
        return "[Recursive content]"
    
    # Handle string content
    if isinstance(content, str):
        return content
        
    # Handle list content (AIMessage format)
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and 'text' in item:
                text_parts.append(item['text'])
        return "\n".join(text_parts)
    
    # Don't try to handle recursion to avoid infinite loops
    # Just return string representation instead
    return str(content)

def format_few_shot_examples(examples):
    """
    Format a list of vector-store example items into a single readable string.
    
    Each example item is expected to have a `value` string containing three sections separated by the markers
    "Original routing:" and "Correct routing:". The resulting string contains one block per example:
    
    Example:
    Email: {email_part}
    Original Classification: {original_routing}
    Correct Classification: {correct_routing}
    ---
    
    Parameters:
        examples (List[Item]): Iterable of example items where `item.value` is a string containing
            the email text followed by "Original routing:" and "Correct routing:" sections.
    
    Returns:
        str: Joined formatted examples separated by newlines, one block per example as shown above.
    """
    formatted = []
    for example in examples:
        # Parse the example value string into components
        email_part = example.value.split('Original routing:')[0].strip()
        original_routing = example.value.split('Original routing:')[1].split('Correct routing:')[0].strip()
        correct_routing = example.value.split('Correct routing:')[1].strip()
        
        # Format into clean string
        formatted_example = f"""Example:
Email: {email_part}
Original Classification: {original_routing}
Correct Classification: {correct_routing}
---"""
        formatted.append(formatted_example)
    
    return "\n".join(formatted)

def extract_tool_calls(messages: List[Any]) -> List[str]:
    """
    Extract unique tool call names from a sequence of messages.
    
    Parses each message for a `tool_calls` collection (supports both dict and object message representations) and returns the list of tool call names normalized to lowercase in the order they are first encountered. Duplicate calls are suppressed by comparing either `(name, id)` when an id is present or `(name, serialized_args)` when no id is available; arguments are JSON-serialized for comparison with a fallback to `repr` when serialization fails.
    
    Parameters:
        messages (List[Any]): Iterable of message objects or dicts that may contain a `tool_calls` field.
    
    Returns:
        List[str]: Ordered list of unique tool call names in lowercase.
    """

    tool_call_names: List[str] = []
    seen_keys: set[tuple] = set()

    for message in messages:
        if isinstance(message, dict):
            tool_calls = message.get("tool_calls") or []
        else:
            tool_calls = getattr(message, "tool_calls", []) or []

        if not tool_calls:
            continue

        for call in tool_calls:
            # Support both dict and object-based tool call representations
            if isinstance(call, dict):
                name_value = call.get("name")
                call_id = call.get("id")
                args = call.get("args")
            else:
                name_value = getattr(call, "name", None)
                call_id = getattr(call, "id", None)
                args = getattr(call, "args", None)

            name = str(name_value or "").lower()
            if not name:
                continue
            call_id = str(call_id or "")

            if call_id:
                key = (name, call_id)
            else:
                try:
                    serialized_args = json.dumps(
                        args or {}, sort_keys=True, default=str
                    )
                except (TypeError, ValueError):
                    serialized_args = repr(args)
                key = (name, serialized_args)

            if key in seen_keys:
                continue
            seen_keys.add(key)
            tool_call_names.append(name)

    return tool_call_names

def format_messages_string(messages: List[Any]) -> str:
    """
    Format a sequence of message-like objects into a normalized, human-readable conversation string.
    
    Parses each item in `messages` to produce lines representing user, assistant, tool calls, and tool results. Tool call names are normalized (e.g., "send_email_tool" -> "write_email", "check_calendar_tool" -> "check_calendar_availability", "schedule_meeting_tool" -> "schedule_meeting"); write_email-style calls have their arguments reduced to a concise `to`, `subject`, and `content` view when possible. The function extracts brief context (Subject/From) from earlier markdown-formatted messages to populate missing email fields, truncates very long tool results for readability, and falls back gracefully for unknown or malformed message shapes.
    
    Parameters:
        messages (List[Any]): Iterable of message-like objects or dicts. Supported shapes include dicts with keys like `role`, `content`, `tool_calls`, and tool result dicts with `tool_call_id`; or objects with attributes `type`/`role`, `content`, and `tool_calls`.
    
    Returns:
        str: A newline-joined string where each line is one of:
            - "user: {text}" for user messages
            - "assistant: {text}" for assistant messages
            - "assistant: tool_call -> {name} {args_json}" for tool calls
            - "tool[{tool_call_id}]: {result}" for tool results
    """

    def normalize_tool_name(name: str) -> str:
        mapping = {
            "send_email_tool": "write_email",
            "check_calendar_tool": "check_calendar_availability",
            "schedule_meeting_tool": "schedule_meeting",
        }
        return mapping.get(name, name)

    lines: List[str] = []
    # Try to capture context (subject/from) from earlier messages for normalization
    context_subject = ""
    context_from = ""
    for m in messages:
        try:
            text = extract_message_content(m)
        except Exception:
            text = ""
        if not isinstance(text, str) or not text:
            continue
        if "**Subject**:" in text and "**From**:" in text:
            # naive extraction from markdown block produced by format_gmail_markdown
            try:
                # Find lines
                for line in text.splitlines():
                    if line.strip().startswith("**Subject**:") and not context_subject:
                        context_subject = line.split(":", 1)[1].strip()
                    if line.strip().startswith("**From**:") and not context_from:
                        context_from = line.split(":", 1)[1].strip()
            except Exception:
                pass

    for m in messages:
        # Tool call messages (AI)
        tool_calls = None
        if isinstance(m, dict):
            tool_calls = m.get("tool_calls")
            role = m.get("role") or "assistant"
            content = m.get("content", "")
        else:
            tool_calls = getattr(m, "tool_calls", None)
            role = getattr(m, "type", None) or getattr(m, "role", None) or "assistant"
            content = extract_message_content(m)

        if tool_calls:
            for tc in tool_calls:
                name_raw = tc.get("name", "")
                name = normalize_tool_name(name_raw)
                args = tc.get("args", {})
                # Normalize args for write_email-style display
                norm_args = args
                if name == "write_email":
                    content = args.get("content") or args.get("response_text")
                    subject = args.get("subject") or (f"Re: {context_subject}" if context_subject else None)
                    to_addr = args.get("to") or context_from or None
                    norm_args = {k: v for k, v in {
                        "to": to_addr,
                        "subject": subject,
                        "content": content,
                    }.items() if v}
                try:
                    args_slim = json.dumps(norm_args, ensure_ascii=False)
                except Exception:
                    args_slim = str(norm_args)
                lines.append(f"assistant: tool_call -> {name} {args_slim}")
            continue

        # Tool result messages
        if isinstance(m, dict) and role == "tool":
            tool_call_id = m.get("tool_call_id", "")
            tool_result = m.get("content", "")
            tool_result = str(tool_result)
            # Trim overly long tool results for readability
            if len(tool_result) > 1000:
                tool_result = tool_result[:1000] + "…"
            lines.append(f"tool[{tool_call_id}]: {tool_result}")
            continue

        # Regular assistant/user text
        if role in ("ai", "assistant"):
            text = content or ""
            lines.append(f"assistant: {text}")
        elif role in ("human", "user"):
            text = content or ""
            lines.append(f"user: {text}")
        else:
            # Fallback to pretty repr if available
            try:
                lines.append(m.pretty_repr())  # type: ignore[attr-defined]
            except Exception:
                lines.append(str(m))

    return "\n".join(lines)

def show_graph(graph, xray=False):
    """Display a LangGraph mermaid diagram with fallback rendering.
    
    Handles timeout errors from mermaid.ink by falling back to pyppeteer.
    
    Args:
        graph: The LangGraph object that has a get_graph() method
    """
    from IPython.display import Image
    try:
        # Try the default renderer first
        return Image(graph.get_graph(xray=xray).draw_mermaid_png())
    except Exception as e:
        # Fall back to pyppeteer if the default renderer fails
        import nest_asyncio
        nest_asyncio.apply()
        from langchain_core.runnables.graph import MermaidDrawMethod
        return Image(graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER))

# --- Override parse_gmail with robust fallbacks (Gmail + dataset) ---
def parse_gmail(email_input: dict) -> tuple[str, str, str, str, str]:
    """Extract key fields from Gmail-like dicts with robust fallbacks.

    Supports:
      - Raw Gmail API payloads (payload.headers/snippet, id)
      - Simplified Gmail-like schema (from/to/subject/body, id/thread_id)
      - Dataset schema used in tests (author/to/subject/email_thread)
    """
    if not email_input or not isinstance(email_input, dict):
        return "", "", "", "", ""

    email_id = (
        email_input.get("id")
        or email_input.get("thread_id")
        or email_input.get("message_id")
        or email_input.get("gmail_id")
        or ""
    )

    # Real Gmail API shape
    if "payload" in email_input:
        headers = email_input.get("payload", {}).get("headers", [])
        author = next((h.get("value", "") for h in headers if h.get("name") == "From"), "")
        to = next((h.get("value", "") for h in headers if h.get("name") == "To"), "")
        subject = next((h.get("value", "") for h in headers if h.get("name") == "Subject"), "")
        email_thread = email_input.get("snippet", "")
        return author, to, subject, email_thread, email_id

    # Simplified Gmail-like OR dataset schema
    author = (
        email_input.get("from")
        or email_input.get("From")
        or email_input.get("author")
        or ""
    )
    to = email_input.get("to") or email_input.get("To") or ""
    subject = email_input.get("subject") or email_input.get("Subject") or ""
    email_thread = (
        email_input.get("body")
        or email_input.get("Body")
        or email_input.get("email_thread")
        or email_input.get("page_content")
        or ""
    )

    return author, to, subject, email_thread, email_id
