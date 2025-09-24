from typing import List, Any
import json
import re

try:
    import html2text  # type: ignore
except ImportError:
    html2text = None

def format_email_markdown(subject, author, to, email_thread, email_id=None):
    """Format email details into a nicely formatted markdown string for display
    
    Args:
        subject: Email subject
        author: Email sender
        to: Email recipient
        email_thread: Email content
        email_id: Optional email ID (for Gmail API)
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
    """Format Gmail email details into a nicely formatted markdown string for display,
    with HTML to text conversion for HTML content
    
    Args:
        subject: Email subject
        author: Email sender
        to: Email recipient
        email_thread: Email content (possibly HTML)
        email_id: Optional email ID (for Gmail API)
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

def format_for_display(tool_call):
    """Format content for display in Agent Inbox
    
    Args:
        tool_call: The tool call to format
    """
    # Initialize empty display
    display = ""
    
    # Add tool call information
    if tool_call["name"] == "write_email" or tool_call["name"] == "send_email_tool":
        display += f"""# Email Draft

**To**: {tool_call["args"].get("to") or ''}
**Subject**: {tool_call["args"].get("subject") or ''}

{tool_call["args"].get("content") or tool_call["args"].get("response_text")}
"""
    elif tool_call["name"] == "schedule_meeting":
        display += f"""# Calendar Invite

**Meeting**: {tool_call["args"].get("subject")}
**Attendees**: {', '.join(tool_call["args"].get("attendees"))}
**Duration**: {tool_call["args"].get("duration_minutes")} minutes
**Day**: {tool_call["args"].get("preferred_day")}
"""
    elif tool_call["name"] == "Question":
        # Special formatting for questions to make them clear
        display += f"""# Question for User

{tool_call["args"].get("content")}
"""
    else:
        # Generic format for other tools
        display += f"""# Tool Call: {tool_call["name"]}

Arguments:"""
        
        # Check if args is a dictionary or string
        if isinstance(tool_call["args"], dict):
            display += f"\n{json.dumps(tool_call['args'], indent=2)}\n"
        else:
            display += f"\n{tool_call['args']}\n"
    return display

def parse_email(email_input: dict) -> tuple[str, str, str, str]:
    """Parse an email input dictionary, accepting multiple common schemas.

    Supports both the repo dataset schema and a Gmail-like schema:
      - Dataset schema: author, to, subject, email_thread
      - Gmail-like schema: from, to, subject, body

    Returns (author, to, subject, email_thread). Missing fields are returned as empty strings
    to keep downstream prompts resilient.
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
    """Format examples into a readable string representation.

    Args:
        examples (List[Item]): List of example items from the vector store, where each item
            contains a value string with the format:
            'Email: {...} Original routing: {...} Correct routing: {...}'

    Returns:
        str: A formatted string containing all examples, with each example formatted as:
            Example:
            Email: {email_details}
            Original Classification: {original_routing}
            Correct Classification: {correct_routing}
            ---
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
    """Extract tool call names from messages, safely handling messages without tool_calls."""
    tool_call_names = []
    for message in messages:
        # Check if message is a dict and has tool_calls
        if isinstance(message, dict) and message.get("tool_calls"):
            tool_call_names.extend([call["name"].lower() for call in message["tool_calls"]])
        # Check if message is an object with tool_calls attribute
        elif hasattr(message, "tool_calls") and message.tool_calls:
            tool_call_names.extend([call["name"].lower() for call in message.tool_calls])
    
    return tool_call_names

def format_messages_string(messages: List[Any]) -> str:
    """Format a conversation to a readable string, normalizing tool names.

    - Maps Gmail tool names to canonical dataset names for evaluation:
      send_email_tool -> write_email, check_calendar_tool -> check_calendar_availability,
      schedule_meeting_tool -> schedule_meeting. Done/Question unchanged.
    - Includes tool arguments succinctly and tool results content.
    - Falls back gracefully for unknown message shapes.
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
    except Exception:
        # Fall back to pyppeteer if the default renderer fails
        import nest_asyncio
        nest_asyncio.apply()
        from langchain_core.runnables.graph import MermaidDrawMethod
        return Image(graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER))

 
