import base64
from typing import Any, Optional


def extract_message_part(payload: Optional[dict[str, Any]] | Any) -> str:
    """Extract text content from a Gmail API message payload.

    Prefers text/plain parts, falls back to text/html, and finally any nested
    parts. Decodes urlsafe base64 bodies.
    """
    if not isinstance(payload, dict):
        return ""

    # Multiparts: prefer text/plain → text/html → recurse
    parts = payload.get("parts") or []
    if parts:
        # Text/plain first
        for part in parts:
            try:
                if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
                    data = part["body"]["data"]
                    return base64.urlsafe_b64decode(data).decode("utf-8")
            except Exception:
                continue
        # Then text/html
        for part in parts:
            try:
                if part.get("mimeType") == "text/html" and part.get("body", {}).get("data"):
                    data = part["body"]["data"]
                    return base64.urlsafe_b64decode(data).decode("utf-8")
            except Exception:
                continue
        # Finally recurse
        for part in parts:
            content = extract_message_part(part)
            if content:
                return content

    # Single-part body
    try:
        data = payload.get("body", {}).get("data")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8")
    except Exception:
        pass

    return ""


__all__ = ["extract_message_part"]

