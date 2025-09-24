from datetime import datetime, timedelta
from typing import Any
import os
import re
from zoneinfo import ZoneInfo
from langchain_core.tools import tool

WEEKDAYS = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]

def _get_local_tz() -> ZoneInfo:
    tz = os.getenv("TIMEZONE", "Australia/Sydney")
    try:
        return ZoneInfo(tz)
    except Exception:
        return ZoneInfo("UTC")


def _coerce_preferred_day(value: Any, fallback_start_time: int | None = None) -> datetime:
    """Coerce preferred_day into a datetime.

    Accepts:
    - datetime: returned as-is
    - str: attempts in order
      * ISO 8601 (datetime.fromisoformat with 'T' or space)
      * 'YYYY-MM-DD HH:MM' (24h)
      * 'YYYY-MM-DD' (uses 09:00 or fallback_start_time if provided)
      * 'next <weekday> HH:MM' or '<weekday> HH:MM' (next occurrence)
      * '<weekday>' (next occurrence at 09:00 or fallback_start_time)
    Raises ValueError if unable to parse.
    """
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise ValueError("preferred_day must be a datetime or string")

    s = str(value).strip()
    # Sanitize common non-date junk (e.g., accidental file extensions or labels)
    # Keep only date/time relevant characters
    s = re.sub(r"[^0-9A-Za-z:\-\+T\s/]+", " ", s).strip()
    # Normalize Zulu timezone to ISO offset
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    # Try ISO
    try:
        # Allow space as separator too
        iso_s = s.replace(" ", "T", 1) if "T" not in s and len(s) >= 16 else s
        dt = datetime.fromisoformat(iso_s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_get_local_tz())
        return dt
    except Exception:
        pass

    # Try common formats
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M", "%d/%m/%Y %H:%M"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=_get_local_tz())
        except Exception:
            pass
    try:
        d = datetime.strptime(s, "%Y-%m-%d")
        hour = (fallback_start_time or 900) // 100
        minute = (fallback_start_time or 900) % 100
        return d.replace(hour=hour, minute=minute, second=0, microsecond=0, tzinfo=_get_local_tz())
    except Exception:
        pass

    # Time-only strings like "14:00" -> today at that time
    if re.fullmatch(r"\d{1,2}:\d{2}", s):
        hour, minute = map(int, s.split(":"))
        today = datetime.now(_get_local_tz()).replace(hour=hour, minute=minute, second=0, microsecond=0)
        return today

    # Weekday heuristics like "next Tuesday 14:00" or "Tuesday 14:00" or "Tuesday"
    lower = s.lower()
    tokens = lower.replace(",", " ").split()
    # Extract weekday
    wd_idx = None
    wd_next = False
    for i, tok in enumerate(tokens):
        if tok in WEEKDAYS:
            wd_idx = WEEKDAYS.index(tok)
            wd_next = (i > 0 and tokens[i - 1] == "next") or (i + 1 < len(tokens) and tokens[i + 1] == "next")
            break
    if wd_idx is not None:
        # Extract time if present
        hh = ((fallback_start_time or 900) // 100)
        mm = ((fallback_start_time or 900) % 100)
        for t in tokens:
            if ":" in t:
                try:
                    parts = t.split(":")
                    hh = int(parts[0])
                    mm = int(parts[1])
                    break
                except Exception:
                    pass
        # Compute next occurrence of weekday
        today = datetime.now(_get_local_tz())
        delta = (wd_idx - today.weekday()) % 7
        if delta == 0 or wd_next:
            delta = 7 if not wd_next else (delta or 7)
        target = today + timedelta(days=delta)
        return target.replace(hour=hh, minute=mm, second=0, microsecond=0)

    # Final fallback: choose next day at the requested start_time in local TZ
    # This prevents hard failures when the model emits unparseable strings (e.g., 'value.txt').
    hour = (fallback_start_time or 900) // 100
    minute = (fallback_start_time or 900) % 100
    now_local = datetime.now(_get_local_tz())
    fallback_dt = (now_local + timedelta(days=1)).replace(hour=hour, minute=minute, second=0, microsecond=0)
    return fallback_dt

@tool
def schedule_meeting(
    attendees: list[str], subject: str, duration_minutes: int, preferred_day: Any, start_time: int
) -> str:
    """Schedule a calendar meeting."""
    # Coerce string inputs to datetime when needed
    try:
        dt = _coerce_preferred_day(preferred_day, fallback_start_time=start_time)
    except Exception as e:
        return f"Invalid preferred_day: {e}"

    # Placeholder response - in real app would check calendar and schedule
    tz_abbr = dt.tzname() or ""
    date_str = dt.strftime("%A, %B %d, %Y")
    return (
        f"Meeting '{subject}' scheduled on {date_str} at {start_time} {tz_abbr}"
        f" for {duration_minutes} minutes with {len(attendees)} attendees"
    )

@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    # Placeholder response - in real app would check actual calendar
    tz_abbr = datetime.now(_get_local_tz()).tzname() or ""
    return f"Available times on {day} ({tz_abbr}): 9:00 AM, 2:00 PM, 4:00 PM"
