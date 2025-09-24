from datetime import datetime
import os
import re

from email_assistant.tools.default.calendar_tools import _coerce_preferred_day


def _is_tz_sydney(dt: datetime) -> bool:
    abbr = dt.tzname() or ""
    # Sydney switches AEDT/AEST; just assert it's not UTC and looks like AU
    return abbr in {"AEST", "AEDT"}


def test_timezone_default_sydney(monkeypatch):
    monkeypatch.setenv("TIMEZONE", "Australia/Sydney")
    dt = _coerce_preferred_day("2025-05-19 14:00")
    assert _is_tz_sydney(dt)


def test_time_only_today_local(monkeypatch):
    monkeypatch.setenv("TIMEZONE", "Australia/Sydney")
    dt = _coerce_preferred_day("14:00")
    assert dt.hour == 14 and dt.minute == 0
    assert _is_tz_sydney(dt)


def test_weekday_parsing_next_occurrence(monkeypatch):
    monkeypatch.setenv("TIMEZONE", "Australia/Sydney")
    dt = _coerce_preferred_day("next Tuesday 14:00")
    assert dt.weekday() == 1  # Tuesday
    assert dt.hour == 14
    assert _is_tz_sydney(dt)

