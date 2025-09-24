from __future__ import annotations

from datetime import datetime, timedelta


def build_gmail_query(
    *,
    email_address: str,
    minutes_since: int = 30,
    include_read: bool = False,
) -> str:
    """Construct a Gmail search query used by tools and ingestion.

    - Matches messages to/from the address
    - Applies an "after" unix timestamp window
    - Excludes read messages unless include_read=True
    """
    after = int((datetime.now() - timedelta(minutes=max(0, int(minutes_since)))).timestamp())
    query = f"(to:{email_address} OR from:{email_address}) after:{after}"
    if not include_read:
        query += " is:unread"
    return query


__all__ = ["build_gmail_query"]

