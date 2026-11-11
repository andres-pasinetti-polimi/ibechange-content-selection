from datetime import datetime, timezone
from dateutil import parser as dtp


def parse_client_ts(ts: str) -> datetime:
    """
    Parse any ISO-8601 timestamp.
    * Reject naïve strings (no offset).
    * Convert everything to UTC.
    """
    dt = dtp.isoparse(ts)  # handles Z, +03:00, -05:30, etc.
    if dt.tzinfo is None:
        raise ValueError("Timezone offset missing")
    return dt.astimezone(timezone.utc)


def utc_iso(dt: datetime) -> str:
    """
    Return an RFC 3339 string in UTC with trailing 'Z' and seconds precision.
    Will never lie: it first converts to UTC.
    """
    if dt.tzinfo is None:
        raise ValueError("Naïve datetime received – timezone info required")
    # Convert then format
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
