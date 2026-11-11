# cs_module/services/time_handler.py
from __future__ import annotations

import copy
from datetime import datetime, timezone
from dateutil import parser as dtp  # pip install python-dateutil
import logging
import traceback

logger = logging.getLogger(__name__)


class TimeHandler:
    """
    Two modes:
      - REAL   : time flows, `now` = system UTC clock
      - FROZEN : time is pinned to an internal cursor set via set_current_time()/set_start_time()

    Endpoints:
      - /set_time_mode {"mode": "REAL"|"FROZEN"}
      - /set_current_time "2025-09-02T08:00:00Z"  (effective only in FROZEN)
      - /set_start_time  "..."                    (effective only in FROZEN)
    """

    REAL = "REAL"
    FROZEN = "FROZEN"

    def __init__(self, current_time: datetime | None = None, mode: str = REAL):
        # Always store aware UTC
        now_utc = datetime.now(timezone.utc)
        self._mode = mode.upper() if mode else self.REAL

        # In REAL mode, cursor is irrelevant; keep it for introspection.
        # In FROZEN mode, this is the "now" cursor.
        self.start_time = None
        self.current_time = current_time.astimezone(timezone.utc) if current_time else now_utc

    # ------------------------ public API ------------------------

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        """
        Switch between REAL and FROZEN at runtime.
        - When switching to FROZEN, keep the current wall-clock as the cursor (so it starts "from now"),
          unless you later overwrite it with /set_current_time.
        - When switching to REAL, we stop using the internal cursor.
        """
        if not mode:
            raise ValueError("mode must be 'REAL' or 'FROZEN'")
        mode = mode.upper()
        if mode not in (self.REAL, self.FROZEN):
            raise ValueError("mode must be 'REAL' or 'FROZEN'")

        if mode == self._mode:
            # logger.info("TimeHandler already in mode %s", mode)
            return

        if mode == self.FROZEN:
            # freeze to the present wall-clock if no explicit cursor set yet
            self.current_time = self._to_utc(self.current_time) or datetime.now(timezone.utc)
            logger.info("Switching TimeHandler to FROZEN; cursor=%s", self.current_time.isoformat())
        else:
            logger.info("Switching TimeHandler to REAL (wall-clock UTC)")

        self._mode = mode

    @property
    def now(self) -> datetime:
        """Return current time in UTC according to mode."""
        if self._mode == self.REAL:
            return datetime.now(timezone.utc)
        return self.current_time

    def set(self, new_time: datetime) -> None:
        """Set the cursor (only meaningful in FROZEN)."""
        if self._mode == self.REAL:
            logger.warning("TimeHandler.set(...) called in REAL mode; ignoring.")
            return
        self.current_time = self._to_utc(self._ensure_tz(new_time))

    def set_start_time(self, start_time: datetime) -> None:
        """Set the start time and (if FROZEN) also move the cursor there."""
        if self._mode == self.REAL:
            logger.info("set_start_time in REAL mode: recording start but not pinning clock.")
            self.start_time = self._to_utc(self._ensure_tz(start_time))
            return
        start_time_utc = self._to_utc(self._ensure_tz(start_time))
        self.start_time = start_time_utc
        self.current_time = copy.deepcopy(start_time_utc)

    # -------------------- parsing / formatting -------------------

    @staticmethod
    def parse_client_ts(ts: str, mute_naive_warning=False) -> datetime:
        """
        Parse ISO-8601 string.
        * If naïve (no tz), log and assume UTC.
        * Return aware UTC datetime.
        """
        if ts is None:
            raise ValueError("timestamp is None")
        if isinstance(ts, datetime):
            dt = ts
        else:
            dt = dtp.isoparse(ts)
        if dt.tzinfo is None and not mute_naive_warning:
            logger.warning(
                "[Naïve Timestamp] Received timestamp without timezone: '%s' — assuming UTC.\nCaller Trace:\n%s",
                ts,
                "".join(traceback.format_stack(limit=2)),
            )
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def utc_iso(dt: datetime) -> str:
        """Canonical ‘YYYY-MM-DDTHH:MM:SSZ’ string."""
        if dt.tzinfo is None:
            raise ValueError("Naïve datetime received")
        return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    # ---------------------- internal utils ----------------------

    @staticmethod
    def _ensure_tz(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            raise ValueError("Naïve datetime supplied; timezone required")
        return dt

    @staticmethod
    def _to_utc(dt: datetime | None) -> datetime | None:
        if dt is None:
            return None
        return dt.astimezone(timezone.utc)
