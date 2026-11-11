"""
Log triage tool. Reads docker-compose log output (stdin or file), extracts
warnings/errors/tracebacks, deduplicates, and writes a compact markdown report.

Usage:
    docker compose -f cs_module/docker-compose.yml logs --no-color --tail=100000 cs_module \
        | python tools/triage_logs.py

    # or from a saved file:
    python tools/triage_logs.py logs/run.log

    # choose output file (default: logs/triage_<timestamp>.md):
    python tools/triage_logs.py logs/run.log -o logs/report.md

    # include WARNING lines (default: errors+tracebacks only):
    python tools/triage_logs.py --warnings
"""

import re
import sys
import os
import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from argparse import ArgumentParser

# ── patterns ───────────────────────────────────────────────────────────────────

# matches: "cs_module-1  | 2026-05-15 14:12:39,706 [LEVEL] message"
LOG_LINE = re.compile(
    r"^(?P<service>[^\s|]+)\s*\|\s*"
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[,\.]\d+)?\s*"
    r"\[(?P<level>[A-Z]+)\]\s*"
    r"(?P<msg>.*)$"
)

# bare continuation lines (tracebacks, Flask output, etc.)
BARE_LINE = re.compile(r"^(?P<service>[^\s|]+)\s*\|\s*(?P<msg>.*)$")

TRACEBACK_START = re.compile(r"Traceback \(most recent call last\)")
TRACEBACK_BODY  = re.compile(r"^\s+|^  File |^\w.*Error:|^\w.*Exception:|^\w.*Warning:")

# variable parts to strip when fingerprinting errors
_NORM = [
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"), "<UUID>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[^\s,]*"), "<TS>"),
    (re.compile(r"\b\d+\.\d+\.\d+\.\d+\b"), "<IP>"),
    (re.compile(r"\bline \d+\b"), "line <N>"),
    (re.compile(r"\b\d{5,}\b"), "<NUM>"),
    (re.compile(r"'[^']{36,}'"), "'<STR>'"),   # long quoted strings
]


def normalize(text):
    for pattern, replacement in _NORM:
        text = pattern.sub(replacement, text)
    return text.strip()


def fingerprint(lines):
    key = normalize(" ".join(lines[-3:]))  # last 3 lines capture the exception type
    return hashlib.md5(key.encode()).hexdigest()[:10]


# ── parsing ────────────────────────────────────────────────────────────────────

class Event:
    __slots__ = ("level", "service", "ts", "lines", "count", "first_ts", "last_ts", "fp")

    def __init__(self, level, service, ts, lines):
        self.level    = level
        self.service  = service
        self.ts       = ts
        self.lines    = lines
        self.count    = 1
        self.first_ts = ts
        self.last_ts  = ts
        self.fp       = fingerprint(lines)


def parse_stream(stream, include_warnings):
    events       = []             # list of Event (ordered by first occurrence)
    fp_map       = {}             # fingerprint → index in events
    service_info = defaultdict(int)   # service → INFO count

    buf       = []   # current error/traceback block being accumulated
    buf_level = None
    buf_svc   = None
    buf_ts    = None
    in_tb     = False  # inside a traceback block

    def flush():
        nonlocal buf, buf_level, buf_svc, buf_ts, in_tb
        if not buf:
            return
        ev = Event(buf_level, buf_svc, buf_ts, list(buf))
        fp = ev.fp
        if fp in fp_map:
            existing = events[fp_map[fp]]
            existing.count  += 1
            existing.last_ts = buf_ts
        else:
            fp_map[fp] = len(events)
            events.append(ev)
        buf = []
        buf_level = None
        buf_svc   = None
        buf_ts    = None
        in_tb     = False

    for raw in stream:
        raw = raw.rstrip("\n")

        m = LOG_LINE.match(raw)
        if m:
            level = m.group("level")
            svc   = m.group("service")
            ts    = m.group("ts") or ""
            msg   = m.group("msg")

            if level == "INFO":
                service_info[svc] += 1
                flush()
                continue

            if level in ("WARNING", "WARN"):
                flush()
                if include_warnings:
                    buf       = [msg]
                    buf_level = "WARNING"
                    buf_svc   = svc
                    buf_ts    = ts
                    flush()
                continue

            if level in ("ERROR", "CRITICAL"):
                flush()
                buf       = [msg]
                buf_level = level
                buf_svc   = svc
                buf_ts    = ts
                in_tb     = False
                continue

            flush()
            continue

        # bare line (no level tag) — could be part of traceback
        bm = BARE_LINE.match(raw)
        if bm:
            msg = bm.group("msg")

            if TRACEBACK_START.search(msg):
                if not buf:
                    buf       = []
                    buf_level = "ERROR"
                    buf_svc   = bm.group("service")
                    buf_ts    = ""
                buf.append(msg)
                in_tb = True
                continue

            if in_tb:
                buf.append(msg)
                # end of traceback: line that IS the exception class
                if re.match(r"^\w+(\.\w+)*(Error|Exception|Warning):", msg):
                    in_tb = False
                    # don't flush yet — might be followed by more context
                continue

            # if we're building a block, append context
            if buf:
                buf.append(msg)

        else:
            if buf:
                buf.append(raw)

    flush()
    return events, service_info


# ── report ─────────────────────────────────────────────────────────────────────

def render_report(events, service_info, include_warnings, source):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Log Triage Report",
        f"**Source:** `{source}`  |  **Generated:** {now}",
        "",
    ]

    # summary counts
    errors   = [e for e in events if e.level in ("ERROR", "CRITICAL")]
    warnings = [e for e in events if e.level == "WARNING"]
    total_e  = sum(e.count for e in errors)
    total_w  = sum(e.count for e in warnings)
    total_i  = sum(service_info.values())

    lines += [
        "## Summary",
        f"| Level | Unique patterns | Total occurrences |",
        f"|-------|----------------|-------------------|",
        f"| ERROR / CRITICAL | {len(errors)} | {total_e} |",
    ]
    if include_warnings:
        lines.append(f"| WARNING | {len(warnings)} | {total_w} |")
    lines += [
        f"| INFO (not shown) | — | {total_i} |",
        "",
        "**Services seen:** " + ", ".join(f"`{s}`" for s in sorted(service_info)),
        "",
    ]

    def render_events(evs, label):
        if not evs:
            lines.append(f"*No {label} events.*\n")
            return
        for ev in sorted(evs, key=lambda e: -e.count):
            ts_info = ev.first_ts
            if ev.count > 1:
                ts_info = f"first: {ev.first_ts or '?'}  last: {ev.last_ts or '?'}"
            lines.append(f"### [{ev.level}] × {ev.count}  —  `{ev.service}`  ·  {ts_info}")
            lines.append("```")
            lines.extend(ev.lines[:30])   # cap at 30 lines per event
            if len(ev.lines) > 30:
                lines.append(f"... ({len(ev.lines) - 30} more lines)")
            lines.append("```")
            lines.append("")

    lines.append("## Errors & Critical")
    render_events(errors, "ERROR")

    if include_warnings:
        lines.append("## Warnings")
        render_events(warnings, "WARNING")

    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = ArgumentParser(description="Triage docker-compose logs into a markdown report.")
    ap.add_argument("logfile", nargs="?", help="Path to log file (default: stdin)")
    ap.add_argument("-o", "--output", help="Output markdown file (default: logs/triage_<ts>.md)")
    ap.add_argument("--warnings", action="store_true",
                    help="Include WARNING lines (default: errors only)")
    args = ap.parse_args()

    source = args.logfile or "stdin"

    if args.logfile:
        with open(args.logfile, "r", errors="replace") as f:
            events, service_info = parse_stream(f, args.warnings)
    else:
        events, service_info = parse_stream(sys.stdin, args.warnings)

    report = render_report(events, service_info, args.warnings, source)

    if args.output:
        out_path = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("logs", exist_ok=True)
        out_path = f"logs/triage_{ts}.md"

    with open(out_path, "w") as f:
        f.write(report)

    errors   = [e for e in events if e.level in ("ERROR", "CRITICAL")]
    warnings = [e for e in events if e.level == "WARNING"]
    total_e  = sum(e.count for e in errors)
    total_w  = sum(e.count for e in warnings)
    print(
        f"→ {len(errors)} unique errors ({total_e} occurrences)"
        f"  |  {len(warnings)} unique warnings ({total_w} occurrences)"
        f"  →  {out_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
