"""Turn raw log lines into structured records.

Two line formats are recognised:

  ISO style      2026-10-20T09:00:00.000Z ERROR [payments] message
  Spring Boot    2026-10-20 09:00:00.000  ERROR 4242 --- [orders] [main] c.e.Foo : message

Any line that does not start a new record (stack-trace frames, "Caused by:",
wrapped messages) is appended to the previous record, so an exception and its
trace always travel together.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

LEVELS = ("TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL")
_LEVEL_ALIASES = {"WARNING": "WARN", "CRITICAL": "FATAL", "SEVERE": "ERROR"}

_ISO = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s+"
    r"(?P<level>[A-Z]+)\s+\[(?P<service>[\w.\-]+)\]\s?(?P<msg>.*)$"
)
_SPRING = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+"
    r"(?P<level>[A-Z]+)\s+\d+\s+---\s+\[(?P<service>[\w.\-]+)\]\s+\[[^\]]*\]\s+(?P<logger>\S+)\s*:\s?(?P<msg>.*)$"
)


@dataclass
class LogRecord:
    ts: datetime
    level: str
    service: str
    message: str
    source: str
    line_no: int
    extra: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Message plus any continuation lines (stack trace)."""
        return "\n".join([self.message, *self.extra]) if self.extra else self.message


def _parse_ts(raw: str) -> datetime:
    raw = raw.replace("Z", "+00:00").replace(" ", "T", 1)
    ts = datetime.fromisoformat(raw)
    # Logs without an offset are treated as UTC so every timestamp is comparable.
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _normalise_level(level: str) -> str:
    level = _LEVEL_ALIASES.get(level, level)
    return level if level in LEVELS else "INFO"


def parse_line(line: str, source: str = "", line_no: int = 0) -> LogRecord | None:
    """Parse one line that starts a record; returns None for continuation lines."""
    for pattern in (_ISO, _SPRING):
        m = pattern.match(line)
        if m:
            return LogRecord(
                ts=_parse_ts(m["ts"]),
                level=_normalise_level(m["level"]),
                service=m["service"],
                message=m["msg"].rstrip(),
                source=source,
                line_no=line_no,
            )
    return None


def parse_lines(lines: Iterable[str], source: str = "") -> Iterator[LogRecord]:
    current: LogRecord | None = None
    for line_no, raw in enumerate(lines, start=1):
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        record = parse_line(line, source, line_no)
        if record:
            if current:
                yield current
            current = record
        elif current:
            current.extra.append(line.rstrip())
        # Lines before the first recognisable record have no owner and are dropped.
    if current:
        yield current


def parse_file(path: Path) -> list[LogRecord]:
    with path.open(encoding="utf-8", errors="replace") as fh:
        return list(parse_lines(fh, source=path.name))
