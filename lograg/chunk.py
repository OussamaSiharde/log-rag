"""Group log records into retrieval chunks.

Retrieval quality depends far more on chunking than on model size, so the
rules here are deliberate:

* A chunk never mixes services: "payments timed out" next to an unrelated
  gateway line produces a blurry embedding that matches neither question.
* A chunk covers a bounded time window, so its timestamps mean something
  when the answer has to say *when* something started.
* An error and its stack trace are never split across chunks.
* Consecutive repeats of the same message collapse into one line with a
  count, so a log storm does not crowd everything else out of the chunk.
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

from .parse import LogRecord

SEVERE = {"ERROR", "FATAL"}


@dataclass
class Chunk:
    source: str
    service: str
    start_ts: datetime
    end_ts: datetime
    levels: list[str]
    error_count: int
    content: str

    @property
    def content_hash(self) -> str:
        key = f"{self.source}|{self.service}|{self.start_ts.isoformat()}|{self.content}"
        return hashlib.sha256(key.encode()).hexdigest()

    def embedding_text(self) -> str:
        """Text sent to the embedding model: a metadata header, then the log lines."""
        header = f"service: {self.service} | levels: {','.join(self.levels)} | {self.start_ts:%Y-%m-%d %H:%M}"
        return f"{header}\n{self.content}"


def _render(records: list[LogRecord]) -> str:
    lines: list[str] = []
    prev_key, repeats = None, 0
    for r in records:
        key = (r.level, r.text)
        if key == prev_key:
            repeats += 1
            continue
        if repeats:
            lines[-1] += f"  (repeated x{repeats + 1})"
        prev_key, repeats = key, 0
        lines.append(f"{r.ts:%H:%M:%S} {r.level} {r.text}")
    if repeats:
        lines[-1] += f"  (repeated x{repeats + 1})"
    return "\n".join(lines)


def _flush(window: list[LogRecord]) -> Chunk:
    levels = sorted({r.level for r in window}, key=lambda lv: ("TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL").index(lv))
    return Chunk(
        source=window[0].source,
        service=window[0].service,
        start_ts=window[0].ts,
        end_ts=window[-1].ts,
        levels=levels,
        error_count=sum(r.level in SEVERE for r in window),
        content=_render(window),
    )


def chunk_records(
    records: Iterable[LogRecord],
    max_records: int = 25,
    max_span: timedelta = timedelta(minutes=5),
    max_chars: int = 1800,
) -> list[Chunk]:
    """Split records into per-service, time-bounded chunks (records must be in file order)."""
    open_windows: dict[str, list[LogRecord]] = {}
    sizes: dict[str, int] = {}
    chunks: list[Chunk] = []

    for r in records:
        window = open_windows.setdefault(r.service, [])
        size = sizes.get(r.service, 0)
        record_len = len(r.text) + 20
        too_long = window and (r.ts - window[0].ts > max_span)
        too_big = window and (len(window) >= max_records or size + record_len > max_chars)
        if too_long or too_big:
            chunks.append(_flush(window))
            window = open_windows[r.service] = []
            size = 0
        window.append(r)
        sizes[r.service] = size + record_len

    chunks.extend(_flush(w) for w in open_windows.values() if w)
    chunks.sort(key=lambda c: (c.start_ts, c.service))
    return chunks
