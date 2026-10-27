"""Ingestion and question answering on top of the store."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable

from .chunk import chunk_records
from .ollama import ChatModel, Embedder
from .parse import parse_file
from .store import Hit, Store

SYSTEM_PROMPT = """You are an SRE assistant answering questions about application logs.
Rules:
- Use ONLY the numbered log excerpts provided. If they do not contain the answer, say so plainly.
- Cite every factual claim with its excerpt number, like [2].
- Give timestamps exactly as they appear in the excerpts, including the date.
- When asked when a problem started, give the earliest matching timestamp you can see.
- Be concise: a short answer first, then at most a few supporting bullet points."""


@dataclass
class IngestReport:
    files: int
    records: int
    chunks: int
    inserted: int


@dataclass
class Answer:
    text: str
    sources: list[Hit]


def ingest(paths: Iterable[Path], store: Store, embedder: Embedder) -> IngestReport:
    files = records = chunks_total = inserted = 0
    for path in paths:
        recs = parse_file(path)
        chunks = chunk_records(recs)
        files, records, chunks_total = files + 1, records + len(recs), chunks_total + len(chunks)
        # Skip chunks already stored so re-running ingest does not re-embed anything.
        known = store.existing_hashes([c.content_hash for c in chunks])
        fresh = [c for c in chunks if c.content_hash not in known]
        if fresh:
            vectors = embedder.embed_documents([c.embedding_text() for c in fresh])
            inserted += store.insert(fresh, vectors)
    return IngestReport(files, records, chunks_total, inserted)


def retrieve(
    question: str,
    store: Store,
    embedder: Embedder,
    k: int = 6,
    context: int = 3,
    lookback: timedelta = timedelta(minutes=30),
    lookahead: timedelta = timedelta(minutes=5),
    service: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    min_level: str | None = None,
    onset_margin: float = 0.05,
) -> list[Hit]:
    """Vector search, then explain *when*: find where the matching pattern started and
    add chunks from other services around that moment (deploys first)."""
    query_vec = embedder.embed_query(question)
    hits = store.search(query_vec, k, service, since, until, min_level)
    if not hits or context <= 0:
        return hits
    best = hits[0]
    anchor = store.onset(query_vec, best.service, best.distance + onset_margin, errors_only=best.error_count > 0) or best
    if anchor.id not in {h.id for h in hits}:
        hits.append(anchor)
    hits += store.temporal_neighbours(anchor, [h.id for h in hits], lookback, lookahead, context)
    return hits


def format_sources(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        tag = "context: same time window" if h.is_context else f"match, distance {h.distance:.3f}"
        blocks.append(
            f"[{i}] service={h.service} file={h.source} "
            f"{h.start_ts:%Y-%m-%d %H:%M:%S} -> {h.end_ts:%H:%M:%S} UTC ({tag})\n{h.content}"
        )
    return "\n\n".join(blocks)


def answer(question: str, hits: list[Hit], llm: ChatModel, on_token: Callable[[str], None] | None = None) -> Answer:
    if not hits:
        return Answer("No log excerpts matched the question (or the filters). Nothing to answer from.", [])
    user = f"Log excerpts:\n\n{format_sources(hits)}\n\nQuestion: {question}"
    return Answer(llm.chat(SYSTEM_PROMPT, user, on_token=on_token), hits)
