"""PostgreSQL + pgvector storage. Retrieval is a plain SQL ORDER BY on cosine distance."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence

import psycopg
from psycopg.rows import dict_row

from .chunk import Chunk

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS lograg_meta (
    key   text PRIMARY KEY,
    value text NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id           bigserial PRIMARY KEY,
    source       text        NOT NULL,
    service      text        NOT NULL,
    start_ts     timestamptz NOT NULL,
    end_ts       timestamptz NOT NULL,
    levels       text[]      NOT NULL,
    error_count  int         NOT NULL,
    content      text        NOT NULL,
    content_hash text        NOT NULL UNIQUE,
    embedding    vector({dim}) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_service_time ON chunks (service, start_ts);
CREATE INDEX IF NOT EXISTS chunks_time ON chunks (start_ts, end_ts);
"""

COLUMNS = "id, source, service, start_ts, end_ts, levels, error_count, content"
SEVERITY = ["TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL"]


@dataclass
class Hit:
    id: int
    source: str
    service: str
    start_ts: datetime
    end_ts: datetime
    levels: list[str]
    error_count: int
    content: str
    distance: float | None = None  # None for chunks added as temporal context

    @property
    def is_context(self) -> bool:
        return self.distance is None


def _vec(v: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


class Store:
    def __init__(self, dsn: str):
        self.conn = psycopg.connect(dsn, autocommit=True, row_factory=dict_row)
        # Wider HNSW candidate list than pgvector's default of 40: better recall, still fast.
        self.conn.execute("SET hnsw.ef_search = 100")

    def close(self) -> None:
        self.conn.close()

    def init_schema(self, dim: int, embed_model: str) -> None:
        self.conn.execute(SCHEMA.format(dim=int(dim)))
        row = self.conn.execute("SELECT value FROM lograg_meta WHERE key = 'embed_model'").fetchone()
        if row and row["value"] != embed_model:
            raise RuntimeError(
                f"This database was indexed with '{row['value']}', not '{embed_model}'. "
                "Vectors from different models are not comparable: run `lograg reset` first."
            )
        self.conn.execute(
            "INSERT INTO lograg_meta VALUES ('embed_model', %s) ON CONFLICT (key) DO NOTHING", (embed_model,)
        )

    def reset(self) -> None:
        self.conn.execute("DROP TABLE IF EXISTS chunks; DROP TABLE IF EXISTS lograg_meta;")

    def existing_hashes(self, hashes: Sequence[str]) -> set[str]:
        rows = self.conn.execute("SELECT content_hash FROM chunks WHERE content_hash = ANY(%s)", (list(hashes),))
        return {r["content_hash"] for r in rows}

    def insert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        """Insert chunks; re-ingesting the same file is a no-op thanks to the content hash."""
        with self.conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks (source, service, start_ts, end_ts, levels, error_count, content, content_hash, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                   ON CONFLICT (content_hash) DO NOTHING""",
                [
                    (c.source, c.service, c.start_ts, c.end_ts, c.levels, c.error_count, c.content, c.content_hash, _vec(v))
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )
            return cur.rowcount

    def search(
        self,
        query_vec: Sequence[float],
        k: int = 6,
        service: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        min_level: str | None = None,
    ) -> list[Hit]:
        where, params = [], {"q": _vec(query_vec), "k": k}
        if service:
            where.append("service = %(service)s")
            params["service"] = service
        if since:
            where.append("end_ts >= %(since)s")
            params["since"] = since
        if until:
            where.append("start_ts <= %(until)s")
            params["until"] = until
        if min_level:
            where.append("levels && %(levels)s")
            params["levels"] = SEVERITY[SEVERITY.index(min_level.upper()) :]
        sql = f"""
            SELECT {COLUMNS}, embedding <=> %(q)s::vector AS distance
            FROM chunks
            {"WHERE " + " AND ".join(where) if where else ""}
            ORDER BY embedding <=> %(q)s::vector
            LIMIT %(k)s
        """
        return [Hit(**row) for row in self.conn.execute(sql, params)]

    def onset(
        self, query_vec: Sequence[float], service: str, max_distance: float, errors_only: bool = False
    ) -> Hit | None:
        """Earliest chunk of a service that still matches the query closely: when did this start?

        With errors_only, healthy chunks that merely share vocabulary with the query
        ("Authorized payment ...") cannot be mistaken for the start of an error pattern.
        """
        row = self.conn.execute(
            f"""
            SELECT {COLUMNS}, embedding <=> %(q)s::vector AS distance
            FROM chunks
            WHERE service = %(service)s AND embedding <=> %(q)s::vector <= %(max)s
              AND (NOT %(errors_only)s OR error_count > 0)
            ORDER BY start_ts
            LIMIT 1
            """,
            {"q": _vec(query_vec), "service": service, "max": max_distance, "errors_only": errors_only},
        ).fetchone()
        return Hit(**row) if row else None

    def temporal_neighbours(
        self, anchor: Hit, exclude: Sequence[int], lookback: timedelta, lookahead: timedelta, n: int
    ) -> list[Hit]:
        """Chunks from *other* services close in time to the anchor: "what else happened around then?".

        The window is asymmetric on purpose: causes precede symptoms, and a bad deploy often
        surfaces 10-30 minutes later. Change events (deploys, rollbacks, config) rank first,
        then other errors, then plain proximity.
        """
        rows = self.conn.execute(
            f"""
            SELECT {COLUMNS}
            FROM chunks
            WHERE id <> ALL(%(exclude)s)
              AND service <> %(service)s
              AND start_ts <= %(t1)s AND end_ts >= %(t0)s
            ORDER BY content ~* '(deploy|rollout|release|rollback|restart|config)' DESC,
                     error_count > 0 DESC,
                     abs(extract(epoch FROM start_ts - %(t)s))
            LIMIT %(n)s
            """,
            {
                "exclude": list(exclude),
                "service": anchor.service,
                "t0": anchor.start_ts - lookback,
                "t1": anchor.end_ts + lookahead,
                "t": anchor.start_ts,
                "n": n,
            },
        )
        return [Hit(**row) for row in rows]

    def stats(self) -> dict:
        row = self.conn.execute(
            """SELECT count(*) AS chunks, count(DISTINCT service) AS services, count(DISTINCT source) AS sources,
                      min(start_ts) AS first, max(end_ts) AS last, coalesce(sum(error_count), 0) AS errors
               FROM chunks"""
        ).fetchone()
        return dict(row)
