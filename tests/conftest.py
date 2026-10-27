import hashlib
import math
import os
import re

import psycopg
import pytest

from lograg.store import Store

DIM = 768
TEST_DSN = os.environ.get("LOGRAG_TEST_DSN", "postgresql://lograg:lograg@localhost:5442/lograg")


class HashEmbedder:
    """Deterministic bag-of-words embedder: lets the pipeline be tested without Ollama."""

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class EchoLLM:
    def __init__(self):
        self.calls = []

    def chat(self, system, user, on_token=None):
        self.calls.append((system, user))
        if on_token:
            on_token("stub answer [1]")
        return "stub answer [1]"


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def store():
    try:
        psycopg.connect(TEST_DSN, connect_timeout=2).close()
    except psycopg.OperationalError:
        pytest.skip("PostgreSQL/pgvector not reachable (docker compose up -d db)")
    s = Store(TEST_DSN)
    # Isolated schema so tests never touch an index you built with the CLI. Recreate the schema
    # instead of calling reset(): an unqualified DROP TABLE would fall through the search_path
    # to public.chunks whenever lograg_test is empty.
    s.conn.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
    s.conn.execute("DROP SCHEMA IF EXISTS lograg_test CASCADE; CREATE SCHEMA lograg_test;")
    s.conn.execute("SET search_path TO lograg_test, public")
    # HNSW is approximate; the hashing embedder spreads vectors far apart, where approximate
    # search can miss the true neighbour. Exact scans keep the tests deterministic.
    s.conn.execute("SET enable_indexscan = off")
    s.init_schema(DIM, "hash-test")
    yield s
    s.conn.execute("DROP SCHEMA lograg_test CASCADE")
    s.close()
