"""End-to-end tests against a real pgvector database, with a fake embedder and LLM."""

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lograg.rag import SYSTEM_PROMPT, answer, ingest, retrieve

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sample_logs(tmp_path_factory):
    out = tmp_path_factory.mktemp("logs")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "gen_logs.py"), str(out)], check=True, capture_output=True)
    return sorted(out.glob("*.log"))


def test_ingest_is_idempotent(store, embedder, sample_logs):
    first = ingest(sample_logs, store, embedder)
    assert first.files == 5 and first.inserted == first.chunks > 100
    again = ingest(sample_logs, store, embedder)
    assert again.inserted == 0
    assert store.stats()["services"] == 5


def test_search_is_ordered_by_distance_and_finds_the_incident(store, embedder, sample_logs):
    ingest(sample_logs, store, embedder)
    hits = store.search(embedder.embed_query("Provider call failed op authorize SocketTimeoutException"), k=5)
    assert [h.distance for h in hits] == sorted(h.distance for h in hits)
    assert hits[0].service == "payments" and "stripe-eu" in hits[0].content


def test_filters(store, embedder, sample_logs):
    ingest(sample_logs, store, embedder)
    q = embedder.embed_query("timeout")
    assert {h.service for h in store.search(q, k=10, service="orders")} == {"orders"}
    since = datetime(2026, 10, 20, 14, 0, tzinfo=timezone.utc)
    assert all(h.end_ts >= since for h in store.search(q, k=10, since=since))
    assert all({"ERROR", "FATAL"} & set(h.levels) for h in store.search(q, k=10, min_level="ERROR"))


def test_temporal_context_surfaces_the_deploy(store, embedder, sample_logs):
    ingest(sample_logs, store, embedder)
    question = "payments provider stripe-eu authorize failed SocketTimeoutException StripeEuClient"
    hits = retrieve(question, store, embedder, k=3, context=3)
    assert hits[0].service == "payments"
    context = [h for h in hits if h.is_context]
    assert context and all(h.service != hits[0].service for h in context)
    assert any("payments v2.14.0" in h.content for h in context)


def test_answer_prompt_carries_numbered_sources(store, embedder, sample_logs, ):
    from conftest import EchoLLM

    ingest(sample_logs, store, embedder)
    llm = EchoLLM()
    hits = retrieve("payment failed timeout", store, embedder, k=2, context=1)
    result = answer("Why do payments fail?", hits, llm)
    system, user = llm.calls[0]
    assert system == SYSTEM_PROMPT
    assert user.startswith("Log excerpts:\n\n[1] service=") and "[3]" in user
    assert user.rstrip().endswith("Question: Why do payments fail?")
    assert result.sources == hits


def test_answer_without_hits_does_not_call_the_llm():
    from conftest import EchoLLM

    llm = EchoLLM()
    assert "Nothing to answer from" in answer("anything", [], llm).text
    assert llm.calls == []
