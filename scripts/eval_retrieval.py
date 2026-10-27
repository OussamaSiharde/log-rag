"""Retrieval eval on the synthetic log set: does the right evidence reach the top k?

A question counts as a hit when at least one retrieved chunk (vector match or
temporal context) contains the expected marker. Run after ingesting sample-logs/:

    python scripts/eval_retrieval.py [-k 6]
"""

import argparse

from lograg.config import Settings
from lograg.ollama import OllamaClient
from lograg.rag import retrieve
from lograg.store import Store

CASES = [
    ("Why are customers getting payment failures?", "provider=stripe-eu"),
    ("When did the payment timeouts start?", "SocketTimeoutException"),
    ("What was deployed shortly before checkout started failing?", "payments v2.14.0"),
    ("Which commit changed the payment provider timeout?", "9f3c2e1"),
    ("Was the bad payments release rolled back?", "Rollback complete"),
    ("Why does checkout return 504 errors?", "504 Gateway Timeout"),
    ("Are orders retrying failed payments?", "attempt 3/3"),
    ("Is the inventory database connection pool exhausted?", "HikariPool-1"),
    ("What changed in the inventory service release?", "inventory v5.3.1"),
    ("Show the stack trace of the provider read timeout", "StripeEuClient.authorize"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=6)
    ap.add_argument("--context", type=int, default=3)
    args = ap.parse_args()
    cfg = Settings.from_env()
    store, client = Store(cfg.dsn), OllamaClient(cfg.ollama_url, cfg.embed_model, cfg.chat_model)
    hits_total = 0
    for question, marker in CASES:
        hits = retrieve(question, store, client, k=args.k, context=args.context)
        rank = next((i for i, h in enumerate(hits, 1) if marker in h.content), None)
        hits_total += rank is not None
        print(f"{'HIT ' if rank else 'MISS'} rank={rank or '-':<2} {question}")
    print(f"\nhit rate: {hits_total}/{len(CASES)} (k={args.k}, context={args.context}, embed={cfg.embed_model})")
    store.close()


if __name__ == "__main__":
    main()
