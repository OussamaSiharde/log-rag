"""Command line interface: lograg {init,ingest,search,ask,stats,reset}."""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .ollama import OllamaClient, OllamaError
from .rag import answer, ingest, retrieve
from .store import Store


def _when(value: str) -> datetime:
    ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _log_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        files.extend(sorted(f for f in p.rglob("*.log"))) if p.is_dir() else files.append(p)
    return files


def _add_filters(p: argparse.ArgumentParser) -> None:
    p.add_argument("question")
    p.add_argument("-k", type=int, default=6, help="number of chunks to retrieve (default 6)")
    p.add_argument("--context", type=int, default=3, help="extra same-time chunks from other services (default 3)")
    p.add_argument("--service", help="only search this service")
    p.add_argument("--since", type=_when, help="ISO timestamp, e.g. 2026-10-20T11:00")
    p.add_argument("--until", type=_when, help="ISO timestamp")
    p.add_argument("--level", choices=["WARN", "ERROR", "FATAL"], help="only chunks containing this level or worse")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lograg", description="Ask questions about your logs, fully offline.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="create the pgvector schema")
    ing = sub.add_parser("ingest", help="parse, chunk, embed and store log files")
    ing.add_argument("paths", nargs="+", help="log files or directories (*.log, recursive)")
    _add_filters(sub.add_parser("search", help="show the retrieved chunks, no LLM call"))
    _add_filters(sub.add_parser("ask", help="answer a question from the logs, with citations"))
    sub.add_parser("stats", help="what is indexed")
    sub.add_parser("reset", help="drop all indexed data")
    return parser


def _print_hits(hits) -> None:
    for i, h in enumerate(hits, start=1):
        tag = "context" if h.is_context else f"d={h.distance:.3f}"
        print(f"[{i}] {h.service:<12} {h.start_ts:%Y-%m-%d %H:%M:%S} -> {h.end_ts:%H:%M:%S}  {tag}  ({h.source})")
        for line in h.content.splitlines()[:6]:
            print(f"      {line[:150]}")
        if h.content.count("\n") >= 6:
            print("      ...")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Settings.from_env()
    store = Store(cfg.dsn)
    llm = OllamaClient(cfg.ollama_url, cfg.embed_model, cfg.chat_model)
    try:
        if args.cmd == "reset":
            store.reset()
            print("Index dropped.")
            return 0
        store.init_schema(cfg.embed_dim, cfg.embed_model)
        if args.cmd == "init":
            print("Schema ready.")
        elif args.cmd == "ingest":
            files = _log_files(args.paths)
            if not files:
                print("No .log files found.", file=sys.stderr)
                return 1
            t0 = time.perf_counter()
            r = ingest(files, store, llm)
            print(f"{r.files} files, {r.records} records -> {r.chunks} chunks, "
                  f"{r.inserted} new, in {time.perf_counter() - t0:.1f}s")
        elif args.cmd == "stats":
            for key, value in store.stats().items():
                print(f"{key:>9}: {value}")
        else:
            filters = dict(k=args.k, context=args.context, service=args.service,
                           since=args.since, until=args.until, min_level=args.level)
            t0 = time.perf_counter()
            hits = retrieve(args.question, store, llm, **filters)
            t_retrieve = time.perf_counter() - t0
            if args.cmd == "search":
                _print_hits(hits)
                print(f"\nretrieval: {t_retrieve * 1000:.0f} ms")
            else:
                print(f"Retrieved {len(hits)} excerpts in {t_retrieve * 1000:.0f} ms; asking {cfg.chat_model}...\n",
                      file=sys.stderr)
                result = answer(args.question, hits, llm,
                                on_token=lambda tok: print(tok, end="", flush=True))
                print("\n\nSources:")
                _print_hits(result.sources)
                print(f"\nretrieval: {t_retrieve * 1000:.0f} ms, total: {time.perf_counter() - t0:.1f}s ({cfg.chat_model})")
        return 0
    except OllamaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
