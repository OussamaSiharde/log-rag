from datetime import timedelta, timezone

from lograg.chunk import chunk_records
from lograg.parse import parse_line, parse_lines

ISO = "2026-10-20T12:07:01.120Z ERROR [payments] Provider call failed: provider=stripe-eu"
SPRING = "2026-10-20 12:07:03.004  WARN 4242 --- [orders] [nio-8080-exec-3] c.e.orders.PaymentGateway : Payment failed: order=4812"


def test_parses_iso_line():
    r = parse_line(ISO)
    assert (r.level, r.service, r.message) == ("ERROR", "payments", "Provider call failed: provider=stripe-eu")
    assert r.ts.tzinfo == timezone.utc and r.ts.minute == 7


def test_parses_spring_boot_line_as_utc():
    r = parse_line(SPRING)
    assert (r.level, r.service, r.message) == ("WARN", "orders", "Payment failed: order=4812")
    assert r.ts.tzinfo == timezone.utc


def test_level_aliases_are_normalised():
    assert parse_line("2026-10-20T12:00:00Z WARNING [x] m").level == "WARN"
    assert parse_line("2026-10-20T12:00:00Z CRITICAL [x] m").level == "FATAL"


def test_stack_trace_stays_with_its_record():
    lines = [ISO, "java.net.SocketTimeoutException: Read timed out", "\tat a.B.c(B.java:1)",
             "2026-10-20T12:07:02Z INFO [payments] next"]
    records = list(parse_lines(lines))
    assert len(records) == 2
    assert records[0].extra == ["java.net.SocketTimeoutException: Read timed out", "\tat a.B.c(B.java:1)"]
    assert "Read timed out" in records[0].text


def test_lines_before_first_record_and_blank_lines_are_ignored():
    assert [r.message for r in parse_lines(["garbage", "", "2026-10-20T12:00:00Z INFO [x] ok"])] == ["ok"]


def _records(n, service="svc", step_s=1, msg="hello {i}"):
    base = parse_line(f"2026-10-20T12:00:00Z INFO [{service}] x").ts
    out = []
    for i in range(n):
        r = parse_line(f"2026-10-20T12:00:00Z INFO [{service}] {msg.format(i=i)}")
        r.ts = base + timedelta(seconds=i * step_s)
        out.append(r)
    return out


def test_chunks_never_mix_services():
    recs = sorted(_records(5, "a") + _records(5, "b"), key=lambda r: r.ts)
    chunks = chunk_records(recs)
    assert {c.service for c in chunks} == {"a", "b"}
    assert all(c.content.count("\n") == 4 for c in chunks)


def test_chunks_respect_time_span_and_size():
    assert len(chunk_records(_records(10, step_s=120), max_span=timedelta(minutes=5))) == 4
    assert len(chunk_records(_records(60), max_records=25)) == 3


def test_repeated_messages_collapse_with_count():
    chunk = chunk_records(_records(30, msg="same message"))[0]
    assert "same message  (repeated x25)" in chunk.content
    assert chunk.content.count("\n") == 0


def test_chunk_metadata():
    recs = _records(3)
    recs[1].level = "ERROR"
    c = chunk_records(recs)[0]
    assert c.levels == ["INFO", "ERROR"] and c.error_count == 1
    assert c.start_ts < c.end_ts
    assert c.embedding_text().startswith("service: svc | levels: INFO,ERROR")
