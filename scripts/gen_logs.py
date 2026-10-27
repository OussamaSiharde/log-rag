"""Generate a deterministic, synthetic multi-service log set with planted incidents.

Nothing here comes from a real system. The timeline (all UTC, 2026-10-20):

  09:00-15:00  normal traffic on gateway, orders, payments, inventory
  10:12-10:31  inventory: DB connection pool saturation warnings (a red herring)
  12:04        deployer: payments v2.14.0 rolled out (commit 9f3c2e1, raises provider timeout to 30s)
  12:07        payments: provider calls start timing out (SocketTimeoutException + stack traces)
  12:07-12:52  orders: "Payment failed ... timeout after 30s" with retries; gateway: 504s
  12:51        deployer: payments rolled back to v2.13.2
  12:53        errors stop
  14:20        deployer: inventory v5.3.1 rolled out (no impact)

Usage: python scripts/gen_logs.py [out_dir]   (default: sample-logs/)
"""

import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

START = datetime(2026, 10, 20, 9, 0, tzinfo=timezone.utc)
END = START + timedelta(hours=6)
DEPLOY = START.replace(hour=12, minute=4)
INCIDENT = START.replace(hour=12, minute=7)
ROLLBACK = START.replace(hour=12, minute=51)
POOL_START, POOL_END = START.replace(hour=10, minute=12), START.replace(hour=10, minute=31)

rng = random.Random(42)
PROVIDERS = ["stripe-eu", "adyen", "payplug"]
SKUS = ["SKU-1042", "SKU-2210", "SKU-3307", "SKU-4471", "SKU-5590"]


def iso(ts: datetime, level: str, service: str, msg: str) -> str:
    return f"{ts:%Y-%m-%dT%H:%M:%S}.{ts.microsecond // 1000:03d}Z {level:<5} [{service}] {msg}"


def spring(ts: datetime, level: str, logger: str, msg: str) -> str:
    # The orders service logs in Spring Boot's default console format.
    return f"{ts:%Y-%m-%d %H:%M:%S}.{ts.microsecond // 1000:03d} {level:>5} 4242 --- [orders] [nio-8080-exec-{rng.randint(1, 9)}] {logger} : {msg}"


def ticks(every_s: float):
    t = START
    while t < END:
        yield t
        t += timedelta(seconds=rng.expovariate(1 / every_s))


def in_incident(t: datetime) -> bool:
    return INCIDENT <= t < ROLLBACK + timedelta(minutes=2)


def payments() -> list[str]:
    out = []
    for t in ticks(9):
        provider = rng.choice(PROVIDERS)
        if in_incident(t) and provider == "stripe-eu" and rng.random() < 0.8:
            out.append(iso(t, "ERROR", "payments", f"Provider call failed: provider={provider} op=authorize timeout after 30000 ms"))
            out += [
                "java.net.SocketTimeoutException: Read timed out",
                "\tat java.base/sun.nio.ch.NioSocketImpl.timedRead(NioSocketImpl.java:278)",
                "\tat com.example.payments.provider.StripeEuClient.authorize(StripeEuClient.java:118)",
                "\tat com.example.payments.service.AuthorizationService.authorize(AuthorizationService.java:64)",
            ]
        else:
            out.append(iso(t, "INFO", "payments", f"Authorized payment provider={provider} amount={rng.randint(5, 400)}.{rng.randint(0, 99):02d} EUR in {rng.randint(80, 900)} ms"))
    return out


def orders() -> list[str]:
    out = []
    for t in ticks(7):
        order = rng.randint(4000, 9999)
        if in_incident(t) and rng.random() < 0.55:
            attempt = rng.randint(1, 3)
            level = "ERROR" if attempt == 3 else "WARN"
            out.append(spring(t, level, "c.e.orders.PaymentGateway", f"Payment failed: order={order}, provider=stripe-eu, timeout after 30s, attempt {attempt}/3"))
        else:
            out.append(spring(t, "INFO", "c.e.orders.OrderService", f"Order {order} created with {rng.randint(1, 5)} items"))
    return out


def gateway() -> list[str]:
    out = []
    for t in ticks(4):
        path = rng.choice(["/api/orders", "/api/catalog", "/api/cart", "/api/checkout"])
        if in_incident(t) and path == "/api/checkout" and rng.random() < 0.6:
            out.append(iso(t, "ERROR", "gateway", f"504 Gateway Timeout POST {path} upstream=orders latency=30012ms"))
        else:
            out.append(iso(t, "INFO", "gateway", f"200 GET {path} latency={rng.randint(8, 240)}ms"))
    return out


def inventory() -> list[str]:
    out = []
    for t in ticks(12):
        if POOL_START <= t < POOL_END and rng.random() < 0.5:
            out.append(iso(t, "WARN", "inventory", f"HikariPool-1 - Connection is not available, request timed out after 5000ms (active=20, idle=0, waiting={rng.randint(3, 14)})"))
        else:
            out.append(iso(t, "INFO", "inventory", f"Stock level {rng.choice(SKUS)} = {rng.randint(0, 500)}"))
    return out


def deployer() -> list[str]:
    def line(t, msg):
        return iso(t, "INFO", "deployer", msg)

    return [
        line(START + timedelta(minutes=1), "Cluster prod-eu-1 healthy, 14 deployments up to date"),
        line(DEPLOY - timedelta(minutes=1), "Starting rollout payments v2.14.0 (commit 9f3c2e1: raise provider read timeout to 30s, new stripe-eu client)"),
        line(DEPLOY, "Rollout complete: payments v2.14.0, 3/3 pods ready"),
        line(ROLLBACK - timedelta(minutes=1), "Starting rollback payments v2.14.0 -> v2.13.2 (requested by on-call)"),
        line(ROLLBACK, "Rollback complete: payments v2.13.2, 3/3 pods ready"),
        line(START.replace(hour=14, minute=19), "Starting rollout inventory v5.3.1 (commit 1b77ad0: stock cache TTL 60s)"),
        line(START.replace(hour=14, minute=20), "Rollout complete: inventory v5.3.1, 4/4 pods ready"),
    ]


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "sample-logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, gen in [("payments", payments), ("orders", orders), ("gateway", gateway),
                      ("inventory", inventory), ("deployer", deployer)]:
        lines = gen()
        (out_dir / f"{name}.log").write_text("\n".join(lines) + "\n")
        print(f"{out_dir / (name + '.log')}: {len(lines)} lines")


if __name__ == "__main__":
    main()
