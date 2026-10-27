"""
Monitoring from the API's JSON prediction log (NFR observability).

Reads log lines, aggregates the last N predictions and flags thresholds agreed in the release plan.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

THRESHOLDS = {
    "latency_p95_ms": 2000.0,
    "rejected_share_max": 0.80,
    "incomplete_share_max": 0.50,
    "canary_failures_max": 0,
}


def parse(lines: list[str]) -> tuple[list[dict[str, object]], int]:
    events, canary_failures = [], 0
    for line in lines:
        match = re.search(r"(\{.*\})\s*$", line)
        if not match:
            continue
        try:
            event = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if event.get("event") == "prediction":
            events.append(event)
        elif event.get("event") == "canary_failed":
            canary_failures += 1
    return events, canary_failures


def summarize(events: list[dict[str, object]], canary_failures: int) -> tuple[dict[str, float], list[str]]:
    decisions = Counter(str(e["decision"]) for e in events)
    n = max(len(events), 1)
    latencies = sorted(float(e["latency_ms"]) for e in events)  # type: ignore[arg-type]
    stats = {
        "predictions": float(len(events)),
        "accepted_share": decisions["accepted"] / n,
        "rejected_share": decisions["rejected"] / n,
        "incomplete_share": sum(bool(e["incomplete_data"]) for e in events) / n,
        "latency_p95_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 2 else 0.0,
        "canary_failures": float(canary_failures),
    }
    alerts = []
    if stats["latency_p95_ms"] > THRESHOLDS["latency_p95_ms"]:
        alerts.append("latency p95 above 2 s")
    if stats["rejected_share"] > THRESHOLDS["rejected_share_max"]:
        alerts.append("filter rejects more than 80% of matches")
    if stats["incomplete_share"] > THRESHOLDS["incomplete_share_max"]:
        alerts.append("more than half of predictions use incomplete data (risk R6)")
    if stats["canary_failures"] > THRESHOLDS["canary_failures_max"]:
        alerts.append("canary prediction failed")
    return stats, alerts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path)
    parser.add_argument("--last", type=int, default=1000)
    args = parser.parse_args()
    events, canary_failures = parse(args.log.read_text(encoding="utf-8").splitlines())
    stats, alerts = summarize(events[-args.last :], canary_failures)
    for key, value in stats.items():
        print(f"{key:>18}: {value:.3f}" if isinstance(value, float) and value < 10 else f"{key:>18}: {value:.0f}")
    for alert in alerts:
        print(f"ALERT  {alert}")
    sys.exit(1 if alerts else 0)


if __name__ == "__main__":
    main()
