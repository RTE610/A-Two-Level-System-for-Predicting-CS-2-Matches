"""
Post-deployment validation of a running Prediction API (Lab 5 release plan, step 5).

Exit code 0 = all checks passed; 1 = at least one failed (trigger the rollback plan).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass

import httpx


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def run(base_url: str, expected_version: str | None, min_schedule: int, pairs: int) -> list[Check]:
    checks: list[Check] = []
    with httpx.Client(base_url=base_url, timeout=15) as api:
        health = api.get("/health")
        body = health.json() if health.status_code == 200 else {}
        checks.append(Check("health + canary prediction", health.status_code == 200, f"HTTP {health.status_code}"))
        if expected_version:
            actual = body.get("model_version")
            checks.append(Check("model version", actual == expected_version, f"{actual} (expected {expected_version})"))

        schedule = api.get("/schedule", params={"days": 14})
        items = schedule.json() if schedule.status_code == 200 else []
        checks.append(
            Check(
                "schedule served", schedule.status_code == 200 and len(items) >= min_schedule, f"{len(items)} matches"
            )
        )
        bad = [m for m in items if m["decision"] == "accepted" and not 0.5 <= m["win_probability"] <= 1.0]
        checks.append(Check("probabilities in range", not bad, f"{len(bad)} invalid"))

        teams = api.get("/teams", params={"limit": pairs + 1}).json()
        latencies, errors = [], 0
        for a, b in zip(teams, teams[1:], strict=False):
            started = time.perf_counter()
            response = api.post("/predict", json={"team1_id": a["id"], "team2_id": b["id"]})
            latencies.append(time.perf_counter() - started)
            errors += response.status_code >= 500
        p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 2 else float("nan")
        checks.append(Check("no 5xx on /predict", errors == 0, f"{errors} errors in {len(latencies)} calls"))
        checks.append(Check("latency p95 < 2 s", p95 < 2.0, f"{p95 * 1000:.0f} ms"))
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--expected-version", default=None, help="e.g. cs2-match-predictor/v2")
    parser.add_argument("--min-schedule", type=int, default=1)
    parser.add_argument("--pairs", type=int, default=20)
    args = parser.parse_args()
    try:
        checks = run(args.url, args.expected_version, args.min_schedule, args.pairs)
    except httpx.HTTPError as exc:
        print(f"FAIL  service unreachable: {exc!r}")
        sys.exit(1)
    for c in checks:
        print(f"{'PASS' if c.passed else 'FAIL'}  {c.name}: {c.detail}")
    ok = all(c.passed for c in checks)
    print("RESULT:", "GO" if ok else "ROLLBACK")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
