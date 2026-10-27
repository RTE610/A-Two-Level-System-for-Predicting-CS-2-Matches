# v2.0: goals (Lab 2) vs result

## Success criteria

| Criterion (Lab 2) | Result | Status |
|---|---|---|
| 1. User picks a match from the schedule and gets a prediction before it starts | Works end to end; a prediction or a reasoned rejection for 19 of 57 scheduled matches, "no data" for 38 | ◐ |
| 2. Data is refreshed regularly; accuracy measured over time | Accuracy by date ✓ (68.9% [64.0; 73.4]); refresh is a manual snapshot, schedule not agreed | ◐ |
| 3. Every change passes review, automated tests and CI | Tests and CI on every PR ✓; independent review not followed | ◐ |

## Innovations

| # | Innovation | Priority | Result |
|---|---|---|---|
| I1 | Team environment: CI/CD, tests, review, Docker | Must | ✅ CI, 70 tests, Docker, lock file · review ◐ |
| I2 | MLflow runs and model registry | Should | ✅ registry with `production` alias, rollback |
| I3 | Validation by date | Must | ✅ train / val / test by date, CI 95% |
| I4 | Data source | Must | ◐ snapshot with validation (ADR-0004), no automatic refresh |
| I5 | New UI | Must | ✅ schedule, match card, hypothetical mode |
| I6 | Prediction for a scheduled match | Must | ✅ with data-coverage limits (R10) |

## Requirements

- FR-1…FR-9: 9 / 9 accepted.
- NFR: 4 / 6 met — latency, reproducibility, observability, model quality; not met — freshness (71%), coverage (64%).

## Metrics

| | v1.0 | v2.0 |
|---|---|---|
| Accuracy with filter | 71.4% (random split) | 68.9% [64.0; 73.4] (by date) |
| Gain from the filter | +8.3 pp | +8.1 pp |
| Coverage | 42.7% | 36.4% |
| Training matches | ~3 000 | 6 403 labelled |
| Automated tests | 0 | 70 |
| Defects tracked with severity / priority | — | 9 |

## Plan vs actual

| Milestone (Lab 2 roadmap) | Planned | Actual |
|---|---|---|
| Sprint 1 — foundation | 23/10 | 19/10 |
| Sprint 2 — data and product | 06/11 | 27/10 |
| Testing | 20/11 | 27/10 (together with sprint 2) |
| Release v2.0 | 04/12 | 28/10 (Conditional Go) |
