# Release readiness checklist — v2.0.0 (28.10.2026)

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | All Must innovations (I1, I3, I4, I5, I6) delivered | ✅ | I4 delivered as snapshot ingest (ADR-0004) |
| 2 | FR-1…FR-9 accepted by test cases | ✅ | test report TC-01…TC-12 |
| 3 | No open critical / major defects | ✅ | 9 found, 7 closed; open: #23, #25 (minor, data) |
| 4 | Regression suite green on the release candidate | ✅ | 70 tests, CI on Python 3.11 / 3.12 |
| 5 | Model quality not below the Lab 3 baseline | ✅ | 68.9% [64.0; 73.4] with filter (TC-21) |
| 6 | NFR latency < 2 s | ✅ | /predict p95 ≈ 0.4–0.5 s in the release environment |
| 7 | NFR freshness: profiles ≤ 7 days | ❌ | 71% (#25) — data source limitation |
| 8 | NFR maintainability: coverage ≥ 70% | ⚠️ | 64%; v2.0 modules 87–98% |
| 9 | Architecture review passed | ✅ | docs/architecture-review.md, finding F1 fixed |
| 10 | Deployment from pinned dependencies | ✅ | requirements.lock, clean Python 3.11 environment |
| 11 | Rollback plan rehearsed | ✅ | model rollback 22 s; incompatible model blocked at start-up |
| 12 | Post-deployment validation and monitoring ready | ✅ | scripts/post_deploy_check.py, scripts/monitor_predictions.py |
| 13 | Release notes, handover, runbook | ✅ | docs/release/ |
| 14 | Data coverage for scheduled matches | ❌ | 38 of 57 scheduled matches without data (R10) |
| 15 | Independent code review on every PR | ❌ | not followed (R11) |

## Decision: **CONDITIONAL GO**

Release v2.0.0 with the 28.09.2026 snapshot, labelled "data as of 28.09.2026".

Conditions (owner, deadline):
1. UI keeps the "no data" / "incomplete data" / stale-profile labels visible — in place.
2. Data request to the PhD students is sent and a refresh schedule agreed — PO, before v2.1.
3. Independent review with branch protection from the next PR — EM, immediately.

No-Go triggers after deployment: any FAIL in `post_deploy_check.py`, a canary failure, p95 > 2 s.
