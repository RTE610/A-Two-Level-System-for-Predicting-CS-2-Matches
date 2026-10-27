# Test report — CS2 Match Predictor v2.0 (Lab 4)

**Build:** `main` after PR #26 · **Model:** `cs2-match-predictor/v2` (MLflow registry) · **Data:** snapshot 28.09.2026, feature set 2.4/1 · **Date:** 27.10.2026

## Summary

| Metric | Value |
|---|---|
| Automated tests | 67 (unit, functional, integration, NFR, smoke, regression) — all passing in CI on Python 3.11 and 3.12 |
| QA test cases (test plan) | 21 (TC-01…TC-21) |
| QA suite on the build before fixes | 13 of 22 QA tests failed or errored |
| QA suite after fixes | 22 of 22 pass |
| Defects found | 9 — critical 1, major 3, minor 4, trivial 1 |
| Defects closed after retest | 7 |
| Open defects | 2 minor (#23 data, P3; #25 NFR freshness, P2) — both caused by the data source |
| Coverage `src/` + `services/` | 64% (target NFR 70%; v2.0 modules 87–98%) |

## Results by test case

| TC | Requirement | Result | Evidence |
|---|---|---|---|
| TC-01 | FR-1 schedule | Pass after fix #20 | 57 of 57 matches shown |
| TC-02 | FR-2 auto prediction | Pass | accepted 9, rejected 10, no data 38 |
| TC-03 | FR-3 rejection with reason | Pass | reason for every rejected match |
| TC-04 | FR-4 data date | Pass | `data_as_of` in every response and in UI |
| TC-05 | FR-5 split by date | Pass | train ≤ 27.09.2025 < val < test 08.11.2025–31.01.2026 |
| TC-06 | FR-6 model from registry | Pass | service loads `cs2-match-predictor/v2`; UI has no ML imports |
| TC-07 | FR-7 runs in MLflow | Pass | params, metrics with CI, data SHA-256 |
| TC-08 | FR-8 real names | Pass after fix #22 | 0 placeholders |
| TC-09 | FR-9 hypothetical pair | Pass after fixes #19, #28 | 105 of 105 pairs answered |
| TC-10 | Negative input | Pass | 422 / 404 as specified |
| TC-11 | Incomplete-data flag | Pass | 57 of 57 scheduled predictions flagged (risk R6) |
| TC-12 | Order invariance | Pass after fix #21 | 0 of 105 pairs order-dependent |
| TC-13 | NFR latency < 2 s | Pass after fix #27 | /predict p95 279 ms; /schedule 0.25 s first, 2 ms cached (was 4.5 s) |
| TC-14 | NFR reproducibility | Pass | retraining reproduced test metrics exactly |
| TC-15 | Schema validation | Pass | 104 of 11 186 invalid (0.93%) counted → #23 |
| TC-16 | NFR observability | Pass | JSON log per prediction with model version and latency |
| TC-17 | NFR freshness ≤ 7 days | **Fail** | 46 of 65 profiles (71%) → #25 |
| TC-18 | NFR maintainability | **Partial** | ruff, mypy clean; coverage 64% < 70% |
| TC-19 | Smoke | Pass | CI and live service (`SMOKE_URL`) |
| TC-20 | UI ↔ API | Pass after fixes #27–#29 | Streamlit AppTest + manual session |
| TC-21 | Model quality regression | Pass | 68.9% [64.0; 73.4] = Lab 3 baseline |

## Defects

| # | Title | Severity | Priority | Found by | Status |
|---|---|---|---|---|---|
| #19 | /predict returns 500 for 53% of team pairs | critical | P1 | exploratory API test on real data | Closed after retest |
| #20 | Schedule silently drops 38 of 57 matches | major | P1 | exploratory API test | Closed after retest |
| #21 | Prediction depends on team order | major | P2 | symmetry test over 105 pairs | Closed after retest |
| #27 | /schedule takes 4.5 s, UI freezes | major | P1 | exploratory UI session | Closed after retest |
| #22 | "Team 139218" instead of the name | minor | P2 | exploratory API test | Closed after retest |
| #28 | Hypothetical mode preselects the same team | minor | P2 | exploratory UI session | Closed after retest |
| #23 | 104 records with a winner outside the match | minor | P3 | schema validation | Open — external (PhD students) |
| #25 | 29% of profiles older than 7 days | minor | P2 | NFR test TC-17 | Open — data source (R8) |
| #29 | Stray "None" on UI pages | trivial | P3 | exploratory UI session | Closed after retest |

## Model quality (test set 08.11.2025–31.01.2026, 961 matches)

| | v1.0 (thesis) | v2.0 random split | v2.0 by date (served) |
|---|---|---|---|
| Accuracy with filter | 71.4% | 73.8% [69.6; 78.0] | **68.9% [64.0; 73.4]** |
| Accuracy without filter | 63.1% | 65.6% | 60.8% |
| Coverage | 42.7% | 42.0% | 36.4% |
| ROC-AUC with filter | — | 0.77 | 0.73 |
| Binary-Filter accuracy | 86.7% | 87.0% | 86.7% |
| Stress test 02–09.2026 (no game stats) | — | — | 67.8% |

## Stability assessment

No open critical or major defects. The service answers every request type on real data; failures are now explicit statuses (`no_data`, `rejected`) instead of errors. The main residual issue is data coverage: for 38 of 57 scheduled matches one team has no statistics, and every profile after 01.2026 lacks game statistics. These are limits of the snapshot, not of the code.

## Not tested

- Docker Compose run end-to-end (no Docker daemon in the test environment): the image is built in CI; API and UI were run as processes with the same commands.
- Load beyond a single user.
