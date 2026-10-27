# Test plan — CS2 Match Predictor v2.0 (Lab 4)

**Scope:** integration of data source → prediction service → UI, and acceptance of FR-1…FR-9 and NFRs from Lab 2.
**Out of scope:** model changes (algorithms and features are frozen, Lab 2), load testing beyond a single user.
**Environments:** (1) CI — synthetic league data, every PR; (2) local — real snapshot 28.09.2026, model from MLflow registry.
**Entry criteria:** sprint 2 code merged, CI green. **Exit criteria:** no open critical/major bugs with P1; every FR has a passing test case.

| TC | Requirement | Type | What is checked | Automated in |
|---|---|---|---|---|
| TC-01 | FR-1 | Functional | Schedule shows teams, date and time for every match | `tests/qa/test_functional_api.py` |
| TC-02 | FR-2 | Functional | Every scheduled match has a result without user action; match card | `test_functional_api.py` |
| TC-03 | FR-3 | Functional | Rejected match: status + reason, no winner | `test_functional_api.py` |
| TC-04 | FR-4 | Functional | Data date in every response and in the team card | `test_functional_api.py` |
| TC-05 | FR-5 | Regression | No test match earlier than training matches | `tests/test_splits.py`, `tests/test_pipeline.py` |
| TC-06 | FR-6 | Integration | Service loads the latest model from the MLflow registry; UI has no model code | `tests/qa/test_integration.py` |
| TC-07 | FR-7 | Integration | Training run logs params, metrics and data hash | `tests/test_tracking.py` |
| TC-08 | FR-8 | Functional | No placeholder team names | `test_functional_api.py` |
| TC-09 | FR-9 | Functional | Any two teams; result labelled hypothetical | `test_functional_api.py` |
| TC-10 | API contract | Functional (negative) | Same team / invalid id / missing field → 422; unknown team or match → 404 | `test_functional_api.py` |
| TC-11 | Risk R6 | Functional | Profiles without game statistics are flagged | `test_functional_api.py` |
| TC-12 | FR-9 consistency | Functional | Swapping teams gives the same winner and probability | `test_functional_api.py` |
| TC-13 | NFR performance | Non-functional | p95 prediction latency < 2 s | `tests/qa/test_nfr.py` + real data run |
| TC-14 | NFR reproducibility | Non-functional | Same data and seed → same metrics | `test_nfr.py` + real data rerun |
| TC-15 | Task 8 | Functional (data) | Invalid snapshot records are counted by type; ingest aborts at > 1% | `tests/test_schema.py` + real ingest |
| TC-16 | NFR observability | Non-functional | Each prediction logged with teams, decision, model version, latency | `test_nfr.py` |
| TC-17 | NFR freshness | Non-functional | Team statistics not older than 7 days at prediction time | real data run (manual) |
| TC-18 | NFR maintainability | Non-functional | ruff and mypy clean; coverage of `src/` ≥ 70% | CI |
| TC-19 | Deployment | Smoke | `/health`, `/schedule`, `/teams`, `/predict` answer; `SMOKE_URL` runs it against a live stack | `tests/qa/test_smoke.py` |
| TC-20 | UI ↔ API | Integration | Streamlit UI renders the API responses without errors | `test_integration.py` |
| TC-21 | Model quality | Regression | Test accuracy with filter not below the Lab 3 baseline (68.9%, CI [64.0; 73.4]) | real data rerun (manual) |

**Defects** are filed with the *Bug report* template (`severity:*`, `priority:*`), fixed with a regression test in `tests/test_regressions.py`, labelled `status:retest` and closed by QA after retest.
