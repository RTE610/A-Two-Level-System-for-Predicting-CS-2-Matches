# Handover / knowledge transfer — v2.0.0

## System in one picture
`snapshot → scripts/prepare_snapshot.py (validation, cutoff) → scripts/run_temporal.py (training, MLflow) → services/api (FastAPI) → ui (Streamlit)`

| Component | Code | Owner role |
|---|---|---|
| Data ingest and validation | `src/snapshot.py`, `src/schema.py`, `scripts/prepare_snapshot.py` | Dev / BA |
| Data access | `src/data_source.py` (`DataSource`, ADR-0004) | Architect |
| Training and evaluation | `scripts/run_temporal.py`, `src/splits.py`, `src/inference.py` | TL |
| Model registry | `src/tracking.py`, `scripts/promote_model.py` | TL |
| Prediction API | `services/api/app.py` | Dev |
| UI | `ui/app.py` | Dev |
| Tests | `tests/` (unit), `tests/qa/` (QA), `tests/test_regressions.py` | QA |

## Runbook

| Task | Command |
|---|---|
| New snapshot from the department | `python scripts/prepare_snapshot.py --snapshot-dir <dir> --as-of <date>` — aborts if > 1% records invalid |
| Retrain | `python scripts/run_temporal.py --mode fast --split temporal --extra-test-json data/matches_v2_post_cutoff.json` |
| Release a model | compare `results/v2/temporal/metrics.json` with the current one → `python scripts/promote_model.py <version>` → restart API |
| Roll back a model | `python scripts/promote_model.py <previous>` → restart API |
| Check a deployment | `python scripts/post_deploy_check.py --url <api> --expected-version cs2-match-predictor/v<N>` |
| Monitoring | `python scripts/monitor_predictions.py <api log>` |
| Run tests | `pytest`; against a live service: `SMOKE_URL=<api> pytest -m smoke` |

## Troubleshooting (from real incidents)

| Symptom | Cause | Action |
|---|---|---|
| API does not start: `IncompatibleModelError` | model from before v2.0.0 promoted | promote a v2-compatible version |
| `/health` 503 "canary failed" | model or data broken | check log `canary_failed`; roll back |
| Many "no data" in the schedule | teams missing in the snapshot | data issue → PhD students |
| "incomplete data" on most predictions | game statistics missing after 01.2026 | expected until backfill (R6) |
| Ingest aborted "invalid share" | snapshot schema changed | see the error counts in the log; contact data owner |
| Accuracy dropped after retraining | data drift or new cutoff | do not promote; compare with TC-21 baseline |

## Open items handed over
- #23, #25 (minor, data) · data request to PhD students · independent review and branch protection · blue/green deployment · coverage to 70%.

## Where to read more
`README.md` · `CONTRIBUTING.md` · `docs/adr/` · `docs/architecture-review.md` · `docs/qa/` · `docs/release/`.
