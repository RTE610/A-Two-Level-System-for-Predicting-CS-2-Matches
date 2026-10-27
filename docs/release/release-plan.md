# Release, deployment and rollback plan — v2.0.0

## Release plan

| Step | Action | Owner | Gate |
|---|---|---|---|
| 1 | Freeze `main`, create `release/v2.0.0` | TL | CI green |
| 2 | Regression on the release candidate (CI + real data) | QA | all tests pass; TC-21 quality ≥ baseline |
| 3 | Readiness checklist and Go / No-Go meeting | PM, PO, QA, Architect | decision recorded |
| 4 | Promote model to `production` alias | TL | `scripts/promote_model.py <version>` |
| 5 | Deploy (below) and post-deployment validation | TL, QA | `post_deploy_check.py` → GO |
| 6 | Tag `v2.0.0`, publish release notes | PM | — |
| 7 | Monitoring for 24 h | QA | no alerts except known R6 |

## Deployment plan

What is deployed: code (git tag) **and** a model version (MLflow alias `production`). They are released and rolled back as a pair.

```bash
git checkout v2.0.0
uv pip install -r requirements.lock                     # or: docker compose build
python scripts/prepare_snapshot.py --snapshot-dir <snapshot> --as-of 2026-09-28
python scripts/promote_model.py 2                        # production -> v2
docker compose up -d                                     # api :8000, ui :8501
python scripts/post_deploy_check.py --url http://localhost:8000 \
       --expected-version cs2-match-predictor/v2 --min-schedule 57
```

Configuration: `DATA_DIR`, `MLFLOW_TRACKING_URI`, optional `MODEL_VERSION` (pin), `API_URL` for the UI.
The API loads and checks the model at start-up; an unavailable or incompatible model stops the deployment.

## Rollback plan

| Trigger | Action | Expected time |
|---|---|---|
| Post-deployment validation FAIL, canary failure, 5xx | `python scripts/promote_model.py <previous>` → restart API → `post_deploy_check.py` | < 1 min |
| Code defect | `git checkout <previous tag>` + its model version → redeploy → validate | < 10 min |
| Bad data snapshot | restore previous `data/` (prepared files are versioned by `snapshot_meta.json`) → restart | < 5 min |

Rule: a model bundle is compatible only with the code that produced it. Model v1 (schema 1) is tagged `compatible_code=<2.0` and is blocked at start-up by `IncompatibleModelError`.

## Rehearsal (28.10.2026, release environment: Python 3.11, requirements.lock)

| Step | Result |
|---|---|
| Deploy code v2.0.0 + model v2 | started in 14 s; validation 6 / 6 PASS → GO |
| Incident: model v1 promoted by mistake | API refused to start: `IncompatibleModelError`; validation → ROLLBACK |
| Rollback: promote v2, restart | started in 5 s; validation PASS → GO; total 22 s |

Findings of the rehearsal (fixed before release):
1. Rolling back to model v1 caused 500 on every prediction while `/health` reported "ok" → `/health` now runs a canary prediction.
2. The model was loaded on the first request → now loaded and checked at start-up (fail fast).
3. Downtime during rollback equals restart time → improvement for v2.1: start the new instance next to the old one, validate, then switch (blue/green).

## Monitoring

`scripts/monitor_predictions.py <api log>` aggregates the JSON prediction log.

| Metric | Threshold | Release rehearsal (178 predictions) |
|---|---|---|
| latency p95 | < 2 s | 432 ms |
| canary failures | 0 | 0 |
| rejected share | ≤ 80% | 66% |
| incomplete-data share | ≤ 50% | **88% — alert (known risk R6)** |
