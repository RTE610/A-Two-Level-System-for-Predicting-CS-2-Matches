# Architecture review — v2.0 (27.10.2026)

Checked against ADR-0001…0004 on `main` after PR #26. Evidence is either a test (TC-xx) or a code reference.

| Check | Result | Evidence | Finding |
|---|---|---|---|
| A1 · inference in a separate service, UI does not train | ✅ | `ui/app.py` imports only `requests`, `pandas`, `streamlit` (TC-06); all predictions go through `services/api` | — |
| A2 · models from the MLflow registry | ✅ | service loads the latest `cs2-match-predictor` version and its bundle (TC-06), v2 in use | F2: compose mounts `mlruns` at the host path because MLflow stores absolute artifact paths — works only when compose runs from the repo where training ran |
| A3 · split by date in every evaluation path | ⚠️ → ✅ | `run_temporal.py` uses date split (TC-05) | F1: three v1.0 scripts (`train.py`, `evaluate.py`, `run_experiments.py`) still used a random split. **Fixed:** moved to `scripts/legacy_v1/` with a warning; v2.0 has one evaluation path |
| A4 · data source behind one interface | ✅ | API depends only on `DataSource`; snapshot specifics live in `SnapshotDataSource` and `scripts/prepare_snapshot.py` | — |
| No train/test leakage | ✅ | one feature-set version and dedup by match id; imputers, scaler, PCA and rejection-reason thresholds fitted on train only; profiles and head-to-head use matches strictly before *as of* | — |
| Offline metrics = served model | ✅ | evaluation and service use the same `TwoStageModel.team1_win_proba` (order-invariant) | — |
| NFR observability | ✅ | JSON log per prediction with model version and latency (TC-16) | — |
| NFR latency < 2 s | ✅ | /predict p95 279 ms; schedule 0.25 s / 2 ms cached (TC-13) | — |
| Dependencies | ⚠️ → ✅ | **Fixed:** `requirements.lock` (uv, Python 3.11) used by CI and Docker | — |
| Model artifact security | ℹ️ | bundles are loaded with joblib/cloudpickle | F3: load only from the team's own registry; never from user-supplied paths |

**Conclusion:** the implementation matches ADR-0001…0004. One deviation (F1) was found and fixed during the review; F2 and F3 are accepted as known limitations for a single-machine deployment.
