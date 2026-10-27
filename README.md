# CS2 Match Predictor v2.0

Two-stage prediction of Counter-Strike 2 match winners: a **Binary-Filter** decides whether a match is predictable, a **Match-Predictor** names the winner and the win probability.

```
department snapshot ─► ingest + schema validation ─► training (split by date) ─► MLflow registry
                                                                                     │
                                         Streamlit UI ◄── REST ── Prediction API ◄───┘
```

## Quick start

```bash
uv pip install -r requirements.lock            # or: pip install -r requirements.lock

# 1. Prepare the snapshot (validation, cutoff, team names, schedule)
python scripts/prepare_snapshot.py --snapshot-dir <path>/28.09.2026 --as-of 2026-09-28

# 2. Train and register the model in MLflow (split by match date)
python scripts/run_temporal.py --mode fast --split temporal --extra-test-json data/matches_v2_post_cutoff.json

# 3. Run the service and the UI
MLFLOW_TRACKING_URI=sqlite:///mlflow.db uvicorn services.api.app:app --port 8000
API_URL=http://localhost:8000 streamlit run ui/app.py
```

With Docker: `docker compose up` (API on :8000, UI on :8501); training job: `docker compose --profile jobs run --rm train`.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | model version, data snapshot and date |
| `GET /schedule?days=14` | scheduled matches with automatic predictions |
| `GET /matches/{id}` | match card: team statistics, head-to-head, prediction |
| `GET /teams?query=` · `GET /teams/{id}` | team search and profile |
| `POST /predict` | hypothetical prediction for any two teams |

## Configuration

| Variable | Default | Used by |
|---|---|---|
| `DATA_DIR` | `data` | API |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | training, API |
| `BUNDLE_PATH` | — | API (local bundle instead of the registry) |
| `API_URL` | `http://localhost:8000` | UI |

## Development

```bash
ruff check . && ruff format --check . && mypy && pytest
SMOKE_URL=http://localhost:8000 pytest -m smoke     # smoke tests against a running service
```

Process: `CONTRIBUTING.md` · decisions: `docs/adr/` · test plan and report: `docs/qa/` · status: `docs/status-report.md`.

Legacy v1.0 scripts (thesis reproduction, random split) are in `scripts/legacy_v1/`.
