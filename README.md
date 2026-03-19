# CS2 Match Predictor (refactored)

This project refactors the original notebook into a modular Python pipeline:
`JSON -> (header) -> DataFrame -> preprocessing -> distance features -> PCA sweep -> training`.

## Setup (local)

1. Create a virtual environment and install dependencies:
   - `uv pip install -r requirements.txt`

## Data preparation

1. Put your CS2 matches JSON file somewhere accessible.
2. (Optional) Provide `headers.txt` (one header per line).  
   If `--header-path` is not provided, headers are generated automatically from the JSON.

## Training

```bash
python scripts/train.py --json-path "./data/matches.json" --distance-set orig
```

Optional:
- `--header-path "./data/headers.txt"`
- `--distance-set aug` (uses augmented distance features)
- `--output-dir "./models"`

By default it uses `difference_distance` (as in the notebook).

## Outputs

Models are saved to `models/<run_id>/dist_<orig|aug>/`:
- `scaler.pkl`
- `best_setting_result.txt`
- `*_best_model.pkl`
- `*_best_pca.pkl`
- `dataset.csv`
- `pca_sweep_results.csv`

## Evaluation

Generate tables and figures:

```bash
python scripts/evaluate.py --model-dir "./models/<run_id>/dist_orig" --output-dir "./results"
```

## Run with Docker

Build the image from the repository root:

```bash
docker build -t cs2-match-predictor .
```

Train (example):

```bash
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" cs2-match-predictor ^
  python scripts/train.py --json-path "/app/data/matches.json" --distance-set orig --output-dir "/app/models"
```

Evaluate (example):

```bash
docker run --rm -v "$(pwd)/results:/app/results" cs2-match-predictor ^
  python scripts/evaluate.py --model-dir "/app/models/<run_id>/dist_orig" --output-dir "/app/results"
```

