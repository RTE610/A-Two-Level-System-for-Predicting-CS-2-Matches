# ADR-0002: MLflow for training runs and model registry

- Status: Accepted (Lab 2 design)
- Owner: Architect

## Context
v1.0 stored models as loose `.pkl` files without parameters, metrics or data version.

## Decision
Every training run is logged to MLflow (params, metrics, data hash). Models are registered in the Model Registry; the prediction service loads the latest registered version.

## Consequences
- Reproducible runs and traceable model versions (FR-6, FR-7).
- The service depends on the tracking store being available at start-up.
