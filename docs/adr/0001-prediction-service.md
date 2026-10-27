# ADR-0001: Prediction as a separate service

- Status: Accepted (Lab 2 design)
- Owner: Architect

## Context
In v1.0 the Streamlit UI trained the Binary-Filter and ran inference itself, so the UI and the model could not change independently.

## Decision
Inference runs in a separate FastAPI service with a REST API. The UI only calls the API. Both are deployed together with Docker Compose.

## Consequences
- Clear boundary between UI and model; the UI can be replaced without touching inference.
- One more deployable component and an API contract to keep stable.
