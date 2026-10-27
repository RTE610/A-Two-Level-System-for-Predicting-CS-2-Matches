# Architecture Decision Records

| ADR | Decision | Status | Owner | Approved by |
|---|---|---|---|---|
| [0001](0001-prediction-service.md) | Prediction as a separate service (FastAPI) | Accepted (Lab 2) | Architect | PO, PM |
| [0002](0002-mlflow-registry.md) | MLflow for runs and model registry | Accepted (Lab 2) | Architect | PO, PM |
| [0003](0003-temporal-split.md) | Train/validation/test split by match date | Accepted (Lab 2) | Architect | PO, PM |
| [0004](0004-data-source.md) | Data source: department snapshot behind one interface | **Proposed** | Architect | pending: PO, PM |

## Process

1. An open architectural question becomes an issue labelled `type:adr` with an owner (the Architect).
2. The owner compares the options against explicit criteria and records open questions.
3. The decision is written as an ADR with status `Proposed`.
4. PO and PM approve it in the issue; status becomes `Accepted`. Work that depends on the decision starts only after that.
5. A superseded decision is not deleted: its status becomes `Superseded by NNNN`.
