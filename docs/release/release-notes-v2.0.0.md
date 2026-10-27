# CS2 Match Predictor v2.0.0 — release notes (28.10.2026)

Data: department snapshot of 28.09.2026 (feature set 2.4/1). Model: `cs2-match-predictor` v2.

## What's new
- **Predictions for scheduled matches.** The schedule shows every upcoming match with an automatic prediction, a rejection with a reason, or "no data".
- **Hypothetical mode.** Prediction for any two teams; the result does not depend on the order of teams.
- **Match card.** Win rate, power rank, last match date and head-to-head count for both teams; data date on every screen.
- **Honest statuses.** Rejected by the Binary-Filter (with reason), "no data", "incomplete data" (no game statistics), stale profile.
- **Prediction service.** FastAPI service; the UI works only through it; the model comes from the MLflow registry.
- **Honest evaluation.** Train / validation / test split by match date; accuracy reported with a 95% CI.

## Quality
- Accuracy with filter 68.9% [64.0; 73.4], coverage 36.4%, ROC-AUC 0.73 (test 08.11.2025–31.01.2026). v1.0 reported 71.4% on a random split, which overestimates by ~5 pp.
- 70 automated tests; 9 defects found in testing, 7 fixed.

## Known issues
- 38 of 57 scheduled matches have no data for one of the teams (lower-tier teams are missing in the snapshot).
- Game statistics are missing from 02.2026; predictions are flagged "incomplete data" (stress test: 67.8%).
- 29% of team profiles are older than 7 days (#25).
- 104 snapshot records have a winner outside the match; excluded at ingest (#23).

## Upgrade notes
- Models trained before v2.0.0 (registry v1) are not compatible and are rejected at start-up.
- Dependencies are pinned in `requirements.lock`; v1.0 scripts moved to `scripts/legacy_v1/`.
