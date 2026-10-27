# ADR-0003: Split by match date

- Status: Accepted (Lab 2 design)
- Owner: Architect

## Context
v1.0 used a random 80/20 split and selected PCA components on the test set, which overestimates accuracy on future matches.

## Decision
Matches are sorted by date: ~70% train, ~15% validation (model and PCA selection), ~15% test (reported once, with a bootstrap 95% CI).

## Consequences
- Honest estimate for matches that have not happened yet (FR-5).
- Reported accuracy is lower than in v1.0; both numbers are reported.
