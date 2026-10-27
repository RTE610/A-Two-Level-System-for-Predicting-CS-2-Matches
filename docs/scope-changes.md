# Sprint scope change log

Rule: a scope change inside a sprint is proposed in an issue labelled `type:scope-change`. The team estimates, the PO decides, the PM approves the impact on the plan. Estimates are changed only for work that has not started; the original estimate is kept in this log.

## Sprint 1 (09/10 – 23/10)

| # | Change | Reason | Type | Original → new | Decided (PO) | Approved (PM) |
|---|---|---|---|---|---|---|
| 1 | Task 5 "API access" removed | ADR-0004 proposes the snapshot, API not needed | Cancelled | 2 → 0 | PO, 27.10 | — |
| 2 | Task 6 "Features in training format" replaced by 6′ "Snapshot loader" | Different work: snapshot already in training format | Replaced | 13 → 3 | PO, 27.10 | — |
| 3 | Task 6a "Data quality report" added | 15 feature versions per match, game stats missing from 02.2026 | Added | — → 2 | PO, 27.10 | — |
| 4 | I3 "Temporal validation" estimated | Was in the Lab 2 roadmap for sprint 1 but had no estimate | Estimate fixed | — → 5 | PO, 27.10 | — |

Note: decisions for sprint 1 are recorded retroactively (after the sprint). From sprint 2 the rule applies before work starts.

## Sprint 2 (23/10 – 06/11)

| # | Change | Reason | Type | Original → new | Decided (PO) | Approved (PM) |
|---|---|---|---|---|---|---|
| 1 | Task 7 "Data update from API on schedule" → "Data update from snapshot (manual trigger)" | ADR-0004; snapshot is delivered manually | Replaced | 3 → 2 | PO, 27.10 | — |
| 2 | Task 11 adds "incomplete data" flag | Risk R6 | Extended | 3 → 3 | PO, 27.10 | — |
| 3 | Final architecture review added | Lab 3 feedback | Added | — → 3 | PO, 27.10 | — |
