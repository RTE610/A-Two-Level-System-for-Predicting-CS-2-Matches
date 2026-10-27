# ADR-0004: Data source for v2.0

- Status: **Proposed** — waiting for approval by PO and PM
- Owner: Architect
- Issue: see `type:adr` issue "Approve ADR-0004"

## Context
Lab 2 left the data source open. v2.0 needs fresh team statistics and the schedule of upcoming matches in the format the model was trained on.

## Options

| Criterion | A. PandaScore API directly | B. Snapshot from the department (PhD students) |
|---|---|---|
| Feature format | Must be recomputed by us (high risk) | Same as training data (verified: 181 predictor features) |
| Freshness | Any schedule | Depends on agreement |
| Upcoming schedule | From API | In snapshot, but goes stale between deliveries |
| Control | Own API key | External dependency |
| Effort | 15 SP (tasks 5–6 in Lab 2 plan) | ~3 SP (loader + version pinning + dedup) |

## Decision (proposed)
Option B. The snapshot of 28.09.2026 is loaded through `src/snapshot.py`; the rest of the system depends only on the data-source interface (`src/data_source.py`), so option A can replace it later without changes in the service or UI.

## Open questions raised by the decision

| Question | Answer / follow-up |
|---|---|
| Which feature-set version to use (15 per match)? | `2.4/1`, the only one without gaps 12.2024–09.2026 |
| Game statistics missing from 02.2026 | Training cutoff 01.2026; request backfill (risk R6) |
| How often is the snapshot refreshed and who owns it? | Not agreed yet (risk R8) — request sent to PhD students |
| Where does the schedule for I6 come from? | 14-day window from the snapshot date; demo "as of" (risk R7) |

## Consequences
- Tasks 5 (API access) and 6 (feature recomputation) are cancelled/replaced — see `docs/scope-changes.md`.
- Product freshness depends on the department (risk R8).
