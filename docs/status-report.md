# Status report — v2.0, end of sprint 2 (27.10.2026)

**Overall: 🟡 AMBER** — the product works end to end and passed testing; data coverage and governance approvals are behind.

| Area | Status | Why |
|---|---|---|
| Functionality | 🟢 Green | FR-1…FR-9 implemented and accepted by tests; user can pick a scheduled match and get a prediction |
| Quality | 🟢 Green | 9 defects found, 7 fixed and retested; no open critical/major; CI green |
| Schedule | 🟢 Green | sprint 2 tasks done before 06/11; tech-debt task #16 moved to Lab 5 |
| Data | 🔴 Red | 38 of 57 scheduled matches have no data; game statistics missing after 01.2026; request to PhD students not sent yet (#9) |
| Governance | 🟡 Amber | ADR-0004 (#7) and scope changes (#8) waiting for PO/PM approval; architecture review (#17) not started; PRs merged without an independent approval |
| NFR | 🟡 Amber | latency, reproducibility, observability met; freshness 71% (#25); coverage 64% of 70% |

## Done (outcomes)

- Prediction service (FastAPI) takes the model from the MLflow registry; UI works only through the API.
- Schedule with automatic predictions; matches without data are shown with a reason.
- Rejected matches show a reason; uncertain data is flagged.
- Hypothetical prediction for any two teams, independent of team order.
- Snapshot ingest validates records and records the data version.
- Test plan by FR/NFR, 67 automated tests, bug process with severity/priority and retest.

## Deviations from plan

| Planned (Lab 2/3) | Actual | Reason |
|---|---|---|
| Data update from API on schedule (task 7) | Snapshot ingest with manual trigger | ADR-0004: data comes as a snapshot |
| Tech debt: legacy random split, pinned deps (task 13) | Not done → Lab 5 | Time went to 9 defects found in testing |
| Final architecture review (#17) | Not started | Waiting for the Architect |
| Approval of ADR-0004 and scope changes | Pending | Not done by PO/PM yet |

## Risks

| Risk | Status | Response | Owner |
|---|---|---|---|
| R6 game statistics missing after 01.2026 | Open, medium | Flag in UI; stress test 67.7% vs 68.9% | PO |
| R7 schedule in the snapshot goes stale | Open, medium | Demo "as of" 28.09.2026 | PM |
| R8 dependency on PhD students | **Open, high** | Send request #9; agree refresh schedule | PM |
| R9 ADR-0004 not approved | Open, high | Approve in #7 | Architect |
| R10 low data coverage: 67% of scheduled matches without data | **New, high** | Ask to include lower-tier teams (request #9, item 6) | PO |

## Next steps (Lab 5)

1. Approve ADR-0004 and scope changes; send the data request.
2. Architecture review (#17) before release.
3. Tech debt #16; release checklist and Go/No-Go.
