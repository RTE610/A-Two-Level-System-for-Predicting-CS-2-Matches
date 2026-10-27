# Retrospective — v2.0 (draft from project facts; to be confirmed by the team)

## Start
- Assign reviewers and protect `main`: no merge without an approval from someone else.
- Test on real data from the first day: synthetic fixtures missed 6 of 9 defects.
- Agree a data contract with the PhD students (refresh schedule, contact, coverage) before planning.
- Record decisions (ADR, scope changes) when they are made, not at the end of the sprint.
- Rehearse rollback before the release, not after.

## Stop
- Merging PRs without an independent approval.
- Reporting progress in story points.
- Estimating tasks with unknown data as if the scope were known (13 SP → 3 SP).
- Treating a "200 OK" health check as proof the service works.

## Continue
- Small PRs with Definition of Done and CI.
- Honest validation by date and reporting both numbers.
- Bug report → fix → regression test → retest.
- Data validation at ingest.
- Status report by outcomes with RAG.

## Lessons learned
1. All plan deviations came from data: the riskiest dependency was outside the team.
2. A random split overestimated accuracy by ~5 pp; v1.0 numbers were optimistic.
3. Tests on clean synthetic data gave false confidence; exploratory testing on real data found 6 of 9 defects.
4. A shallow health check hid a broken model; only the rollback rehearsal revealed it.
5. Written rules are not followed rules: CONTRIBUTING required reviews, PRs were merged without them.
6. One person as PO, TL and EM made approvals formal; decisions need a second role.

## Improvement actions (v2.1)

| # | Action | Owner | Deadline |
|---|---|---|---|
| 1 | Send the data request, agree refresh schedule and coverage | PO | 2 weeks |
| 2 | Branch protection + required reviewer on `main` | EM | immediately |
| 3 | Representative test fixtures sampled from real data | QA | next sprint |
| 4 | Blue/green deployment: validate the new instance before switching | TL | next sprint |
| 5 | Raise coverage to 70% (evaluation, visualization) | Dev | next sprint |
| 6 | Retrain after the game-statistics backfill and compare with v2 | TL | after action 1 |
