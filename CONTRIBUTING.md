# Contributing

## Branching: GitHub Flow

- `main` is always releasable and protected: no direct pushes, merge only through a pull request.
- One branch per task, created from `main`:
  - `feature/<short-name>` — new functionality
  - `fix/<short-name>` — bug fixes
  - `chore/<short-name>` — tooling, CI, refactoring
  - `docs/<short-name>` — documentation only
- Keep branches short-lived (≤ 3 days) and PRs small (≤ ~400 changed lines when possible).
- Merge strategy: squash merge; delete the branch after merge.

## Code review rules

- Every PR needs **1 approving review** from someone other than the author.
- Changes in `src/`, `services/` and `scripts/` (model, data, inference) are reviewed by the **Architect**.
  The author is never the only reviewer; if the author is the Tech Lead, the Architect reviews.
- At the end of every sprint the Architect runs an **architecture review** against the ADRs in `docs/adr/`.
- Reviewer checks: correctness, data leakage risks, tests, readability.
- Author resolves every comment (fix or reply) before merge.
- Review turnaround target: within 24 hours.

## Definition of Done

A task is **Done** when:

1. Code is merged to `main` through a reviewed PR.
2. CI is green: `ruff check`, `ruff format --check`, `mypy`, `pytest`, `docker build`.
3. New logic has unit tests; overall coverage of new v2.0 modules ≥ 70%.
4. No data, secrets or model binaries in git (`data/`, `models/`, `mlruns/` are ignored).
5. Training runs are logged to MLflow (params, metrics, data version).
6. README updated if a CLI or workflow changed.
7. Acceptance criterion of the linked FR is demonstrably met.

## Local checks

```bash
uv pip install -e ".[dev]"
ruff check . && ruff format --check .
mypy
pytest --cov=src
```

## Architecture decisions

Open architectural questions become an issue labelled `type:adr`, owned by the Architect.
The decision is recorded in `docs/adr/` and approved by PO and PM before dependent work starts. See `docs/adr/README.md`.

## Scope changes inside a sprint

1. Anyone proposes the change in an issue labelled `type:scope-change` (what, why, impact).
2. The team estimates it; only work that has not started may be re-estimated, and the original estimate is kept.
3. The **PO** decides; the **PM** approves the impact on the plan. A task is added *instead of* something, not on top.
4. The change is recorded in `docs/scope-changes.md`.

Progress is reported by outcomes (what works now). Story points are a planning tool only.

## Bugs

Bugs are GitHub issues created from the *Bug report* template, labelled `type:bug`, `severity:*` and `priority:*`.

| Severity | Meaning |
|---|---|
| critical | Core function broken or results silently wrong (e.g. data leakage) |
| major | Feature works incorrectly, no workaround |
| minor | Incorrect behaviour with a workaround |
| trivial | Cosmetic |

| Priority | Meaning |
|---|---|
| P1 | Fix before the next demo |
| P2 | Fix within the sprint |
| P3 | Backlog |

Lifecycle: `open` → fix PR (`Fixes #N`, with a regression test) → label `status:retest` → QA retests → closed or reopened.

## Commit messages

Short imperative description, no Conventional Commits prefixes, e.g. `Add temporal split`.
