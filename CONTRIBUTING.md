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
- Changes in `src/` (model / data logic) are reviewed by the Tech Lead or the Architect.
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

## Commit messages

Short imperative description, no Conventional Commits prefixes, e.g. `Add temporal split`.
