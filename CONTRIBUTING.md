# Contributing

Thanks for your interest in the semantic-aware TSN scheduling framework.

## Getting started

1. Fork the repo and clone your fork.
2. `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
3. Run the fast test suite: `pytest tests/ -q -m "not slow"` (full suite incl. training tests: `pytest tests/ -q`)

## What we welcome

- **Bug reports with repros** — the simulator's event loop and schedulability
  checks are subtle; a failing test beats a description.
- **Reproducibility fixes** — seeding, metric definitions, experiment scripts.
- **Documentation** — QUICKSTART, experiment reports, paper materials.
- **New experiment modes** — e.g. shared-queue multiplexing, deadline < period
  configurations, non-periodic flows.

## Guidelines

- Keep the 250-test suite green before opening a PR.
- All training/evaluation must be seed-controlled (see `CSRLAgent`).
- Completion metrics follow the expected-packet definition (rejected or
  unscheduled flows count as incomplete) — do not reintroduce survivorship bias.
- Run `pytest tests/ -q -m "not slow"` locally; the full suite runs in CI on
  a schedule.

## Code of conduct

Be professional. This is a research codebase; disagreements about claims belong
in issues with data attached, not in rhetoric.
