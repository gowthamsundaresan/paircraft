# Contributing

## Development environment

Install uv, then create the locked development environment:

```sh
uv sync --locked --all-groups
uv run pre-commit install
```

The repository supports CPython 3.13 through 3.15 and uses Python 3.13 for local development
and container execution.

## Making changes

1. Read [AGENTS.md](AGENTS.md), including its scientific-integrity rules.
1. Add dependencies through `pyproject.toml`, then run `uv lock`; never edit
   `uv.lock` manually.
1. Keep domain behavior under `src/paircraft/` and test public behavior under `tests/`.
1. Run the relevant canonical gates listed in `AGENTS.md`.
1. Update current-state documentation when commands, support, or behavior change.

The ignored `workbench/` directory may contain private plans, evidence notes, and dossiers.
Its contents must never be committed.

## Tests

Tests are behavior contracts, not a coverage collection exercise. Add one case for each
distinct branch of observable behavior. A new case must fail for a reason no existing case
already covers.

## Scientific changes

Scientific implementations must state units, chemical state, conditions, provenance,
uncertainty, and the validation boundary. Retrospective fit quality is not evidence of
prospective predictive performance.

## Licensing

The repository does not currently grant an open-source license. Contributions require an
explicit contribution and ownership agreement before they can be accepted from external
contributors.
