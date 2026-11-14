# Paircraft

Paircraft is an early-stage computational platform for designing and validating
magnetosensitive flavoproteins. The intended system connects protein sequence and structural
ensembles to spin dynamics, photocycle kinetics, and experimentally measured fluorescence
while retaining provenance and predictive uncertainty.

The repository currently provides the reproducible Python package, command-line, testing,
continuous-integration, and container foundation. It does not yet claim a validated
sequence-to-magnetoresponse model.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- CPython 3.13 for development; CPython 3.13–3.15 are supported
- Docker for container builds

## Setup

```sh
uv sync --locked --all-groups
uv run pre-commit install
```

## Quality gates

```sh
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
uv run pyproject-fmt --check pyproject.toml
uv run mdformat --check AGENTS.md README.md CONTRIBUTING.md SECURITY.md
uv run yamllint .github .pre-commit-config.yaml
uv run pip-audit
uv build
```

## CLI

```sh
uv run paircraft --version
uv run python -m paircraft --version
```

## Container

```sh
docker build --tag paircraft:local .
docker run --rm paircraft:local --version
```

## Project structure

```text
src/paircraft/  installable application package
tests/          behavior-focused tests
.github/        continuous integration and dependency automation
workbench/      ignored private research and planning material
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development contract and
[SECURITY.md](SECURITY.md) for vulnerability reporting.

## License

No open-source license has been granted. All rights are reserved until a license is selected.
