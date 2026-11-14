# Paircraft

Paircraft is an early-stage computational platform for designing and validating
magnetosensitive flavoproteins. The intended system connects protein sequence and structural
ensembles to spin dynamics, photocycle kinetics, and experimentally measured fluorescence
while retaining provenance and predictive uncertainty.

The repository currently provides the reproducible Python package, command-line, testing,
continuous-integration, and container foundation. It does not yet claim a validated
sequence-to-magnetoresponse model.

## Photocycle/readout model

`paircraft.readout` implements the four-state phenomenological model from Ross, Lodesani,
and Aiello (bioRxiv 2026.07.08.737352, equations 1-6 and supplementary equations 16-21).
It numerically evolves ground, excited, singlet-radical-pair, and triplet-radical-pair
populations and supplies the paper's analytical steady state and fluorescence observable.

All kinetic rates and imposed singlet yields carry units, chemical state, structured citation
chains, conditions, evidence status, and uncertainty status. Unverified locations in an
originating source are explicit rather than guessed. The published default singlet yields are
marked as assumed. The package does not derive a yield from a magnetic field, spin
Hamiltonian, protein sequence, or structure; reproducing this readout layer is not a MagLOV2
prediction.

The publication does not provide machine-readable Supplemental Figure 1 values. Regression
tests therefore check its stated numerical/analytical identity and mathematical limiting
behavior instead of choosing tolerances from digitized plot points.

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
