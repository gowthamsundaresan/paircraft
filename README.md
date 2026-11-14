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

## Spin-yield/readout interface

`paircraft.coupling` projects externally supplied field-dependent singlet yields through the
steady-state readout implementation. Every curve has one explicit spin-system/run identity;
each sample retains a stable point ID, the full magnetic flux density **B** vector in tesla,
coordinate-frame definition, covariance-aware field uncertainty, and the original yield
provenance and uncertainty. A reference is selected by point ID, never by approximate field
matching.

The output is a steady-state readout projection with normalized population total 1.0. It keeps
the exact input curve and readout rates and records calculated emission and relative fluorescence
change separately. Readout conditions are derived from the exact provenance-bearing rate values;
callers cannot attach a contradictory free-form condition. It does not run spin dynamics,
interpolate fields, fit kinetics, propagate uncertainty without a declared joint model, or make a
MagLOV2 or biological prediction.

## Experimental traces

`paircraft.experiments` provides an immutable, typed boundary for measured scalar traces. It
currently reads the Oxford MARY scalar OME-TIFF/configuration/schedule format and Oxford
key/value-header photodiode CSV files. Every trace retains exact content hashes, source versions,
locations, licenses, sample and construct labels, processing class, observable, units, acquisition
conditions, and magnetic-control current.

The folded MFE analysis implements an explicit OFF/ON sign convention,
`(I_on - I_off) / I_off`, and records its discard, steady-state, and spline-baseline configuration.
Calculated results retain the executing algorithm/version and explicitly state that uncertainty is
not calculated when no replicate hierarchy or joint error model is available.
Magnet current is retained as amperes and is never presented as magnetic flux density **B** without
a separate calibration. The reader refuses non-scalar TIFF frames, ambiguous terminal schedule
events, object/pickled arrays, inconsistent frame counts, and derived traces passed as raw input.
It does not segment cells, deserialize deposited pickle bundles, reproduce the unavailable QBI
curve, or establish a low-field biological response.

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
