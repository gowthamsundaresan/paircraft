# Paircraft agent contract

## Purpose

Paircraft builds an experimentally calibrated, uncertainty-aware platform connecting
flavoprotein sequence to magnetic response. The repository must make scientific claims
auditable and keep the software reproducible.

This contract applies to the entire repository. Update it when a recurring, evidenced rule
is discovered. A process preference without a concrete instance, mechanism, and cost does
not belong here.

## Repository state

- The tracked repository represents the current intended system, not a diary of its evolution.
- Do not put phase numbers, dates, status journals, or temporary plans into tracked product
  paths.
- Private research, plans, dossiers, decision logs, and working notes live under
  `workbench/`. The whole directory is gitignored and must never be staged.
- Do not modify or stage files in the parent `/Users/gowtham/Projects` worktree.
- Do not commit secrets, credentials, proprietary papers, paywalled content, raw large
  datasets, generated trajectories, model weights, or generated results.
- Local scientific inputs and outputs belong in the ignored `data/`, `artifacts/`, and
  `outputs/` directories until a versioned data-storage contract exists.

## Python toolchain

- Use CPython 3.13 for development. The supported range is 3.13 through 3.15.
- Use uv for Python installation, dependency resolution, environments, locking, and builds.
- Add dependencies to `pyproject.toml`; never hand-edit `uv.lock`.
- Use a `src/` layout and absolute imports.
- Public functions and module boundaries must be typed. Mypy runs in strict mode.
- Ruff is the Python formatter and linter. Do not add a competing formatter or linter.
- Tests use pytest and should exercise public behavior rather than implementation details.
- A test earns its place only if removing it makes a distinct class of failure invisible.
  Scaffolding and build glue receive smoke coverage, not exhaustive tests.

## Canonical commands

```sh
uv sync --locked --all-groups
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
uv run pyproject-fmt --check pyproject.toml
uv run mdformat --check AGENTS.md README.md CONTRIBUTING.md SECURITY.md
uv run yamllint .github .pre-commit-config.yaml
uv run pip-audit
uv build
docker build --tag paircraft:local .
docker run --rm paircraft:local --version
```

Before handing off a change, run every gate relevant to the files changed. CI and local
development must use these same commands rather than equivalent variants.

## Scientific integrity

- Keep sequence/structure, spin dynamics, photocycle kinetics, and measured fluorescence as
  separately testable layers. Do not imply that validation of one validates another.
- Every scientific parameter must carry units, chemical state, experimental or computational
  conditions, provenance, and uncertainty where available.
- Distinguish measured, fitted, calculated, assumed, and inherited values.
- Preserve the exact source location and citation chain for extracted evidence.
- Use SI units internally. Boundary adapters may accept domain units such as microtesla,
  millitesla, nanoseconds, or megahertz, but conversions must be explicit and tested.
- Never silently mix magnetic field strength `H` with magnetic flux density `B`.
- Never infer magnetic flux density `B` from coil current without a versioned, cited calibration;
  preserve the instrument current as a separate control/readback quantity.
- Never treat a best-fit parameter set as ground truth when the model is non-identifiable.
- Never present retrospective fitting as prospective prediction.
- Never use millitesla-range agreement alone to validate a sub-Earth-field claim.
- Optimization must remain uncertainty-aware and guarded against out-of-distribution
  sequences. A simulator optimum is not an experimental result.
- Prefer primary sources. Mark preprints, non-peer-reviewed results, and unresolved conflicts
  explicitly.

## Engineering rules

- Keep domain logic independent of notebooks, CLIs, storage, and external APIs.
- Validate external data at ingestion boundaries and fail loudly on missing units or
  provenance.
- Treat downloaded pickle/object-array payloads as opaque and unsafe by default; do not deserialize
  them in the analysis process.
- Make randomness explicit and seedable. Record model, code, data, and configuration versions
  for computational results.
- Do not add abstractions or dependencies without a current caller.
- Preserve user changes and keep unrelated modifications out of the current change.
- Use `rg` for repository searches and `apply_patch` for manual file edits.
- Never weaken a quality gate merely to make CI green. Fix the cause or document a bounded,
  evidence-based exception.

## Security and dependencies

- Keep runtime dependencies minimal and locked.
- Pin GitHub Actions by full commit SHA and retain the version comment.
- GitHub workflows use least-privilege permissions and must not expose secrets to untrusted
  pull-request code.
- Containers run as a non-root user and contain only runtime requirements.
- Report vulnerabilities according to `SECURITY.md`; do not publish exploit details in a
  public issue.

## Documentation

- Update README and contributor documentation when commands, supported versions, or public
  behavior change.
- Documentation describes current behavior. Historical reasoning belongs in the ignored
  workbench or version-control history.
- Do not claim a capability, benchmark, partnership, license, or scientific result that has
  not been verified.
