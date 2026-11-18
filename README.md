# Paircraft

Paircraft currently provides auditable radical-pair simulation, evidence, wet-lab-audit, and agent-
benchmark infrastructure. It is being built toward an experimentally calibrated,
uncertainty-aware platform for designing magnetosensitive flavoproteins. The intended future system
connects protein sequence and structural ensembles to spin dynamics, photocycle kinetics, and
experimentally measured fluorescence while retaining provenance and predictive uncertainty.

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

## Oracle contract and reference implementation

`paircraft.oracle` defines a model-agnostic, uncertainty-aware query/result boundary for
parameterized flavin radical-pair systems. Queries use exact catalog and parameter hashes,
structured radical chemical states, Cartesian magnetic flux density **B** in tesla, structured
conditions, an explicit physical model family, and exact target/reference semantics. Results are a
tagged prediction-or-abstention union, so an unsupported query cannot accidentally carry a numeric
answer. Predictions retain applicability bounds, model/corpus identity, parameter-level provenance,
uncertainty status, and a claim boundary.

The package includes two current callers. A reduced one-proton FAD-radical-anion/tryptophan-radical-
cation reference oracle independently solves an eight-state Haberkorn system and supports absolute
singlet/triplet reaction yields and their target/reference relative changes. A separate Maeda 2012
simulator oracle exposes only the exact AtCry1 and EcPL E109A point-estimate systems, secondary-pair
yield and its relative magnetic effect, and the publication's 0-50 mT model domain. The two oracles
have distinct physical model-family identities because the Maeda system adds competing RP2
formation and singlet-triplet dephasing. Both abstain on unknown schema or parameter hashes,
unsupported observables, altered protocols, fields outside their exact domains, and numerical
failure.

Protocol quantities preserve knowledge component by component. The Maeda context therefore records
known 460 nm wavelength and 5 ns duration alongside unknown irradiance, areal dose, viscosity, and
oxygen state. The reported 5-7 mJ pulse energy is retained as context but is not converted to areal
dose without a beam area.

The numerical kernel is isolated behind the provenance-bearing oracle rather than exposed as a
bare-float public API. It has one fixed topology: two electrons, one isotropic spin-1/2 nucleus, an
initial electronic singlet, an unpolarized nuclear state, and strictly positive singlet/triplet
Haberkorn reaction rates. It is intentionally not a general protein spin solver: nuclear Zeeman,
exchange, dipolar coupling, relaxation, molecular motion, and anisotropic hyperfine interactions
are absent.

This implementation is an engine-conformance benchmark with assumed reaction rates and inherited
hyperfine input. It returns an uncalibrated deterministic point result and is explicitly marked
`simulator_only`; it is not the learned hybrid surrogate, a wet-lab-calibrated oracle, or a MagLOV
prediction. Its reduced kernel has been cross-checked against pinned RadicalPy commit
`3f547bd5afa628a34ff68218fbbc351418b3e247` with the same omitted interactions; that establishes
engine agreement only.

## Published five-nucleus spin model

`paircraft.spin` independently implements the reduced five-nucleus model reported by Maeda et al.
(PNAS 2012 and Supporting Information) for the published AtCry1 and EcPL E109A rate sets. The
Hamiltonian includes identical electron Zeeman terms and five signed isotropic hyperfine
interactions. Its kinetics include singlet-selective back recombination, spin-independent secondary
radical-pair formation, and the reported singlet-triplet dephasing model. The calculated observable
is the infinite-time secondary-pair yield, not fluorescence or the singlet/triplet product yield of
the smaller reference oracle.

The complete 192-state Hilbert model is solved with an energy-basis, Jacobi-preconditioned,
matrix-free GMRES calculation rather than constructing the roughly 22 GB dense Liouvillian. Each
result retains the exact Cartesian **B** vector, full provenance-bearing specification and parameter
hash, model identity, solver/version, achieved residual, uncertainty limitation, and claim boundary.
The two named protein systems identify published point-estimate rate sets; they are not
sequence-resolved models.

Tests verify analytical limits, probability conservation, isotropic rotation invariance, the exact
published parameter values, and the reported qualitative millitesla curve topology. This is a
simulator-only reproduction of the reduced model. The publication does not deposit the numerical
Figure 6 data, fit covariance, exact solver settings, or a complete AtCry construct identity, so
Paircraft does not claim exact fit reproduction, experimental calibration, predictive accuracy, or
validation below Earth's field.

Each absolute simulator prediction carrying a structured numerical execution may be wrapped as a
canonical `SimulationDatum`; this currently applies to the Maeda simulator oracle. The datum embeds
the complete prediction and execution, retains model/code/configuration hashes and dependency
grouping, and emits deterministic JSON. Relative predictions are rejected until composite
target/reference execution provenance exists. Evidence origin is structurally `simulation`, and the
only current corpus membership is `reference_only_not_training`. Paircraft does not call a single
field calculation an agent trajectory and does not assign training or holdout roles without a
separate versioned corpus manifest.

## Corpus manifest

`paircraft.corpus` provides the assignment authority connecting exact simulation and measurement
records without embedding their numerical payloads. Each entry references a datum hash and
multiple leakage-group hashes, records its physical evidence layer, evaluates every possible use,
and keeps scientific eligibility separate from documented commercial-computational and
redistribution permissions. Manifest serialization is canonical and content-addressed.

Version 1 structurally permits only `reference_only` assignment. The training, calibration, and
prospective-evaluation roles exist solely as explicit eligibility questions and must remain
`ineligible` or `unresolved`. A later schema must first define stable cross-version leakage keys,
sealed-outcome provenance, and the mapping between physical evidence layers. Exact rights evidence
is content-addressed; a permissive measurement-rights assertion must match the datum's authenticated
license member.

The current evidence supports only a reference inventory. York MagLOV is legally reusable with
attribution, but its ON field is an approximate uncalibrated scalar, its OFF magnitude is
unquantified, and no calibrated field vector, replicate-aware uncertainty, or validated
spin-to-fluorescence mapping is available. Maeda simulations are deterministic spin-layer
engine-conformance records with unresolved inherited reuse rights, not independent truth. They
coexist in an inventory but are not paired examples. Paircraft therefore currently has zero
manifest-authorized training, physical-calibration, or prospective-evaluation records. Real
generated manifests belong under ignored output paths rather than tracked source.

## Deterministic simulation sweeps

`paircraft.sweeps` plans and executes a reference field sweep through the existing Maeda simulator
oracle. The default design is the explicit Cartesian product of the two exact published
point-estimate systems and 14 ideal z-directed magnetic flux-density values from 0 to 50 mT,
including 10, 20, and 50 µT. It therefore materializes 28 content-addressed jobs without embedding
numerical results. There is no random sampling or seed.

Every target resolves through the authenticated parameter catalog, and execution reconstructs the
exact absolute-secondary-yield query before calling the provenance-bearing oracle. Completed jobs
are accepted only when their query, system, catalog, field, model family, and dependency group match
the plan. All fields for one parameter set share a scientific-family hash that a future
training-capable manifest must keep together rather than splitting them as independent parameter
draws.

This is an enumerated field sensitivity of two reduced published models, not a parameter
distribution, uncertainty ensemble, biological or mutation variability, MagLOV prediction,
surrogate-training corpus, or experimental low-field validation. The current public solver
deliberately rejects altered rates and hyperfines; those require a separately identified
hypothetical-sensitivity model.

The bundle runner executes this plan into one immutable canonical JSON file, resumes only from
individually authenticated partial results, and refuses overwrite. Callers must place generated
files under the ignored `outputs/` boundary required by this repository. Inspection is bound to the
caller's expected plan and replays each job through the canonical deterministic oracle before
accepting the stored bytes. First inspection is therefore compute-bearing: content hashes prove
internal consistency, not that a number was emitted by the intended solver.

The QA view preserves one row per planned field and explicitly pairs every derived relative effect
with its zero-field reference, parameter set, dependency group, execution configuration, and source
datum hashes. Curve summaries report only facts on the enumerated grid, including the exact 10, 20,
and 50 µT points when present; they do not interpolate peaks or crossings. QA publication first
authenticates the source bundle and labels the result simulator-only, uncertainty-unavailable, and
ineligible as evidence of biological variability, mutations, calibration, or prospective
prediction.

## Hypothetical local sensitivity

The separate Maeda local-sensitivity design asks a narrower engineering question: how does the
reduced spin model change when one declared parameter is perturbed while every other input remains
fixed? For each of the two published parent systems it evaluates a baseline plus lower and upper
log-scale perturbations of eight parameters at 0, 10, 20, 50, 100, and 200 µT. The complete design
contains 204 jobs in 34 parameter families and two parent-level dependency groups.

These levels are an experimental design for the simulator, not parameter uncertainty or samples
from a posterior. They are not mutations, proteins, biological replicates, or training examples.
Every result retains the signed baseline value, proposed value, delta, unit, model identity, family
identity, and parent dependency. This matters for negative hyperfine values, where a positive
log-scale change makes the signed value more negative.

The sensitivity bundle runner is locked and resumable. It checkpoints each job, authenticates
partials by exact oracle replay, publishes without replacement, and raises a distinct
published-but-durability-indeterminate error if the final name is linked but its directory cannot be
synchronized. Callers must keep generated bundles under ignored output paths. The grid-only QA report makes
exact same-family zero-field and same-parent baseline comparisons; it does not rank parameters or
interpolate between fields.

## Simulator-backed agent benchmark

`paircraft.benchmark` currently implements one task: adaptive field selection followed by prediction
of a withheld simulator yield at 200 µT. An agent sees evaluator-issued, receipt-bearing
observations at 0 and 10 µT,
may query at most two of 20, 50, and 100 µT, and returns one scalar prediction. The evaluator keeps
the target, oracle, parameter family, answer commitment, and transcript state outside the agent
artifact. Task IDs and receipts are opaque keyed hashes.

The deployable agent bundle contains only a standard-library contract, prompt, request/response
schemas, and a non-secret `manifest.json` inventory. Tests run it in an isolated Python process that
cannot import Paircraft's evaluator or oracle. A separate evaluator-side local runner accepts any
typed Python agent callback and can
execute five trials while retaining failed factory, agent, answer, and evaluator attempts. This
local callback interface is model-vendor-neutral, but same-process Python introspection is not a
security boundary; deployed agents require process isolation and transport RPC.

Scoring is deterministic numerical error plus interaction-policy compliance and query budget. Its
tolerance is a grading scale, not predictive uncertainty. The benchmark does not yet ask an agent
to design a wet-lab protocol, score scientific novelty, or calibrate an effect against oracle
uncertainty. It is a simulator-backed benchmark skeleton, not the learned hybrid oracle described
by the product thesis.

Run the complete demonstration. The caller must choose a generated-output location; the repository's
gitignored `outputs/` tree is recommended:

```sh
uv run paircraft run-benchmark-demo --output-root outputs/benchmark-demo
```

The command deterministically selects one non-baseline 200 µT simulator task, emits the isolated
agent contract bundle, runs a transparent nearest-field persistence baseline through the local
callback runner, and publishes a canonical content-addressed evaluator summary. It generates a
fresh evaluator nonce in memory and never serializes or returns it. The scientific target selection
is deterministic, but the nonce affects the opaque task, prompt, bundle, answer commitment, and
summary identities. Exact run artifacts therefore cannot be reproduced, and the commitment cannot
later be reopened after the nonce is discarded. The baseline is a heuristic, not a learned agent;
the demo remains simulator-only, uncalibrated, and non-secure in its local callback execution.

## Wet-lab audit boundary

`paircraft.audit` defines where a future wet-lab result connects without becoming benchmark truth.
The trusted evaluator can exact-replay a sealed simulator target; a prospective registration binds
that target to declared protocol and condition identities; and content authentication checks exact
source, analysis, and verifier artifact bytes.

Version 1 deliberately cannot emit `match` or `drift`. It has no external authority that verifies
seal chronology, deterministic extraction, independent preparations, interval-calibration
provenance, or a versioned required-condition schema. Every comparison therefore returns
`insufficient_evidence`, even if self-asserted intervals overlap or the measured scalar exactly
equals the simulator. Byte identity is not scientific validation, and an audit never promotes an
oracle's calibration tier or authorizes data for training.

## What remains for the product thesis

Paircraft does not yet contain a learned surrogate trained on eligible literature measurements and
spin trajectories, calibrated predictive uncertainty, a prospective wet-lab evaluation set, or a
structured experimental-proposal grader. Those require a leakage-safe data contract, documented
computational-use rights, independent experimental replicates, authenticated calibrated fields,
and a validated mapping between spin yield and the measured readout. Until then, the repository is
an auditable simulator, evidence inventory, persistence layer, and one oracle-separated,
process-separable adaptive-prediction benchmark—not a predictive sequence-to-magnetoresponse
platform.

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

For the final Oxford Nature Figure 1c deposit, Paircraft safely ingests the authenticated 153x900
processed single-cell trace matrix and reconstructs the illustratively selected row-40 display.
It also reports E2, a response-blind distribution of per-cell medians over technical cycles. E3 is
a separate same-acquisition pooled ratio over eligible cell-cycle numerators and denominators,
verified against the denominator-weighted eligible cell-cycle ratios; it does not weight the E2
per-cell medians. These are deposited ROI-derived traces: the final
deposit has no raw images, individual row-to-mask mapping, or complete segmentation parameters.
Paircraft therefore does not call this a raw segmentation reproduction or a biological-population
estimate.

The Oxford calibration-corpus command authenticates all five final-v3 MFE archives, freezes the
descriptive transform before outcome access, and records every numeric read in a durable,
hash-chained journal. It emits a compact Tier-D corpus with 644 anonymous deposited rows, their
technical cycles, and frame-phase trajectories. The acquisitions are described separately: the
corpus does not fit a model, create a holdout, establish biological replication, or rank variants.

Paircraft also authenticates and reads the CC BY 4.0 York MagLOV Figure 6 trace directly from its
versioned project archive. The record preserves the 498 deposited `Mean` counts-per-pixel values,
decodes the embedded PCO frame timestamps, and aligns the pinned script's unsynchronized
field-on/field-off annotations without executing downloaded code. The source reports approximately
10 mT when the field is on; this remains an uncalibrated approximate scalar, while the off-state
field is explicitly unquantified. Illumination is represented as separate approximate source
reports in SI units: 470 nm center wavelength, 20 nm half-bandwidth, and 500 W/m² irradiance, plus
continuous temporal mode. The trace is a deposited figure input whose pixel aggregation geometry
is not reported, not a raw-image stack or a spin-yield measurement. The supporting publication is
explicitly marked as a non-peer-reviewed preprint.

The canonical `MeasuredDatum` keeps source/member hashes, CC BY attribution, conditions,
replication and uncertainty status, limitations, and deterministic content hashes. This first
datum is structurally `measurement` and `reference_only_not_training`: the deposit contains no
independent-replicate evidence or pointwise uncertainty for this showcased trace, so it cannot
calibrate the oracle by itself.

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
uv run paircraft reproduce-oxford-fig1c \
  --processed-npz data/public-inputs/experiments/payloads/oxford-v3/59258873_MFE_single_cell_data_0.npz \
  --metadata data/public-inputs/experiments/payloads/oxford-v3/59258273_metadata.json \
  --notebook data/public-inputs/experiments/payloads/oxford-v3/59258753_Fig1_PanelC.ipynb \
  --output-dir outputs/oxford-fig1c
uv run paircraft build-oxford-calibration-corpus \
  --input-root data/public-inputs/experiments/payloads/oxford-v3 \
  --metadata data/public-inputs/experiments/payloads/oxford-v3/59258273_metadata.json \
  --output-root outputs/oxford-calibration \
  --result-name corpus
```

The output directory is published as one complete atomic unit and must not already exist. Paircraft
refuses to overwrite a prior scientific result.

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
