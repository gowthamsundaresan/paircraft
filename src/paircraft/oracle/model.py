"""Model-agnostic contract for uncertainty-aware radical-pair oracles."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256
from math import isclose, isfinite
from string import hexdigits
from typing import Protocol

import numpy as np

from paircraft.coupling import CoordinateFrame


def _require_text(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be explicit")


def _require_sha256(value: str, name: str) -> None:
    digest = value.lower()
    if len(digest) != 64 or any(character not in hexdigits for character in digest):
        raise ValueError(f"{name} must be a 64-character hexadecimal digest")


class SpinModelFamily(StrEnum):
    """Physical model family used to generate a spin-chemical prediction."""

    HABERKORN_STATIC_PAIR = "haberkorn_static_pair"
    HABERKORN_STATIC_PAIR_WITH_ST_DEPHASING = "haberkorn_static_pair_with_st_dephasing"
    SEMICLASSICAL_HYPERFINE = "semiclassical_hyperfine"
    STOCHASTIC_DIFFUSIVE_PAIR = "stochastic_diffusive_pair"


class InitialSpinState(StrEnum):
    """Prepared electron-spin state of the radical pair."""

    SINGLET = "singlet"
    TRIPLET = "triplet"


class ObservableKind(StrEnum):
    """Spin-chemical observable requested from the oracle."""

    SINGLET_REACTION_YIELD = "singlet_reaction_yield"
    TRIPLET_REACTION_YIELD = "triplet_reaction_yield"
    SECONDARY_RADICAL_YIELD = "secondary_radical_yield"
    SINGLET_REACTION_YIELD_MFE = "singlet_reaction_yield_mfe"
    TRIPLET_REACTION_YIELD_MFE = "triplet_reaction_yield_mfe"
    SECONDARY_RADICAL_YIELD_MFE = "secondary_radical_yield_mfe"

    @property
    def is_relative(self) -> bool:
        """Return whether the observable is a target/reference relative change."""

        return self in {
            ObservableKind.SINGLET_REACTION_YIELD_MFE,
            ObservableKind.TRIPLET_REACTION_YIELD_MFE,
            ObservableKind.SECONDARY_RADICAL_YIELD_MFE,
        }


class ConditionStatus(StrEnum):
    """Whether an experimental condition is known or applicable."""

    VALUE = "value"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class FieldIntent(StrEnum):
    """Epistemic role of a field in a proposed protocol."""

    IDEAL_SIMULATION_TARGET = "ideal_simulation_target"
    CALIBRATED_APPLIED_FIELD = "calibrated_applied_field"
    MEASURED_FIELD = "measured_field"


class CalibrationTier(StrEnum):
    """Highest validation tier earned by an oracle artifact."""

    SIMULATOR_ONLY = "simulator_only"
    PUBLIC_RETROSPECTIVE = "public_retrospective"
    PROSPECTIVE_WET_LAB = "prospective_wet_lab"


class ApplicabilityStatus(StrEnum):
    """Relationship between a query and the oracle support domain."""

    IN_DOMAIN = "in_domain"
    BOUNDARY = "boundary"
    EXTRAPOLATION = "extrapolation"
    OUT_OF_DOMAIN = "out_of_domain"
    UNKNOWN = "unknown"


class AbstentionReason(StrEnum):
    """Machine-readable reason that a coherent query received no prediction."""

    UNSUPPORTED_SYSTEM = "unsupported_system"
    UNSUPPORTED_OBSERVABLE = "unsupported_observable"
    UNRESOLVED_PARAMETER_SET = "unresolved_parameter_set"
    INCOMPLETE_PROTOCOL = "incomplete_protocol"
    OUTSIDE_SUPPORTED_DOMAIN = "outside_supported_domain"
    MODEL_FAMILY_DISAGREEMENT = "model_family_disagreement"
    UNCALIBRATED_UNCERTAINTY = "uncalibrated_uncertainty"
    NUMERICAL_FAILURE = "numerical_failure"
    ORACLE_ARTIFACT_UNAVAILABLE = "oracle_artifact_unavailable"


@dataclass(frozen=True, slots=True)
class DeclaredCondition:
    """One condition whose unknown and not-applicable states remain explicit."""

    status: ConditionStatus
    value: float | str | None
    unit: str
    note: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, ConditionStatus):
            raise ValueError("condition status must be a ConditionStatus")
        _require_text(self.unit, "condition unit")
        _require_text(self.note, "condition note")
        if self.status is ConditionStatus.VALUE:
            if self.value is None:
                raise ValueError("a valued condition requires a value")
            if isinstance(self.value, float) and not isfinite(self.value):
                raise ValueError("numeric condition values must be finite")
            if isinstance(self.value, str):
                _require_text(self.value, "condition value")
        elif self.value is not None:
            raise ValueError("unknown or inapplicable conditions cannot carry a value")


@dataclass(frozen=True, slots=True)
class DeclaredQuantity:
    """One numerical condition with explicit known, unknown, or inapplicable status."""

    status: ConditionStatus
    value: float | None
    unit: str
    note: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, ConditionStatus):
            raise ValueError("quantity status must be a ConditionStatus")
        _require_text(self.unit, "quantity unit")
        _require_text(self.note, "quantity note")
        if self.status is ConditionStatus.VALUE:
            if self.value is None or not isfinite(self.value):
                raise ValueError("a valued quantity requires a finite numeric value")
        elif self.value is not None:
            raise ValueError("unknown or inapplicable quantities cannot carry a value")


@dataclass(frozen=True, slots=True)
class RequestedMagneticFluxDensity:
    """A proposed Cartesian magnetic flux density B, expressed only in tesla."""

    x_tesla: float
    y_tesla: float
    z_tesla: float
    coordinate_frame: CoordinateFrame
    intent: FieldIntent
    calibration_id: str | None = None
    component_standard_uncertainty_tesla: tuple[float, float, float] | None = None
    covariance_tesla_squared: (
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
        | None
    ) = None

    def __post_init__(self) -> None:
        if any(not isfinite(value) for value in self.components_tesla):
            raise ValueError("magnetic flux density components must be finite")
        if self.intent is FieldIntent.IDEAL_SIMULATION_TARGET:
            if (
                self.calibration_id is not None
                or self.component_standard_uncertainty_tesla is not None
                or self.covariance_tesla_squared is not None
            ):
                raise ValueError(
                    "an ideal simulation field cannot claim calibration or uncertainty"
                )
        elif self.calibration_id is None:
            raise ValueError("calibrated or measured fields require a calibration ID")
        elif not self.calibration_id.strip():
            raise ValueError("field calibration ID must be explicit")
        else:
            components = self.component_standard_uncertainty_tesla
            covariance = self.covariance_tesla_squared
            if components is None or covariance is None:
                raise ValueError("calibrated or measured fields require uncertainty and covariance")
            if (
                len(components) != 3
                or len(covariance) != 3
                or any(len(row) != 3 for row in covariance)
            ):
                raise ValueError("field uncertainty requires three components and a 3x3 covariance")
            if any(not isfinite(value) or value < 0 for value in components):
                raise ValueError("field component uncertainties must be finite and nonnegative")
            if any(not isfinite(value) for row in covariance for value in row):
                raise ValueError("field covariance values must be finite")
            for index in range(3):
                if not isclose(
                    covariance[index][index],
                    components[index] ** 2,
                    rel_tol=1e-12,
                    abs_tol=0.0,
                ):
                    raise ValueError("field covariance diagonal must match component uncertainties")
                for other in range(3):
                    if not isclose(
                        covariance[index][other],
                        covariance[other][index],
                        rel_tol=1e-12,
                        abs_tol=0.0,
                    ):
                        raise ValueError("field covariance must be symmetric")
            matrix = np.asarray(covariance, dtype=np.float64)
            eigenvalues = np.linalg.eigvalsh(matrix)
            scale = float(np.max(np.abs(matrix)))
            if scale != 0.0 and float(np.min(eigenvalues)) < -scale * 1e-12:
                raise ValueError("field covariance must be positive semidefinite")

    @property
    def components_tesla(self) -> tuple[float, float, float]:
        """Return the signed Cartesian components of B in tesla."""

        return (self.x_tesla, self.y_tesla, self.z_tesla)


@dataclass(frozen=True, slots=True)
class RadicalSpecies:
    """Structured chemical and isotopic identity of one radical."""

    identity: str
    formal_charge: int
    redox_state: str
    protonation_state: str
    isotopic_composition: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.identity, "radical identity"),
            (self.redox_state, "radical redox state"),
            (self.protonation_state, "radical protonation state"),
            (self.isotopic_composition, "radical isotopic composition"),
        ):
            _require_text(value, name)


@dataclass(frozen=True, slots=True)
class RadicalPairSystemRef:
    """Exact catalog and parameter identity of one radical-pair system."""

    system_id: str
    radical_one: RadicalSpecies
    radical_two: RadicalSpecies
    initial_spin_state: InitialSpinState
    parameter_set_id: str
    parameter_set_sha256: str
    parameter_catalog_version: str
    parameter_catalog_sha256: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.system_id, "system ID"),
            (self.parameter_set_id, "parameter-set ID"),
            (self.parameter_catalog_version, "parameter-catalog version"),
        ):
            _require_text(value, name)
        _require_sha256(self.parameter_set_sha256, "parameter-set sha256")
        _require_sha256(self.parameter_catalog_sha256, "parameter-catalog sha256")


@dataclass(frozen=True, slots=True)
class IlluminationProtocol:
    """Optical preparation with independently declared component knowledge."""

    wavelength: DeclaredQuantity
    irradiance: DeclaredQuantity
    duration: DeclaredQuantity
    dose: DeclaredQuantity
    timing: str
    note: str

    def __post_init__(self) -> None:
        _require_text(self.timing, "illumination timing")
        _require_text(self.note, "illumination note")
        quantities = (
            (self.wavelength, "m", "wavelength"),
            (self.irradiance, "W m^-2", "irradiance"),
            (self.duration, "s", "duration"),
            (self.dose, "J m^-2", "dose"),
        )
        for quantity, expected_unit, name in quantities:
            if quantity.unit != expected_unit:
                raise ValueError(f"illumination {name} must use {expected_unit}")
            if quantity.status is ConditionStatus.VALUE and (
                quantity.value is None or quantity.value <= 0
            ):
                raise ValueError(f"valued illumination {name} must be strictly positive")


@dataclass(frozen=True, slots=True)
class ObservationWindow:
    """Delay and integration window defining the requested outcome."""

    start_seconds: float
    end_seconds: float | None
    definition: str

    def __post_init__(self) -> None:
        if not isfinite(self.start_seconds) or self.start_seconds < 0:
            raise ValueError("observation start must be finite and nonnegative")
        if self.end_seconds is not None and (
            not isfinite(self.end_seconds) or self.end_seconds <= self.start_seconds
        ):
            raise ValueError("finite observation end must be later than its start")
        _require_text(self.definition, "observation-window definition")


@dataclass(frozen=True, slots=True)
class ExperimentProtocol:
    """Structured non-readout protocol for one spin-chemical calculation."""

    field: RequestedMagneticFluxDensity
    temperature_kelvin: float
    solvent: str
    viscosity: DeclaredQuantity
    orientation_regime: str
    ph: DeclaredCondition
    oxygen: DeclaredCondition
    illumination: IlluminationProtocol
    observation_window: ObservationWindow
    reaction_model_id: str
    relaxation_model_id: str

    def __post_init__(self) -> None:
        if not isfinite(self.temperature_kelvin) or self.temperature_kelvin <= 0:
            raise ValueError("temperature must be finite and positive in kelvin")
        if self.viscosity.unit != "Pa s":
            raise ValueError("viscosity must use Pa s")
        if self.viscosity.status is ConditionStatus.VALUE and (
            self.viscosity.value is None or self.viscosity.value < 0
        ):
            raise ValueError("valued viscosity must be nonnegative")
        for value, name in (
            (self.solvent, "solvent"),
            (self.orientation_regime, "orientation regime"),
            (self.reaction_model_id, "reaction model ID"),
            (self.relaxation_model_id, "relaxation model ID"),
        ):
            _require_text(value, name)

    def non_field_identity(self) -> tuple[object, ...]:
        """Return every protocol field except the declared magnetic intervention."""

        return (
            self.temperature_kelvin,
            self.solvent,
            self.viscosity,
            self.orientation_regime,
            self.ph,
            self.oxygen,
            self.illumination,
            self.observation_window,
            self.reaction_model_id,
            self.relaxation_model_id,
        )


@dataclass(frozen=True, slots=True)
class OracleQuery:
    """One completely resolved request to a radical-pair oracle."""

    schema_version: str
    query_id: str
    system: RadicalPairSystemRef
    target_protocol: ExperimentProtocol
    reference_protocol: ExperimentProtocol | None
    observable: ObservableKind
    model_family: SpinModelFamily

    def __post_init__(self) -> None:
        _require_text(self.schema_version, "query schema version")
        _require_text(self.query_id, "query ID")
        if self.observable.is_relative:
            if self.reference_protocol is None:
                raise ValueError("a relative observable requires a complete reference protocol")
            if (
                self.target_protocol.non_field_identity()
                != self.reference_protocol.non_field_identity()
            ):
                raise ValueError("target and reference may differ only in magnetic flux density B")
        elif self.reference_protocol is not None:
            raise ValueError("an absolute observable cannot carry a reference protocol")


@dataclass(frozen=True, slots=True)
class OracleIdentity:
    """Content-addressed identity of one oracle artifact and its training corpus."""

    oracle_id: str
    oracle_version: str
    artifact_sha256: str
    code_version: str
    training_manifest_id: str
    training_manifest_sha256: str
    parameter_catalog_version: str
    parameter_catalog_sha256: str
    supported_model_families: tuple[SpinModelFamily, ...]
    supported_observables: tuple[ObservableKind, ...]
    calibration_tier: CalibrationTier

    def __post_init__(self) -> None:
        for value, name in (
            (self.oracle_id, "oracle ID"),
            (self.oracle_version, "oracle version"),
            (self.code_version, "code version"),
            (self.training_manifest_id, "training-manifest ID"),
            (self.parameter_catalog_version, "parameter-catalog version"),
        ):
            _require_text(value, name)
        _require_sha256(self.artifact_sha256, "oracle artifact sha256")
        _require_sha256(self.training_manifest_sha256, "training-manifest sha256")
        _require_sha256(self.parameter_catalog_sha256, "parameter-catalog sha256")
        if not self.supported_model_families or not self.supported_observables:
            raise ValueError("oracle identity requires explicit supported families and observables")


@dataclass(frozen=True, slots=True)
class PredictiveDistribution:
    """Prediction interval and uncertainty decomposition for one scalar outcome."""

    mean: float
    lower: float
    upper: float
    coverage_probability: float | None
    unit: str
    method: str
    epistemic_standard_uncertainty: float | None
    aleatoric_standard_uncertainty: float | None
    simulator_discrepancy_standard_uncertainty: float | None
    numerical_standard_uncertainty: float | None
    decomposition_status: str
    uncertainty_assumptions: str
    calibration_status: str
    note: str

    def __post_init__(self) -> None:
        values = (self.mean, self.lower, self.upper)
        if any(not isfinite(value) for value in values):
            raise ValueError("prediction and interval values must be finite")
        if not self.lower <= self.mean <= self.upper:
            raise ValueError("prediction interval must contain its mean")
        if self.coverage_probability is not None and (
            not isfinite(self.coverage_probability) or not 0 < self.coverage_probability < 1
        ):
            raise ValueError("interval coverage probability must lie strictly between zero and one")
        uncertainties = (
            self.epistemic_standard_uncertainty,
            self.aleatoric_standard_uncertainty,
            self.simulator_discrepancy_standard_uncertainty,
            self.numerical_standard_uncertainty,
        )
        if any(value is not None and (not isfinite(value) or value < 0) for value in uncertainties):
            raise ValueError("reported standard uncertainties must be finite and nonnegative")
        for value, name in (
            (self.unit, "prediction unit"),
            (self.method, "prediction method"),
            (self.decomposition_status, "uncertainty decomposition status"),
            (self.uncertainty_assumptions, "uncertainty assumptions"),
            (self.calibration_status, "prediction calibration status"),
            (self.note, "prediction note"),
        ):
            _require_text(value, name)


@dataclass(frozen=True, slots=True)
class SupportBound:
    """One named numerical boundary of the oracle support domain."""

    dimension: str
    lower: float
    upper: float
    unit: str
    definition: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.dimension, "support-bound dimension"),
            (self.unit, "support-bound unit"),
            (self.definition, "support-bound definition"),
        ):
            _require_text(value, name)
        if not isfinite(self.lower) or not isfinite(self.upper) or self.lower > self.upper:
            raise ValueError("support bounds must be finite and ordered")


@dataclass(frozen=True, slots=True)
class ApplicabilityAssessment:
    """Structured assessment of whether a query lies within oracle support."""

    status: ApplicabilityStatus
    reasons: tuple[str, ...]
    unsupported_dimensions: tuple[str, ...]
    support_evidence_ids: tuple[str, ...]
    support_bounds: tuple[SupportBound, ...]
    model_family_disagreement: str

    def __post_init__(self) -> None:
        if not self.reasons:
            raise ValueError("applicability requires at least one reason")
        for reason in self.reasons:
            _require_text(reason, "applicability reason")
        for dimension in self.unsupported_dimensions:
            _require_text(dimension, "unsupported dimension")
        for evidence_id in self.support_evidence_ids:
            _require_text(evidence_id, "support evidence ID")
        _require_text(self.model_family_disagreement, "model-family disagreement status")


@dataclass(frozen=True, slots=True)
class LineageArtifact:
    """One content-addressed artifact contributing to an oracle result."""

    artifact_id: str
    sha256: str
    evidence_role: str
    evidence_status: str
    reuse_status: str
    membership: str
    exact_location: str
    conditions: str
    uncertainty_note: str
    source_uri: str
    source_artifact_sha256: str | None
    license: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.artifact_id, "lineage artifact ID"),
            (self.evidence_role, "evidence role"),
            (self.evidence_status, "evidence status"),
            (self.reuse_status, "reuse status"),
            (self.membership, "artifact membership"),
            (self.exact_location, "artifact exact location"),
            (self.conditions, "artifact conditions"),
            (self.uncertainty_note, "artifact uncertainty note"),
            (self.source_uri, "artifact source URI"),
            (self.license, "artifact license"),
        ):
            _require_text(value, name)
        _require_sha256(self.sha256, "lineage artifact sha256")
        if self.source_artifact_sha256 is not None:
            _require_sha256(self.source_artifact_sha256, "source artifact sha256")


@dataclass(frozen=True, slots=True)
class ResolvedExperimentSetup:
    """Exact system and protocols used for one returned prediction."""

    system: RadicalPairSystemRef
    target_protocol: ExperimentProtocol
    reference_protocol: ExperimentProtocol | None


@dataclass(frozen=True, slots=True)
class SolverSetting:
    """One named numerical setting retained with a simulator execution."""

    name: str
    value: str
    unit: str
    definition: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.name, "solver-setting name"),
            (self.value, "solver-setting value"),
            (self.unit, "solver-setting unit"),
            (self.definition, "solver-setting definition"),
        ):
            _require_text(value, name)


@dataclass(frozen=True, slots=True)
class NumericalExecution:
    """Structured code and solver identity for one deterministic calculation."""

    model_id: str
    model_family: SpinModelFamily
    solver_name: str
    solver_versions: tuple[tuple[str, str], ...]
    code_version: str
    code_artifact_sha256: str
    settings: tuple[SolverSetting, ...]
    achieved_relative_residual: float
    iteration_count: int
    deterministic: bool
    random_seed: int | None
    randomness_note: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.model_id, "execution model ID"),
            (self.solver_name, "solver name"),
            (self.code_version, "execution code version"),
            (self.randomness_note, "execution randomness note"),
        ):
            _require_text(value, name)
        _require_sha256(self.code_artifact_sha256, "execution code artifact sha256")
        if not self.solver_versions:
            raise ValueError("execution requires at least one solver dependency version")
        version_keys = tuple(key for key, _ in self.solver_versions)
        if version_keys != tuple(sorted(set(version_keys))):
            raise ValueError("solver dependency versions must have unique sorted keys")
        for key, version in self.solver_versions:
            _require_text(key, "solver dependency name")
            _require_text(version, "solver dependency version")
        setting_names = tuple(setting.name for setting in self.settings)
        if setting_names != tuple(sorted(set(setting_names))):
            raise ValueError("solver settings must have unique sorted names")
        if (
            not isfinite(self.achieved_relative_residual)
            or self.achieved_relative_residual < 0
            or self.iteration_count <= 0
        ):
            raise ValueError("execution convergence diagnostics are invalid")
        if self.deterministic and self.random_seed is not None:
            raise ValueError("deterministic execution cannot carry a random seed")
        if not self.deterministic and self.random_seed is None:
            raise ValueError("stochastic execution requires an explicit random seed")

    @property
    def configuration_sha256(self) -> str:
        """Hash the solver and configuration independently of its numerical outcome."""

        payload = {
            "model_id": self.model_id,
            "model_family": self.model_family,
            "solver_name": self.solver_name,
            "solver_versions": self.solver_versions,
            "code_version": self.code_version,
            "code_artifact_sha256": self.code_artifact_sha256,
            "settings": [asdict(setting) for setting in self.settings],
            "deterministic": self.deterministic,
            "random_seed": self.random_seed,
            "randomness_note": self.randomness_note,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode()).hexdigest()

    @property
    def execution_sha256(self) -> str:
        """Hash the configuration together with achieved convergence diagnostics."""

        payload = {
            "configuration_sha256": self.configuration_sha256,
            "achieved_relative_residual": self.achieved_relative_residual,
            "iteration_count": self.iteration_count,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class OraclePrediction:
    """Supported scalar prediction with explicit applicability and lineage."""

    query: OracleQuery
    resolved_setup: ResolvedExperimentSetup
    oracle: OracleIdentity
    executed_model_family: SpinModelFamily
    distribution: PredictiveDistribution
    applicability: ApplicabilityAssessment
    lineage: tuple[LineageArtifact, ...]
    claim_boundary: str
    execution: NumericalExecution | None = None
    executed_model_id: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.claim_boundary, "claim boundary")
        if self.resolved_setup != ResolvedExperimentSetup(
            self.query.system,
            self.query.target_protocol,
            self.query.reference_protocol,
        ):
            raise ValueError("resolved setup must preserve the exact resolved query")
        if self.executed_model_family is not self.query.model_family:
            raise ValueError("executed model family must match the resolved query")
        if (
            self.execution is not None
            and self.execution.model_family is not self.query.model_family
        ):
            raise ValueError("execution model family must match the resolved query")
        if self.execution is None:
            if self.executed_model_id is not None:
                raise ValueError("prediction without execution cannot claim an executed model ID")
        elif self.executed_model_id != self.execution.model_id:
            raise ValueError("executed model ID must match the numerical execution")
        if self.applicability.status in {
            ApplicabilityStatus.OUT_OF_DOMAIN,
            ApplicabilityStatus.UNKNOWN,
        }:
            raise ValueError("an unsupported query cannot carry a numeric prediction")
        if not self.lineage:
            raise ValueError("a prediction must retain at least one lineage artifact")
        if self.query.observable in {
            ObservableKind.SINGLET_REACTION_YIELD,
            ObservableKind.TRIPLET_REACTION_YIELD,
            ObservableKind.SECONDARY_RADICAL_YIELD,
        } and not (0 <= self.distribution.lower <= self.distribution.upper <= 1):
            raise ValueError("absolute reaction-yield intervals must lie in [0, 1]")
        if self.query.observable.is_relative and self.distribution.lower < -1:
            raise ValueError("relative reaction-yield change cannot be less than -1")


@dataclass(frozen=True, slots=True)
class OracleAbstention:
    """Scientifically coherent query for which the oracle returns no number."""

    query_id: str
    oracle: OracleIdentity
    reason: AbstentionReason
    unsupported_dimensions: tuple[str, ...]
    applicability: ApplicabilityAssessment
    note: str

    def __post_init__(self) -> None:
        _require_text(self.query_id, "abstention query ID")
        _require_text(self.note, "abstention note")
        if self.applicability.status not in {
            ApplicabilityStatus.OUT_OF_DOMAIN,
            ApplicabilityStatus.UNKNOWN,
        }:
            raise ValueError("an abstention must report unsupported or unknown applicability")


type OracleResult = OraclePrediction | OracleAbstention


class Oracle(Protocol):
    """Model-agnostic interface implemented by every Paircraft oracle."""

    @property
    def identity(self) -> OracleIdentity:
        """Return the content-addressed oracle identity."""

        ...

    def predict(self, query: OracleQuery) -> OracleResult:
        """Predict a supported query or return a structured abstention."""

        ...
