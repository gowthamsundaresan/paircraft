"""Typed, conservative comparison of oracle predictions with wet-lab evidence.

This module authenticates byte identity and registration binding only. Version 1
has no external scientific verifier, so it cannot emit match or drift. It does
not establish registration chronology, laboratory authorship, analysis validity,
replicate independence, interval calibration, promote an oracle's calibration
tier, or turn a measurement into benchmark ground truth.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import InitVar, asdict, dataclass
from enum import StrEnum
from hashlib import sha256
from math import isclose, isfinite
from string import hexdigits
from typing import Final

from paircraft.oracle.model import CalibrationTier, ObservableKind, SpinModelFamily
from paircraft.oracle.simulation import SimulationDatum

_REGISTRATION_SCHEMA: Final = "paircraft-prospective-wet-lab-audit-registration-v1"
_RESULT_SCHEMA: Final = "paircraft-authenticated-wet-lab-scalar-result-v1"
_REPORT_SCHEMA: Final = "paircraft-wet-lab-audit-report-v1"
_AUTHENTICATION_TOKEN: Final = object()
_REPORT_TOKEN: Final = object()
_PREDICTION_TOKEN: Final = object()
_REGISTRATION_TOKEN: Final = object()
_CONTENT_SEAL_SCOPE: Final = (
    "Receipt byte identity only; seal chronology and authority are unverified in v1."
)
_CONTENT_AUTHENTICATION_SCOPE: Final = (
    "Exact artifact bytes and registration binding only; lab authorship, analysis or extraction "
    "validity, replicate independence, interval calibration, and condition completeness are "
    "unverified in v1."
)
_CLAIM_BOUNDARY: Final = (
    "V1 content-only audit of one result against one exact oracle prediction. V1 can only report "
    "insufficient evidence: it has no external verifier for seal chronology or authority, "
    "analysis or extraction, replicate independence, interval calibration provenance, or a "
    "required condition schema. Authentication proves byte identity and registration binding "
    "only, not lab authorship, experimental validity, predictive calibration, biological "
    "validity, or eligibility as training or benchmark ground truth. This report never changes "
    "an oracle's calibration tier."
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be explicit")


def _digest(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in hexdigits for character in value)
        or value != value.lower()
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _finite(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
        raise ValueError(f"{label} must be a finite real scalar")
    return float(value)


class PhysicalEvidenceLayer(StrEnum):
    """Physical layer at which an outcome is defined."""

    SPIN_CHEMICAL_YIELD = "spin_chemical_yield"
    PHOTOCYCLE_KINETICS = "photocycle_kinetics"
    FLUORESCENCE_READOUT = "fluorescence_readout"


class IntervalCalibrationStatus(StrEnum):
    """Unverified declaration attached to an interval in the v1 content boundary."""

    DECLARED_CALIBRATED = "declared_calibrated_unverified"
    DECLARED_UNCALIBRATED = "declared_uncalibrated"


class ConditionEvidenceStatus(StrEnum):
    """Epistemic status of one result condition."""

    MEASURED = "measured"
    SOURCE_REPORTED = "source_reported"
    UNKNOWN = "unknown"


class AuditDisposition(StrEnum):
    """Conservative result of comparing one registered measurement and prediction."""

    MATCH_WITHIN_DECLARED_INTERVAL = "match_within_declared_interval"
    DRIFT_OUTSIDE_DECLARED_INTERVAL = "drift_outside_declared_interval"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class AuditReason(StrEnum):
    """Machine-readable reason contributing to an audit disposition."""

    TARGET_IDENTITY_MISMATCH = "target_identity_mismatch"
    REGISTRATION_BINDING_MISMATCH = "registration_binding_mismatch"
    PROTOCOL_MISMATCH = "protocol_mismatch"
    CONDITIONS_MISMATCH = "conditions_mismatch"
    OBSERVABLE_MISMATCH = "observable_mismatch"
    EVIDENCE_LAYER_MISMATCH = "evidence_layer_mismatch"
    UNIT_MISMATCH = "unit_mismatch"
    INSUFFICIENT_INDEPENDENT_REPLICATES = "insufficient_independent_replicates"
    PREDICTION_INTERVAL_UNAVAILABLE = "prediction_interval_unavailable"
    PREDICTION_INTERVAL_UNCALIBRATED = "prediction_interval_uncalibrated"
    MEASUREMENT_INTERVAL_UNAVAILABLE = "measurement_interval_unavailable"
    MEASUREMENT_INTERVAL_UNCALIBRATED = "measurement_interval_uncalibrated"
    COVERAGE_MISMATCH = "coverage_mismatch"
    SEAL_CHRONOLOGY_AUTHORITY_UNVERIFIED = "seal_chronology_authority_unverified"
    ANALYSIS_EXTRACTION_UNVERIFIED = "analysis_extraction_unverified"
    REPLICATE_INDEPENDENCE_UNVERIFIED = "replicate_independence_unverified"
    INTERVAL_CALIBRATION_PROVENANCE_UNVERIFIED = "interval_calibration_provenance_unverified"
    REQUIRED_CONDITION_SCHEMA_UNAVAILABLE_OR_UNKNOWN = (
        "required_condition_schema_unavailable_or_unknown"
    )


@dataclass(frozen=True, slots=True)
class AuditInterval:
    """One declared scalar interval; it is not assumed calibrated by default."""

    lower: float
    upper: float
    coverage_probability: float
    method: str
    calibration_status: IntervalCalibrationStatus

    def __post_init__(self) -> None:
        lower = _finite(self.lower, "interval lower bound")
        upper = _finite(self.upper, "interval upper bound")
        coverage = _finite(self.coverage_probability, "interval coverage")
        if lower > upper or not 0.0 < coverage < 1.0:
            raise ValueError("audit interval bounds or coverage are invalid")
        _text(self.method, "interval method")
        if not isinstance(self.calibration_status, IntervalCalibrationStatus):
            raise ValueError("interval calibration status must be explicit")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "coverage_probability", coverage)


@dataclass(frozen=True, slots=True)
class ObservedCondition:
    """One observed, source-reported, or explicitly unknown experimental condition."""

    name: str
    value: float | str | None
    unit: str
    status: ConditionEvidenceStatus
    exact_source_location: str
    standard_uncertainty: float | None = None

    def __post_init__(self) -> None:
        _text(self.name, "condition name")
        _text(self.unit, "condition unit")
        _text(self.exact_source_location, "condition source location")
        if not isinstance(self.status, ConditionEvidenceStatus):
            raise ValueError("condition evidence status must be explicit")
        if self.status is ConditionEvidenceStatus.UNKNOWN:
            if self.value is not None or self.standard_uncertainty is not None:
                raise ValueError("an unknown condition cannot carry a value or uncertainty")
            return
        if self.value is None or isinstance(self.value, bool):
            raise ValueError("a known condition requires a numeric or text value")
        if isinstance(self.value, int | float):
            object.__setattr__(self, "value", _finite(self.value, "condition value"))
        elif isinstance(self.value, str):
            _text(self.value, "condition value")
        else:
            raise ValueError("condition value must be numeric or text")
        if self.standard_uncertainty is not None:
            uncertainty = _finite(self.standard_uncertainty, "condition standard uncertainty")
            if uncertainty < 0 or not isinstance(self.value, float):
                raise ValueError("condition uncertainty requires a nonnegative numeric value")
            object.__setattr__(self, "standard_uncertainty", uncertainty)


@dataclass(frozen=True, slots=True)
class SourceArtifact:
    """Exact source artifact whose bytes are checked during result authentication."""

    role: str
    exact_location: str
    sha256: str

    def __post_init__(self) -> None:
        _text(self.role, "source-artifact role")
        _text(self.exact_location, "source-artifact location")
        _digest(self.sha256, "source-artifact sha256")


@dataclass(frozen=True, slots=True)
class WetLabProvenance:
    """Content identity and declared origin of a wet-lab result."""

    result_id: str
    acquisition_id: str
    source_uri: str
    source_artifacts: tuple[SourceArtifact, ...]
    analysis_artifact_sha256: str
    verifier_id: str
    verifier_version: str
    verifier_artifact_sha256: str
    authentication_scope: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.result_id, "result ID"),
            (self.acquisition_id, "acquisition ID"),
            (self.source_uri, "source URI"),
            (self.verifier_id, "verifier ID"),
            (self.verifier_version, "verifier version"),
            (self.authentication_scope, "authentication scope"),
        ):
            _text(value, label)
        for value, label in (
            (self.analysis_artifact_sha256, "analysis artifact sha256"),
            (self.verifier_artifact_sha256, "verifier artifact sha256"),
        ):
            _digest(value, label)
        if not self.source_artifacts or any(
            not isinstance(item, SourceArtifact) for item in self.source_artifacts
        ):
            raise ValueError("wet-lab provenance requires typed source artifacts")
        roles = tuple(item.role for item in self.source_artifacts)
        if roles != tuple(sorted(set(roles))):
            raise ValueError("source-artifact roles must be unique and sorted")
        if self.authentication_scope != _CONTENT_AUTHENTICATION_SCOPE:
            raise ValueError("v1 result authentication must remain content-only and unverified")


@dataclass(frozen=True, slots=True)
class AuditablePrediction:
    """Trusted-side identity and scalar needed for a future wet-lab comparison."""

    case_sha256: str
    query_sha256: str
    prediction_sha256: str
    execution_sha256: str
    oracle_artifact_sha256: str
    model_family: SpinModelFamily
    model_id: str
    calibration_tier: CalibrationTier
    observable: ObservableKind
    evidence_layer: PhysicalEvidenceLayer
    value: float
    unit: str
    interval: AuditInterval | None
    _construction_key: InitVar[object]

    def __post_init__(self, _construction_key: object) -> None:
        if _construction_key is not _PREDICTION_TOKEN:
            raise ValueError("audit predictions require an exact trusted adapter")
        for value, label in (
            (self.case_sha256, "case sha256"),
            (self.query_sha256, "query sha256"),
            (self.prediction_sha256, "prediction sha256"),
            (self.execution_sha256, "execution sha256"),
            (self.oracle_artifact_sha256, "oracle artifact sha256"),
        ):
            _digest(value, label)
        _text(self.model_id, "prediction model ID")
        _text(self.unit, "prediction unit")
        object.__setattr__(self, "value", _finite(self.value, "prediction value"))
        if not isinstance(self.model_family, SpinModelFamily):
            raise ValueError("prediction model family must be explicit")
        if not isinstance(self.calibration_tier, CalibrationTier):
            raise ValueError("prediction calibration tier must be explicit")
        if not isinstance(self.observable, ObservableKind):
            raise ValueError("prediction observable must be explicit")
        if self.evidence_layer is not PhysicalEvidenceLayer.SPIN_CHEMICAL_YIELD:
            raise ValueError("v1 oracle audit predictions must remain at the spin-yield layer")
        if self.interval is not None and not isinstance(self.interval, AuditInterval):
            raise ValueError("prediction interval must be typed")

    @property
    def target_sha256(self) -> str:
        """Hash the complete trusted audit target."""

        return _sha(asdict(self))


def auditable_prediction_from_simulation(
    *, case_sha256: str, datum: SimulationDatum
) -> AuditablePrediction:
    """Create a simulator-only target without inventing a calibrated interval."""

    if not isinstance(datum, SimulationDatum):
        raise ValueError("audit target requires a SimulationDatum")
    prediction = datum.prediction
    return AuditablePrediction(
        case_sha256=case_sha256,
        query_sha256=datum.query_sha256,
        prediction_sha256=datum.prediction_sha256,
        execution_sha256=datum.execution.execution_sha256,
        oracle_artifact_sha256=prediction.oracle.artifact_sha256,
        model_family=prediction.executed_model_family,
        model_id=datum.execution.model_id,
        calibration_tier=prediction.oracle.calibration_tier,
        observable=prediction.query.observable,
        evidence_layer=PhysicalEvidenceLayer.SPIN_CHEMICAL_YIELD,
        value=prediction.distribution.mean,
        unit=prediction.distribution.unit,
        interval=None,
        _construction_key=_PREDICTION_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class ProspectiveAuditRegistration:
    """Content-bound planned registration with unverified external chronology."""

    target_sha256: str
    case_sha256: str
    prediction_sha256: str
    observable: ObservableKind
    evidence_layer: PhysicalEvidenceLayer
    unit: str
    protocol_sha256: str
    conditions_sha256: str
    minimum_independent_replicates: int
    seal_authority: str
    seal_receipt_sha256: str
    seal_scope: str
    _construction_key: InitVar[object]
    schema_version: str = _REGISTRATION_SCHEMA

    def __post_init__(self, _construction_key: object) -> None:
        if _construction_key is not _REGISTRATION_TOKEN:
            raise ValueError("audit registrations require the content-binding factory")
        for value, label in (
            (self.target_sha256, "target sha256"),
            (self.case_sha256, "case sha256"),
            (self.prediction_sha256, "prediction sha256"),
            (self.protocol_sha256, "protocol sha256"),
            (self.conditions_sha256, "conditions sha256"),
            (self.seal_receipt_sha256, "seal receipt sha256"),
        ):
            _digest(value, label)
        for value, label in (
            (self.unit, "registration unit"),
            (self.seal_authority, "seal authority"),
            (self.seal_scope, "seal scope"),
        ):
            _text(value, label)
        if not isinstance(self.observable, ObservableKind) or not isinstance(
            self.evidence_layer, PhysicalEvidenceLayer
        ):
            raise ValueError("registration scientific identity must be typed")
        if (
            isinstance(self.minimum_independent_replicates, bool)
            or not isinstance(self.minimum_independent_replicates, int)
            or self.minimum_independent_replicates <= 0
        ):
            raise ValueError("minimum independent replicates must be a positive integer")
        if self.schema_version != _REGISTRATION_SCHEMA:
            raise ValueError("prospective registration schema is unsupported")
        if self.seal_scope != _CONTENT_SEAL_SCOPE:
            raise ValueError("v1 registration seal scope must remain content-only and unverified")

    @property
    def registration_sha256(self) -> str:
        """Hash the complete registration claim."""

        return _sha(asdict(self))


def create_prospective_audit_registration(
    target: AuditablePrediction,
    *,
    protocol_sha256: str,
    conditions: tuple[ObservedCondition, ...],
    minimum_independent_replicates: int,
    seal_authority: str,
    seal_receipt: bytes,
    seal_scope: str,
) -> ProspectiveAuditRegistration:
    """Bind a planned measurement to an exact target and external seal receipt.

    Verifying the receipt bytes proves their identity, not when or by whom they
    were created. ``seal_scope`` must state that external limitation.
    """

    if not isinstance(target, AuditablePrediction):
        raise ValueError("registration requires an AuditablePrediction")
    if not isinstance(conditions, tuple) or any(
        not isinstance(item, ObservedCondition) for item in conditions
    ):
        raise ValueError("registration conditions must be an immutable typed tuple")
    if not isinstance(seal_receipt, bytes) or not seal_receipt:
        raise ValueError("registration requires nonempty external seal-receipt bytes")
    return ProspectiveAuditRegistration(
        target_sha256=target.target_sha256,
        case_sha256=target.case_sha256,
        prediction_sha256=target.prediction_sha256,
        observable=target.observable,
        evidence_layer=target.evidence_layer,
        unit=target.unit,
        protocol_sha256=protocol_sha256,
        conditions_sha256=_sha([asdict(item) for item in conditions]),
        minimum_independent_replicates=minimum_independent_replicates,
        seal_authority=seal_authority,
        seal_receipt_sha256=sha256(seal_receipt).hexdigest(),
        seal_scope=seal_scope,
        _construction_key=_REGISTRATION_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class WetLabResultCandidate:
    """Untrusted scalar result awaiting exact artifact authentication."""

    registration_sha256: str
    protocol_sha256: str
    conditions: tuple[ObservedCondition, ...]
    observable: ObservableKind
    evidence_layer: PhysicalEvidenceLayer
    value: float
    unit: str
    interval: AuditInterval | None
    independent_replicates: int
    dependency_group_sha256: str
    provenance: WetLabProvenance

    def __post_init__(self) -> None:
        for value, label in (
            (self.registration_sha256, "registration sha256"),
            (self.protocol_sha256, "protocol sha256"),
            (self.dependency_group_sha256, "dependency-group sha256"),
        ):
            _digest(value, label)
        if not isinstance(self.conditions, tuple) or any(
            not isinstance(item, ObservedCondition) for item in self.conditions
        ):
            raise ValueError("result conditions must be an immutable typed tuple")
        if not isinstance(self.observable, ObservableKind) or not isinstance(
            self.evidence_layer, PhysicalEvidenceLayer
        ):
            raise ValueError("result scientific identity must be typed")
        object.__setattr__(self, "value", _finite(self.value, "result value"))
        _text(self.unit, "result unit")
        if self.interval is not None:
            if not isinstance(self.interval, AuditInterval):
                raise ValueError("result interval must be typed")
            if not self.interval.lower <= self.value <= self.interval.upper:
                raise ValueError("result interval must contain the reported value")
        if (
            isinstance(self.independent_replicates, bool)
            or not isinstance(self.independent_replicates, int)
            or self.independent_replicates <= 0
        ):
            raise ValueError("independent replicate count must be a positive integer")
        if not isinstance(self.provenance, WetLabProvenance):
            raise ValueError("result provenance must be typed")

    @property
    def conditions_sha256(self) -> str:
        return _sha([asdict(item) for item in self.conditions])

    @property
    def candidate_sha256(self) -> str:
        return _sha(asdict(self))


@dataclass(frozen=True, slots=True, init=False)
class ContentAuthenticatedWetLabResult:
    """Result constructible only after content checks, without scientific verification."""

    candidate: WetLabResultCandidate
    authentication_receipt_sha256: str
    schema_version: str = _RESULT_SCHEMA

    def __init__(
        self,
        *,
        candidate: WetLabResultCandidate,
        authentication_receipt_sha256: str,
        _token: object,
    ) -> None:
        if _token is not _AUTHENTICATION_TOKEN:
            raise ValueError("authenticated results require the evidence authenticator")
        _digest(authentication_receipt_sha256, "authentication receipt sha256")
        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(self, "authentication_receipt_sha256", authentication_receipt_sha256)
        object.__setattr__(self, "schema_version", _RESULT_SCHEMA)

    @property
    def result_sha256(self) -> str:
        """Hash the authenticated result and receipt identity."""

        return _sha(
            {
                "schema_version": self.schema_version,
                "candidate_sha256": self.candidate.candidate_sha256,
                "authentication_receipt_sha256": self.authentication_receipt_sha256,
            }
        )


def content_authenticate_wet_lab_result(
    candidate: WetLabResultCandidate,
    *,
    registration: ProspectiveAuditRegistration,
    source_payloads_by_role: Mapping[str, bytes],
    analysis_artifact_payload: bytes,
    verifier_artifact_payload: bytes,
) -> ContentAuthenticatedWetLabResult:
    """Authenticate exact bytes and registration binding only."""

    if not isinstance(candidate, WetLabResultCandidate) or not isinstance(
        registration, ProspectiveAuditRegistration
    ):
        raise ValueError("wet-lab authentication requires typed inputs")
    if candidate.registration_sha256 != registration.registration_sha256:
        raise ValueError("result does not bind to the supplied registration")
    artifacts = candidate.provenance.source_artifacts
    if set(source_payloads_by_role) != {artifact.role for artifact in artifacts}:
        raise ValueError("source payload roles contradict result provenance")
    for artifact in artifacts:
        payload = source_payloads_by_role[artifact.role]
        if not isinstance(payload, bytes) or sha256(payload).hexdigest() != artifact.sha256:
            raise ValueError("source payload bytes contradict result provenance")
    if (
        not isinstance(analysis_artifact_payload, bytes)
        or sha256(analysis_artifact_payload).hexdigest()
        != candidate.provenance.analysis_artifact_sha256
    ):
        raise ValueError("analysis artifact bytes contradict result provenance")
    if (
        not isinstance(verifier_artifact_payload, bytes)
        or sha256(verifier_artifact_payload).hexdigest()
        != candidate.provenance.verifier_artifact_sha256
    ):
        raise ValueError("verifier artifact bytes contradict result provenance")
    receipt_payload = {
        "registration_sha256": registration.registration_sha256,
        "candidate_sha256": candidate.candidate_sha256,
        "source_artifacts": [asdict(item) for item in artifacts],
        "verifier_id": candidate.provenance.verifier_id,
        "verifier_version": candidate.provenance.verifier_version,
        "verifier_artifact_sha256": candidate.provenance.verifier_artifact_sha256,
        "authentication_scope": candidate.provenance.authentication_scope,
    }
    return ContentAuthenticatedWetLabResult(
        candidate=candidate,
        authentication_receipt_sha256=_sha(receipt_payload),
        _token=_AUTHENTICATION_TOKEN,
    )


def content_authenticated_wet_lab_result_document(
    result: ContentAuthenticatedWetLabResult,
) -> dict[str, object]:
    """Serialize the complete content-authenticated candidate and provenance."""

    if not isinstance(result, ContentAuthenticatedWetLabResult):
        raise ValueError("result serialization requires a ContentAuthenticatedWetLabResult")
    document = json.loads(
        _canonical_bytes(
            {
                "schema_version": result.schema_version,
                "candidate": asdict(result.candidate),
                "candidate_sha256": result.candidate.candidate_sha256,
                "authentication_receipt_sha256": result.authentication_receipt_sha256,
                "result_sha256": result.result_sha256,
            }
        )
    )
    if not isinstance(document, dict):  # pragma: no cover - fixed object serialization
        raise AssertionError("authenticated result did not serialize to an object")
    return document


@dataclass(frozen=True, slots=True)
class WetLabAuditReport:
    """Immutable, content-addressed audit result without calibration promotion."""

    target_sha256: str
    registration_sha256: str
    result_sha256: str
    prediction_value: float
    observed_value: float
    unit: str
    model_family: SpinModelFamily
    model_id: str
    calibration_tier: CalibrationTier
    disposition: AuditDisposition
    reasons: tuple[AuditReason, ...]
    prediction_interval: AuditInterval | None
    measurement_interval: AuditInterval | None
    interval_overlap: bool | None
    _construction_key: InitVar[object]
    schema_version: str = _REPORT_SCHEMA
    claim_boundary: str = _CLAIM_BOUNDARY

    def __post_init__(self, _construction_key: object) -> None:
        if _construction_key is not _REPORT_TOKEN:
            raise ValueError("audit reports require authenticated comparison construction")
        for value, label in (
            (self.target_sha256, "target sha256"),
            (self.registration_sha256, "registration sha256"),
            (self.result_sha256, "result sha256"),
        ):
            _digest(value, label)
        _text(self.unit, "audit unit")
        _text(self.model_id, "audit model ID")
        object.__setattr__(self, "prediction_value", _finite(self.prediction_value, "prediction"))
        object.__setattr__(self, "observed_value", _finite(self.observed_value, "observation"))
        if not isinstance(self.disposition, AuditDisposition) or not self.reasons:
            raise ValueError("audit disposition and reasons must be explicit")
        if self.reasons != tuple(sorted(set(self.reasons), key=str)):
            raise ValueError("audit reasons must be unique and canonically sorted")
        if self.schema_version != _REPORT_SCHEMA or self.claim_boundary != _CLAIM_BOUNDARY:
            raise ValueError("audit report schema or claim boundary is unsupported")
        required_v1_reasons = {
            AuditReason.SEAL_CHRONOLOGY_AUTHORITY_UNVERIFIED,
            AuditReason.ANALYSIS_EXTRACTION_UNVERIFIED,
            AuditReason.REPLICATE_INDEPENDENCE_UNVERIFIED,
            AuditReason.INTERVAL_CALIBRATION_PROVENANCE_UNVERIFIED,
            AuditReason.REQUIRED_CONDITION_SCHEMA_UNAVAILABLE_OR_UNKNOWN,
        }
        if (
            self.disposition is not AuditDisposition.INSUFFICIENT_EVIDENCE
            or self.interval_overlap is not None
            or not required_v1_reasons.issubset(self.reasons)
        ):
            raise ValueError("v1 content-only audit reports must remain insufficient evidence")

    @property
    def report_sha256(self) -> str:
        """Hash the complete audit report claim."""

        return _sha(asdict(self))


def audit_wet_lab_result(
    target: AuditablePrediction,
    registration: ProspectiveAuditRegistration,
    result: ContentAuthenticatedWetLabResult,
) -> WetLabAuditReport:
    """Return the fail-closed v1 content audit.

    A future external scientific verifier must produce a new verified evidence
    schema and adapter; it must not relax or reinterpret this v1 result.
    """

    if (
        not isinstance(target, AuditablePrediction)
        or not isinstance(registration, ProspectiveAuditRegistration)
        or not isinstance(result, ContentAuthenticatedWetLabResult)
    ):
        raise ValueError("wet-lab audit requires typed target, registration, and result")
    candidate = result.candidate
    reasons: set[AuditReason] = {
        AuditReason.SEAL_CHRONOLOGY_AUTHORITY_UNVERIFIED,
        AuditReason.ANALYSIS_EXTRACTION_UNVERIFIED,
        AuditReason.REPLICATE_INDEPENDENCE_UNVERIFIED,
        AuditReason.INTERVAL_CALIBRATION_PROVENANCE_UNVERIFIED,
        AuditReason.REQUIRED_CONDITION_SCHEMA_UNAVAILABLE_OR_UNKNOWN,
    }
    if (
        registration.target_sha256 != target.target_sha256
        or registration.case_sha256 != target.case_sha256
        or registration.prediction_sha256 != target.prediction_sha256
    ):
        reasons.add(AuditReason.TARGET_IDENTITY_MISMATCH)
    if candidate.registration_sha256 != registration.registration_sha256:
        reasons.add(AuditReason.REGISTRATION_BINDING_MISMATCH)
    if candidate.protocol_sha256 != registration.protocol_sha256:
        reasons.add(AuditReason.PROTOCOL_MISMATCH)
    if candidate.conditions_sha256 != registration.conditions_sha256:
        reasons.add(AuditReason.CONDITIONS_MISMATCH)
    if (
        candidate.observable is not registration.observable
        or candidate.observable is not target.observable
    ):
        reasons.add(AuditReason.OBSERVABLE_MISMATCH)
    if (
        candidate.evidence_layer is not registration.evidence_layer
        or candidate.evidence_layer is not target.evidence_layer
    ):
        reasons.add(AuditReason.EVIDENCE_LAYER_MISMATCH)
    if candidate.unit != registration.unit or candidate.unit != target.unit:
        reasons.add(AuditReason.UNIT_MISMATCH)
    if candidate.independent_replicates < registration.minimum_independent_replicates:
        reasons.add(AuditReason.INSUFFICIENT_INDEPENDENT_REPLICATES)
    prediction_interval = target.interval
    measurement_interval = candidate.interval
    if prediction_interval is None:
        reasons.add(AuditReason.PREDICTION_INTERVAL_UNAVAILABLE)
    elif (
        prediction_interval.calibration_status is not IntervalCalibrationStatus.DECLARED_CALIBRATED
    ):
        reasons.add(AuditReason.PREDICTION_INTERVAL_UNCALIBRATED)
    if measurement_interval is None:
        reasons.add(AuditReason.MEASUREMENT_INTERVAL_UNAVAILABLE)
    elif (
        measurement_interval.calibration_status is not IntervalCalibrationStatus.DECLARED_CALIBRATED
    ):
        reasons.add(AuditReason.MEASUREMENT_INTERVAL_UNCALIBRATED)
    if (
        prediction_interval is not None
        and measurement_interval is not None
        and not isclose(
            prediction_interval.coverage_probability,
            measurement_interval.coverage_probability,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        reasons.add(AuditReason.COVERAGE_MISMATCH)

    return WetLabAuditReport(
        target_sha256=target.target_sha256,
        registration_sha256=registration.registration_sha256,
        result_sha256=result.result_sha256,
        prediction_value=target.value,
        observed_value=candidate.value,
        unit=target.unit,
        model_family=target.model_family,
        model_id=target.model_id,
        calibration_tier=target.calibration_tier,
        disposition=AuditDisposition.INSUFFICIENT_EVIDENCE,
        reasons=tuple(sorted(reasons, key=str)),
        prediction_interval=prediction_interval,
        measurement_interval=measurement_interval,
        interval_overlap=None,
        _construction_key=_REPORT_TOKEN,
    )


def wet_lab_audit_report_document(report: WetLabAuditReport) -> dict[str, object]:
    """Return a canonical JSON-compatible audit report."""

    if not isinstance(report, WetLabAuditReport):
        raise ValueError("audit serialization requires a WetLabAuditReport")
    document = json.loads(
        _canonical_bytes({**asdict(report), "report_sha256": report.report_sha256})
    )
    if not isinstance(document, dict):  # pragma: no cover - dataclass serialization is an object
        raise AssertionError("audit report did not serialize to an object")
    return document
