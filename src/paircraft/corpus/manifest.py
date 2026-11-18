"""Fail-closed corpus assignments spanning simulation and measurement evidence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256
from string import hexdigits
from typing import Any, cast

from paircraft.experiments.measured import MeasuredDatum, SourceMemberRole
from paircraft.oracle.simulation import SimulationDatum

_SCHEMA_VERSION = "paircraft-corpus-manifest-v1"


def _require_text(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be explicit")


def _require_sha256(value: str, name: str) -> None:
    digest = value.lower()
    if len(digest) != 64 or any(character not in hexdigits for character in digest):
        raise ValueError(f"{name} must be a 64-character hexadecimal digest")


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return sha256(encoded).hexdigest()


class EvidenceKind(StrEnum):
    """Origin of an exact datum referenced by a corpus entry."""

    SIMULATION = "simulation"
    MEASUREMENT = "measurement"


class EvidenceLayer(StrEnum):
    """Physical layer represented by the datum outcome."""

    SPIN_DYNAMICS_OUTPUT = "spin_dynamics_output"
    FLUORESCENCE_READOUT = "fluorescence_readout"


class CorpusRole(StrEnum):
    """Authorized use of a datum inside one exact corpus manifest."""

    REFERENCE_ONLY = "reference_only"
    SIMULATOR_SURROGATE_TRAINING = "simulator_surrogate_training"
    PHYSICAL_ORACLE_CALIBRATION = "physical_oracle_calibration"
    PROSPECTIVE_ORACLE_EVALUATION = "prospective_oracle_evaluation"


class EligibilityStatus(StrEnum):
    """Decision about whether a datum may serve one corpus role."""

    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"
    UNRESOLVED = "unresolved"


class DocumentedPermission(StrEnum):
    """What the retained rights evidence establishes."""

    LICENSE_PERMITS = "license_permits"
    NOT_ESTABLISHED = "not_established"
    PROHIBITED = "prohibited"


class OutcomeAccess(StrEnum):
    """Whether an outcome was hidden before evaluation assignment."""

    KNOWN_BEFORE_MANIFEST = "known_before_manifest"
    SEALED_UNTIL_EVALUATION = "sealed_until_evaluation"


class LeakageGroupKind(StrEnum):
    """Dependency identity that must remain within one assigned corpus role."""

    EXACT_DATUM_DEPENDENCY = "exact_datum_dependency"
    MEASUREMENT_SOURCE_BUNDLE = "measurement_source_bundle"
    SIMULATION_PARAMETER_SET = "simulation_parameter_set"


@dataclass(frozen=True, slots=True)
class RoleEligibility:
    """One explicit eligibility decision for one possible corpus role."""

    role: CorpusRole
    status: EligibilityStatus
    reason_codes: tuple[str, ...]
    rationale: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, CorpusRole):
            raise ValueError("eligibility role must be a CorpusRole")
        if not isinstance(self.status, EligibilityStatus):
            raise ValueError("eligibility status must be an EligibilityStatus")
        if not isinstance(self.reason_codes, tuple):
            raise ValueError("eligibility reason codes must be an immutable tuple")
        if any(not isinstance(code, str) or not code.strip() for code in self.reason_codes):
            raise ValueError("eligibility reason codes must be explicit strings")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("eligibility reason codes must be unique")
        object.__setattr__(self, "reason_codes", tuple(sorted(self.reason_codes)))
        _require_text(self.rationale, "eligibility rationale")
        if self.status is not EligibilityStatus.ELIGIBLE and not self.reason_codes:
            raise ValueError("ineligible or unresolved roles require a reason code")


@dataclass(frozen=True, slots=True)
class RightsEvidence:
    """One exact artifact supporting a documented rights assertion."""

    exact_location: str
    sha256: str
    statement: str

    def __post_init__(self) -> None:
        _require_text(self.exact_location, "rights evidence location")
        _require_sha256(self.sha256, "rights evidence sha256")
        _require_text(self.statement, "rights evidence statement")
        object.__setattr__(self, "sha256", self.sha256.lower())


@dataclass(frozen=True, slots=True)
class RightsProfile:
    """Documented permissions, kept separate from scientific eligibility."""

    license_expression: str
    commercial_computational_use: DocumentedPermission
    datum_redistribution: DocumentedPermission
    source_payload_redistribution: DocumentedPermission
    attribution: str
    evidence: tuple[RightsEvidence, ...]
    rationale: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.license_expression, "license expression"),
            (self.attribution, "rights attribution"),
            (self.rationale, "rights rationale"),
        ):
            _require_text(value, name)
        permissions = (
            self.commercial_computational_use,
            self.datum_redistribution,
            self.source_payload_redistribution,
        )
        if any(not isinstance(value, DocumentedPermission) for value in permissions):
            raise ValueError("rights permissions must be DocumentedPermission values")
        if not isinstance(self.evidence, tuple) or any(
            not isinstance(item, RightsEvidence) for item in self.evidence
        ):
            raise ValueError("rights evidence must be an immutable tuple of RightsEvidence")
        evidence_keys = [(item.exact_location, item.sha256) for item in self.evidence]
        if len(evidence_keys) != len(set(evidence_keys)):
            raise ValueError("rights evidence artifacts must be unique")
        object.__setattr__(
            self,
            "evidence",
            tuple(sorted(self.evidence, key=lambda item: (item.exact_location, item.sha256))),
        )
        if DocumentedPermission.LICENSE_PERMITS in permissions and not self.evidence:
            raise ValueError("permitted rights require exact supporting evidence")


@dataclass(frozen=True, slots=True)
class LeakageGroup:
    """One content-addressed dependency boundary used to prevent split leakage."""

    kind: LeakageGroupKind
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, LeakageGroupKind):
            raise ValueError("leakage group kind must be a LeakageGroupKind")
        _require_sha256(self.sha256, "leakage group sha256")
        object.__setattr__(self, "sha256", self.sha256.lower())


@dataclass(frozen=True, slots=True)
class CorpusEntry:
    """A small policy reference to one complete, separately stored datum."""

    entry_id: str
    evidence_kind: EvidenceKind
    evidence_layer: EvidenceLayer
    observable: str
    datum_schema_version: str
    datum_sha256: str
    leakage_groups: tuple[LeakageGroup, ...]
    assigned_role: CorpusRole
    eligibility: tuple[RoleEligibility, ...]
    rights: RightsProfile
    outcome_access: OutcomeAccess
    assignment_rationale: str
    exclusion_reasons: tuple[str, ...]
    claim_boundary: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.entry_id, "corpus entry ID"),
            (self.observable, "corpus entry observable"),
            (self.datum_schema_version, "datum schema version"),
            (self.assignment_rationale, "assignment rationale"),
            (self.claim_boundary, "claim boundary"),
        ):
            _require_text(value, name)
        _require_sha256(self.datum_sha256, "datum sha256")
        object.__setattr__(self, "datum_sha256", self.datum_sha256.lower())
        if not isinstance(self.evidence_kind, EvidenceKind):
            raise ValueError("evidence kind must be an EvidenceKind")
        if not isinstance(self.evidence_layer, EvidenceLayer):
            raise ValueError("evidence layer must be an EvidenceLayer")
        if not isinstance(self.assigned_role, CorpusRole):
            raise ValueError("assigned role must be a CorpusRole")
        if not isinstance(self.outcome_access, OutcomeAccess):
            raise ValueError("outcome access must be an OutcomeAccess")
        if not isinstance(self.rights, RightsProfile):
            raise ValueError("rights must be a RightsProfile")
        for tuple_value, name in (
            (self.leakage_groups, "leakage groups"),
            (self.eligibility, "eligibility decisions"),
            (self.exclusion_reasons, "exclusion reasons"),
        ):
            if not isinstance(tuple_value, tuple):
                raise ValueError(f"{name} must be an immutable tuple")
        if not self.leakage_groups or any(
            not isinstance(group, LeakageGroup) for group in self.leakage_groups
        ):
            raise ValueError("an entry requires typed leakage groups")
        canonical_groups = tuple(
            sorted(self.leakage_groups, key=lambda item: (item.kind, item.sha256))
        )
        if len(canonical_groups) != len({(group.kind, group.sha256) for group in canonical_groups}):
            raise ValueError("leakage groups must be unique")
        object.__setattr__(self, "leakage_groups", canonical_groups)
        expected_group_kinds = (
            {
                LeakageGroupKind.EXACT_DATUM_DEPENDENCY,
                LeakageGroupKind.SIMULATION_PARAMETER_SET,
            }
            if self.evidence_kind is EvidenceKind.SIMULATION
            else {
                LeakageGroupKind.EXACT_DATUM_DEPENDENCY,
                LeakageGroupKind.MEASUREMENT_SOURCE_BUNDLE,
            }
        )
        if (
            len(canonical_groups) != len(expected_group_kinds)
            or {group.kind for group in canonical_groups} != expected_group_kinds
        ):
            raise ValueError("entry requires the exact leakage-group kinds for its evidence origin")
        if any(not isinstance(item, RoleEligibility) for item in self.eligibility):
            raise ValueError("eligibility decisions must be RoleEligibility values")
        by_role = {item.role: item for item in self.eligibility}
        if len(self.eligibility) != len(CorpusRole) or set(by_role) != set(CorpusRole):
            raise ValueError("eligibility must contain exactly one decision for every corpus role")
        canonical_eligibility = tuple(by_role[role] for role in CorpusRole)
        object.__setattr__(self, "eligibility", canonical_eligibility)
        assigned = by_role[self.assigned_role]
        if assigned.status is not EligibilityStatus.ELIGIBLE:
            raise ValueError("the assigned corpus role must be eligible")
        if self.assigned_role is not CorpusRole.REFERENCE_ONLY:
            raise ValueError("corpus manifest v1 permits reference-only assignment")
        if any(
            by_role[role].status is EligibilityStatus.ELIGIBLE
            for role in CorpusRole
            if role is not CorpusRole.REFERENCE_ONLY
        ):
            raise ValueError("non-reference roles must remain ineligible or unresolved in v1")
        if self.evidence_kind is EvidenceKind.SIMULATION:
            if self.evidence_layer is not EvidenceLayer.SPIN_DYNAMICS_OUTPUT:
                raise ValueError("simulation evidence must remain a spin-dynamics output")
        else:
            if self.evidence_layer is not EvidenceLayer.FLUORESCENCE_READOUT:
                raise ValueError("measurement evidence must remain a fluorescence readout")
        if self.assigned_role is CorpusRole.REFERENCE_ONLY and not self.exclusion_reasons:
            raise ValueError("reference-only evidence requires explicit exclusion reasons")
        if any(
            not isinstance(reason, str) or not reason.strip() for reason in self.exclusion_reasons
        ):
            raise ValueError("exclusion reasons must be explicit strings")
        if len(self.exclusion_reasons) != len(set(self.exclusion_reasons)):
            raise ValueError("exclusion reasons must be unique")
        object.__setattr__(self, "exclusion_reasons", tuple(sorted(self.exclusion_reasons)))


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    """One deterministic assignment of exact data to scientific uses."""

    manifest_id: str
    purpose: str
    entries: tuple[CorpusEntry, ...]
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.manifest_id, "manifest ID")
        _require_text(self.purpose, "manifest purpose")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("corpus manifest schema version is unsupported")
        if not isinstance(self.entries, tuple):
            raise ValueError("manifest entries must be an immutable tuple")
        if not self.entries or any(not isinstance(entry, CorpusEntry) for entry in self.entries):
            raise ValueError("a manifest requires typed corpus entries")
        entries = tuple(sorted(self.entries, key=lambda item: item.entry_id))
        object.__setattr__(self, "entries", entries)
        entry_ids = [entry.entry_id for entry in entries]
        datum_hashes = [entry.datum_sha256 for entry in entries]
        if len(entry_ids) != len(set(entry_ids)):
            raise ValueError("manifest entry IDs must be unique")
        if len(datum_hashes) != len(set(datum_hashes)):
            raise ValueError("each datum may appear only once in a manifest")
        roles_by_group: dict[tuple[LeakageGroupKind, str], CorpusRole] = {}
        for entry in entries:
            for group in entry.leakage_groups:
                key = (group.kind, group.sha256)
                previous = roles_by_group.setdefault(key, entry.assigned_role)
                if previous is not entry.assigned_role:
                    raise ValueError("one leakage group cannot cross assigned corpus roles")

    @property
    def records_sha256(self) -> str:
        """Hash the canonical ordered entry references and policy decisions."""

        return _canonical_sha256([asdict(entry) for entry in self.entries])

    @property
    def manifest_sha256(self) -> str:
        """Hash the complete manifest identity and canonical records hash."""

        return _canonical_sha256(
            {
                "schema_version": self.schema_version,
                "manifest_id": self.manifest_id,
                "purpose": self.purpose,
                "records_sha256": self.records_sha256,
                "entries": [asdict(entry) for entry in self.entries],
            }
        )


def simulation_corpus_entry(
    datum: SimulationDatum,
    *,
    entry_id: str,
    assigned_role: CorpusRole,
    eligibility: tuple[RoleEligibility, ...],
    rights: RightsProfile,
    outcome_access: OutcomeAccess,
    assignment_rationale: str,
    exclusion_reasons: tuple[str, ...],
    claim_boundary: str,
) -> CorpusEntry:
    """Reference one validated simulation datum without embedding its numerical payload."""

    if not isinstance(datum, SimulationDatum):
        raise ValueError("simulation corpus entries require a SimulationDatum")
    if not isinstance(rights, RightsProfile):
        raise ValueError("simulation corpus entries require a RightsProfile")
    simulation_permissions = (
        rights.commercial_computational_use,
        rights.datum_redistribution,
        rights.source_payload_redistribution,
    )
    if any(
        permission is not DocumentedPermission.NOT_ESTABLISHED
        for permission in simulation_permissions
    ):
        raise ValueError("simulation rights must remain not established in corpus manifest v1")
    system = datum.prediction.query.system
    parameter_group_sha256 = _canonical_sha256(
        {
            "system_id": system.system_id,
            "parameter_set_id": system.parameter_set_id,
            "parameter_set_sha256": system.parameter_set_sha256,
            "parameter_catalog_version": system.parameter_catalog_version,
            "parameter_catalog_sha256": system.parameter_catalog_sha256,
            "model_family": datum.prediction.query.model_family,
        }
    )
    return CorpusEntry(
        entry_id=entry_id,
        evidence_kind=EvidenceKind.SIMULATION,
        evidence_layer=EvidenceLayer.SPIN_DYNAMICS_OUTPUT,
        observable=str(datum.prediction.query.observable),
        datum_schema_version=datum.schema_version,
        datum_sha256=datum.datum_sha256,
        leakage_groups=(
            LeakageGroup(LeakageGroupKind.EXACT_DATUM_DEPENDENCY, datum.dependency_group_sha256),
            LeakageGroup(LeakageGroupKind.SIMULATION_PARAMETER_SET, parameter_group_sha256),
        ),
        assigned_role=assigned_role,
        eligibility=eligibility,
        rights=rights,
        outcome_access=outcome_access,
        assignment_rationale=assignment_rationale,
        exclusion_reasons=exclusion_reasons,
        claim_boundary=claim_boundary,
    )


def measurement_corpus_entry(
    datum: MeasuredDatum,
    *,
    entry_id: str,
    assigned_role: CorpusRole,
    eligibility: tuple[RoleEligibility, ...],
    rights: RightsProfile,
    outcome_access: OutcomeAccess,
    assignment_rationale: str,
    exclusion_reasons: tuple[str, ...],
    claim_boundary: str,
) -> CorpusEntry:
    """Reference one validated measurement datum without embedding its trace payload."""

    if not isinstance(datum, MeasuredDatum):
        raise ValueError("measurement corpus entries require a MeasuredDatum")
    if not isinstance(rights, RightsProfile):
        raise ValueError("measurement corpus entries require a RightsProfile")
    measurement_permissions = (
        rights.commercial_computational_use,
        rights.datum_redistribution,
        rights.source_payload_redistribution,
    )
    if DocumentedPermission.LICENSE_PERMITS in measurement_permissions:
        license_member = datum.provenance.member(SourceMemberRole.LICENSE)
        if not any(
            item.exact_location == license_member.exact_location
            and item.sha256 == license_member.sha256.lower()
            for item in rights.evidence
        ):
            raise ValueError(
                "permitted measurement rights require evidence matching the datum license member"
            )
    return CorpusEntry(
        entry_id=entry_id,
        evidence_kind=EvidenceKind.MEASUREMENT,
        evidence_layer=EvidenceLayer.FLUORESCENCE_READOUT,
        observable=datum.observable,
        datum_schema_version=datum.schema_version,
        datum_sha256=datum.datum_sha256,
        leakage_groups=(
            LeakageGroup(LeakageGroupKind.EXACT_DATUM_DEPENDENCY, datum.dependency_group_sha256),
            LeakageGroup(LeakageGroupKind.MEASUREMENT_SOURCE_BUNDLE, datum.source_group_sha256),
        ),
        assigned_role=assigned_role,
        eligibility=eligibility,
        rights=rights,
        outcome_access=outcome_access,
        assignment_rationale=assignment_rationale,
        exclusion_reasons=exclusion_reasons,
        claim_boundary=claim_boundary,
    )


def corpus_manifest_document(manifest: CorpusManifest) -> dict[str, object]:
    """Return a complete JSON-compatible manifest with derived content hashes."""

    document: dict[str, Any] = {
        "schema_version": manifest.schema_version,
        "manifest_id": manifest.manifest_id,
        "purpose": manifest.purpose,
        "entries": [asdict(entry) for entry in manifest.entries],
        "hashes": {
            "records_sha256": manifest.records_sha256,
            "manifest_sha256": manifest.manifest_sha256,
        },
    }
    normalized = json.loads(
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    if not isinstance(normalized, dict):
        raise AssertionError("corpus manifest normalization did not produce an object")
    return cast(dict[str, object], normalized)


def corpus_manifest_json(manifest: CorpusManifest) -> bytes:
    """Return canonical compact JSON bytes; no persistence policy is implied."""

    return json.dumps(
        corpus_manifest_document(manifest),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
