from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace

import pytest

from paircraft.corpus import (
    CorpusEntry,
    CorpusManifest,
    CorpusRole,
    DocumentedPermission,
    EligibilityStatus,
    EvidenceKind,
    EvidenceLayer,
    LeakageGroup,
    LeakageGroupKind,
    OutcomeAccess,
    RightsEvidence,
    RightsProfile,
    RoleEligibility,
    corpus_manifest_document,
    corpus_manifest_json,
    measurement_corpus_entry,
    simulation_corpus_entry,
)
from paircraft.experiments import (
    DeclaredMagnetSchedule,
    MagnetControlState,
    MeasuredDatum,
    PublicationStatus,
    ReportedFieldCondition,
    SourceBundle,
    SourceMember,
    SourceMemberRole,
    SourceReportedCondition,
    SourceReportStatus,
    declared_schedule_states,
)
from paircraft.oracle import (
    Maeda2012SimulatorOracle,
    OraclePrediction,
    SimulationDatum,
    maeda_2012_query,
)
from paircraft.spin import MaedaProteinSystem


def _decision(
    role: CorpusRole,
    status: EligibilityStatus,
    reason: str | None = None,
) -> RoleEligibility:
    reasons = () if reason is None else (reason,)
    return RoleEligibility(role, status, reasons, f"Decision for {role.value}.")


def _reference_eligibility(*, simulation: bool) -> tuple[RoleEligibility, ...]:
    return (
        _decision(CorpusRole.REFERENCE_ONLY, EligibilityStatus.ELIGIBLE),
        _decision(
            CorpusRole.SIMULATOR_SURROGATE_TRAINING,
            EligibilityStatus.UNRESOLVED if simulation else EligibilityStatus.INELIGIBLE,
            "rights-or-origin-not-admissible",
        ),
        _decision(
            CorpusRole.PHYSICAL_ORACLE_CALIBRATION,
            EligibilityStatus.INELIGIBLE,
            "not-calibration-evidence",
        ),
        _decision(
            CorpusRole.PROSPECTIVE_ORACLE_EVALUATION,
            EligibilityStatus.INELIGIBLE,
            "outcome-already-known",
        ),
    )


def _york_rights(datum: MeasuredDatum | None = None) -> RightsProfile:
    measurement = _measurement_datum() if datum is None else datum
    license_member = measurement.provenance.member(SourceMemberRole.LICENSE)
    permitted = DocumentedPermission.LICENSE_PERMITS
    return RightsProfile(
        license_expression="CC-BY-4.0",
        commercial_computational_use=permitted,
        datum_redistribution=permitted,
        source_payload_redistribution=permitted,
        attribution="Andrew G. York et al.",
        evidence=(
            RightsEvidence(
                license_member.exact_location,
                license_member.sha256,
                "The authenticated archive member declares CC BY 4.0.",
            ),
        ),
        rationale="The exact license member permits reuse with attribution.",
    )


def _unresolved_rights() -> RightsProfile:
    unknown = DocumentedPermission.NOT_ESTABLISHED
    return RightsProfile(
        license_expression="NOASSERTION",
        commercial_computational_use=unknown,
        datum_redistribution=unknown,
        source_payload_redistribution=unknown,
        attribution="Source lineage retained in the referenced datum.",
        evidence=(),
        rationale="Commercial computational use and redistribution were not established.",
    )


def _measurement_datum(trace_id: str = "trace-a") -> MeasuredDatum:
    member_hashes = {role: f"{index + 1:x}" * 64 for index, role in enumerate(SourceMemberRole)}
    source = SourceBundle(
        source_id="fixture-source",
        source_url="https://example.test/source",
        source_version="fixture-v1",
        archive_sha256="a" * 64,
        members=tuple(
            SourceMember(role, f"fixture/{role.value}", member_hashes[role])
            for role in SourceMemberRole
        ),
        license="CC BY 4.0",
        attribution="Fixture attribution",
        publication_status=PublicationStatus.PREPRINT_NON_PEER_REVIEWED,
    )
    times = (0.0, 4.0)
    off = (3.66,)
    on = (8.69,)
    schedule = DeclaredMagnetSchedule(
        initial_state=MagnetControlState.FIELD_ON,
        field_off_transition_seconds=off,
        field_on_transition_seconds=on,
        state_by_observation=declared_schedule_states(times, off, on),
        source_member_sha256=member_hashes[SourceMemberRole.SCHEDULE_SCRIPT],
    )
    context = source.member(SourceMemberRole.CONTEXT_HTML).exact_location
    return MeasuredDatum(
        trace_id=trace_id,
        sample_label="fixture sample",
        construct_id="fixture construct",
        frame_index=(1, 2),
        time_seconds=times,
        signal=(10.0, 11.0),
        signal_unit="count_per_pixel",
        schedule=schedule,
        reported_fields=(
            ReportedFieldCondition(
                MagnetControlState.FIELD_ON,
                "approximately 10 mT on",
                0.010,
                context,
            ),
            ReportedFieldCondition(
                MagnetControlState.FIELD_OFF,
                "off magnitude unreported",
                None,
                context,
            ),
        ),
        conditions=(
            SourceReportedCondition(
                "measurement_system",
                "fixture camera",
                "description",
                SourceReportStatus.SOURCE_REPORTED,
                context,
            ),
        ),
        provenance=source,
        limitations=("Fixture carries no replicate identity or pointwise uncertainty.",),
    )


@pytest.fixture(scope="module")
def simulation_datum() -> SimulationDatum:
    result = Maeda2012SimulatorOracle().predict(
        maeda_2012_query(
            MaedaProteinSystem.ATCRY1,
            (0.0, 0.0, 1e-3),
            query_id="corpus-manifest-fixture",
        )
    )
    assert isinstance(result, OraclePrediction)
    assert result.execution is not None
    return SimulationDatum(result, result.execution)


def _measurement_entry(
    datum: MeasuredDatum | None = None,
    *,
    entry_id: str = "york-reference",
) -> CorpusEntry:
    measurement = _measurement_datum() if datum is None else datum
    return measurement_corpus_entry(
        measurement,
        entry_id=entry_id,
        assigned_role=CorpusRole.REFERENCE_ONLY,
        eligibility=_reference_eligibility(simulation=False),
        rights=_york_rights(measurement),
        outcome_access=OutcomeAccess.KNOWN_BEFORE_MANIFEST,
        assignment_rationale=(
            "Inventory-only evidence; no model loss or benchmark use is authorized."
        ),
        exclusion_reasons=("missing-replicate-aware-calibration-evidence",),
        claim_boundary="A processed fluorescence trace, not a spin-yield target.",
    )


def _simulation_entry(datum: SimulationDatum) -> CorpusEntry:
    return simulation_corpus_entry(
        datum,
        entry_id="maeda-reference",
        assigned_role=CorpusRole.REFERENCE_ONLY,
        eligibility=_reference_eligibility(simulation=True),
        rights=_unresolved_rights(),
        outcome_access=OutcomeAccess.KNOWN_BEFORE_MANIFEST,
        assignment_rationale="Engine-conformance reference only.",
        exclusion_reasons=("simulator-output-is-not-independent-truth",),
        claim_boundary="A deterministic spin-layer calculation, not experimental evidence.",
    )


def test_factories_reference_exact_data_without_embedding_payloads(
    simulation_datum: SimulationDatum,
) -> None:
    measurement = _measurement_datum()
    measurement_membership = measurement.membership
    simulation_membership = simulation_datum.membership
    measured_entry = _measurement_entry(measurement)
    simulated_entry = _simulation_entry(simulation_datum)

    assert measured_entry.datum_sha256 == measurement.datum_sha256
    assert simulated_entry.datum_sha256 == simulation_datum.datum_sha256
    assert {group.kind for group in measured_entry.leakage_groups} == {
        LeakageGroupKind.EXACT_DATUM_DEPENDENCY,
        LeakageGroupKind.MEASUREMENT_SOURCE_BUNDLE,
    }
    assert {group.kind for group in simulated_entry.leakage_groups} == {
        LeakageGroupKind.EXACT_DATUM_DEPENDENCY,
        LeakageGroupKind.SIMULATION_PARAMETER_SET,
    }
    encoded = json.dumps([measured_entry, simulated_entry], default=str)
    assert "10.0" not in encoded
    assert "prediction" not in encoded
    assert measurement.membership is measurement_membership
    assert simulation_datum.membership is simulation_membership


def test_manifest_json_is_canonical_and_input_order_independent(
    simulation_datum: SimulationDatum,
) -> None:
    measured = _measurement_entry()
    simulated = _simulation_entry(simulation_datum)
    first = CorpusManifest(
        "reference-inventory", "Inventory current blockers.", (measured, simulated)
    )
    second = CorpusManifest(
        "reference-inventory", "Inventory current blockers.", (simulated, measured)
    )

    assert first.entries == second.entries
    assert corpus_manifest_json(first) == corpus_manifest_json(second)
    assert corpus_manifest_document(first)["hashes"]["manifest_sha256"] == first.manifest_sha256  # type: ignore[index]
    assert {entry.assigned_role for entry in first.entries} == {CorpusRole.REFERENCE_ONLY}


def test_manifest_hash_changes_with_a_policy_decision() -> None:
    entry = _measurement_entry()
    baseline = CorpusManifest("reference-inventory", "Inventory current blockers.", (entry,))
    changed_entry = replace(entry, claim_boundary="Changed claim boundary.")
    changed = CorpusManifest("reference-inventory", "Inventory current blockers.", (changed_entry,))

    assert baseline.records_sha256 != changed.records_sha256
    assert baseline.manifest_sha256 != changed.manifest_sha256


def test_entry_rejects_every_nonreference_assignment_in_v1() -> None:
    entry = _measurement_entry()
    with pytest.raises(ValueError, match="assigned corpus role must be eligible"):
        replace(entry, assigned_role=CorpusRole.PHYSICAL_ORACLE_CALIBRATION)
    calibration_eligible = tuple(
        replace(decision, status=EligibilityStatus.ELIGIBLE, reason_codes=())
        if decision.role is CorpusRole.PHYSICAL_ORACLE_CALIBRATION
        else decision
        for decision in entry.eligibility
    )
    with pytest.raises(ValueError, match="v1 permits reference-only"):
        replace(
            entry,
            assigned_role=CorpusRole.PHYSICAL_ORACLE_CALIBRATION,
            eligibility=calibration_eligible,
        )
    for role in CorpusRole:
        if role is CorpusRole.REFERENCE_ONLY:
            continue
        eligible = tuple(
            replace(decision, status=EligibilityStatus.ELIGIBLE, reason_codes=())
            if decision.role is role
            else decision
            for decision in entry.eligibility
        )
        with pytest.raises(ValueError, match="ineligible or unresolved in v1"):
            replace(entry, eligibility=eligible)


def test_manifest_rejects_duplicate_data() -> None:
    first = _measurement_entry()
    with pytest.raises(ValueError, match="datum may appear only once"):
        CorpusManifest(
            "duplicates", "Reject duplicate data.", (first, replace(first, entry_id="b"))
        )

    uppercase = replace(first, datum_sha256=first.datum_sha256.upper(), entry_id="uppercase")
    assert uppercase.datum_sha256 == first.datum_sha256
    with pytest.raises(ValueError, match="datum may appear only once"):
        CorpusManifest("case", "Reject case-bypassed duplicates.", (first, uppercase))


def test_low_level_contracts_fail_closed() -> None:
    entry = _measurement_entry()
    with pytest.raises(ValueError, match="immutable tuple"):
        replace(entry, leakage_groups=list(entry.leakage_groups))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="every corpus role"):
        replace(entry, eligibility=entry.eligibility[:-1])
    with pytest.raises(ValueError, match="exclusion reasons"):
        replace(entry, exclusion_reasons=())
    with pytest.raises(ValueError, match="64-character"):
        replace(entry.leakage_groups[0], sha256="bad")
    assert (
        replace(entry.leakage_groups[0], sha256=entry.leakage_groups[0].sha256.upper()).sha256
        == entry.leakage_groups[0].sha256
    )
    with pytest.raises(ValueError, match="exact leakage-group kinds"):
        replace(entry, leakage_groups=entry.leakage_groups[:1])
    wrong_group = LeakageGroup(LeakageGroupKind.SIMULATION_PARAMETER_SET, "f" * 64)
    with pytest.raises(ValueError, match="exact leakage-group kinds"):
        replace(entry, leakage_groups=(entry.leakage_groups[0], wrong_group))
    duplicate_kind = replace(entry.leakage_groups[0], sha256="e" * 64)
    with pytest.raises(ValueError, match="exact leakage-group kinds"):
        replace(entry, leakage_groups=(*entry.leakage_groups, duplicate_kind))
    with pytest.raises(ValueError, match="immutable tuple"):
        CorpusManifest("bad", "Bad tuple.", [entry])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reason code"):
        RoleEligibility(
            CorpusRole.PHYSICAL_ORACLE_CALIBRATION,
            EligibilityStatus.INELIGIBLE,
            (),
            "Missing reason code.",
        )
    with pytest.raises(ValueError, match="permissions"):
        replace(_york_rights(), datum_redistribution="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="supporting evidence"):
        replace(_york_rights(), evidence=())
    with pytest.raises(ValueError, match="immutable tuple"):
        replace(_york_rights(), evidence=[])  # type: ignore[arg-type]
    evidence = _york_rights().evidence[0]
    with pytest.raises(ValueError, match="unique"):
        replace(_york_rights(), evidence=(evidence, evidence))


def test_typed_policy_boundaries_reject_ambiguous_values() -> None:
    entry = _measurement_entry()
    decision = entry.eligibility[0]
    group = entry.leakage_groups[0]
    invalid: tuple[Callable[[], object], ...] = (
        lambda: replace(decision, role="reference_only"),  # type: ignore[arg-type]
        lambda: replace(decision, status="eligible"),  # type: ignore[arg-type]
        lambda: replace(decision, reason_codes=[]),  # type: ignore[arg-type]
        lambda: replace(decision, reason_codes=(" ",)),
        lambda: replace(decision, reason_codes=("duplicate", "duplicate")),
        lambda: replace(decision, rationale=" "),
        lambda: replace(_york_rights(), license_expression=" "),
        lambda: replace(group, kind="dependency"),  # type: ignore[arg-type]
        lambda: replace(entry, evidence_kind="measurement"),  # type: ignore[arg-type]
        lambda: replace(entry, evidence_layer="fluorescence"),  # type: ignore[arg-type]
        lambda: replace(entry, assigned_role="reference"),  # type: ignore[arg-type]
        lambda: replace(entry, outcome_access="known"),  # type: ignore[arg-type]
        lambda: replace(entry, rights="CC-BY-4.0"),  # type: ignore[arg-type]
        lambda: replace(entry, leakage_groups=()),
        lambda: replace(entry, leakage_groups=(group, group)),
        lambda: replace(entry, eligibility=("bad",) * 4),  # type: ignore[arg-type]
        lambda: replace(entry, evidence_layer=EvidenceLayer.SPIN_DYNAMICS_OUTPUT),
        lambda: replace(entry, exclusion_reasons=(" ",)),
        lambda: replace(entry, exclusion_reasons=("duplicate", "duplicate")),
        lambda: CorpusManifest("bad", "Bad schema.", (entry,), schema_version="v2"),
        lambda: CorpusManifest("bad", "No entries.", ()),
        lambda: CorpusManifest("bad", "Bad entry.", ("bad",)),  # type: ignore[arg-type]
        lambda: CorpusManifest("bad", "Duplicate IDs.", (entry, _measurement_entry())),
    )
    for constructor in invalid:
        with pytest.raises(ValueError):
            constructor()

    assert entry.evidence_kind is EvidenceKind.MEASUREMENT
    assert isinstance(group, LeakageGroup)


def test_simulation_cannot_claim_physical_calibration(
    simulation_datum: SimulationDatum,
) -> None:
    entry = _simulation_entry(simulation_datum)
    calibration_eligible = tuple(
        replace(decision, status=EligibilityStatus.ELIGIBLE, reason_codes=())
        if decision.role is CorpusRole.PHYSICAL_ORACLE_CALIBRATION
        else decision
        for decision in entry.eligibility
    )
    with pytest.raises(ValueError, match="ineligible or unresolved in v1"):
        replace(entry, eligibility=calibration_eligible)
    with pytest.raises(ValueError, match="spin-dynamics output"):
        replace(entry, evidence_layer=EvidenceLayer.FLUORESCENCE_READOUT)
    with pytest.raises(ValueError, match="rights must remain not established"):
        simulation_corpus_entry(
            simulation_datum,
            entry_id="bad-rights",
            assigned_role=CorpusRole.REFERENCE_ONLY,
            eligibility=_reference_eligibility(simulation=True),
            rights=_york_rights(),
            outcome_access=OutcomeAccess.KNOWN_BEFORE_MANIFEST,
            assignment_rationale="Invalid rights inference.",
            exclusion_reasons=("invalid",),
            claim_boundary="Invalid rights inference.",
        )


def test_measurement_permissions_must_match_the_exact_license_member() -> None:
    datum = _measurement_datum()
    rights = _york_rights(datum)
    mismatched = replace(
        rights,
        evidence=(replace(rights.evidence[0], sha256="f" * 64),),
    )
    with pytest.raises(ValueError, match="matching the datum license member"):
        measurement_corpus_entry(
            datum,
            entry_id="bad-rights",
            assigned_role=CorpusRole.REFERENCE_ONLY,
            eligibility=_reference_eligibility(simulation=False),
            rights=mismatched,
            outcome_access=OutcomeAccess.KNOWN_BEFORE_MANIFEST,
            assignment_rationale="Invalid evidence binding.",
            exclusion_reasons=("invalid",),
            claim_boundary="Invalid evidence binding.",
        )


def test_entry_requires_the_correct_live_datum_type() -> None:
    with pytest.raises(ValueError, match="SimulationDatum"):
        simulation_corpus_entry(
            "not-a-datum",  # type: ignore[arg-type]
            entry_id="bad",
            assigned_role=CorpusRole.REFERENCE_ONLY,
            eligibility=_reference_eligibility(simulation=True),
            rights=_unresolved_rights(),
            outcome_access=OutcomeAccess.KNOWN_BEFORE_MANIFEST,
            assignment_rationale="Invalid fixture.",
            exclusion_reasons=("invalid",),
            claim_boundary="Invalid fixture.",
        )
    with pytest.raises(ValueError, match="MeasuredDatum"):
        measurement_corpus_entry(
            "not-a-datum",  # type: ignore[arg-type]
            entry_id="bad",
            assigned_role=CorpusRole.REFERENCE_ONLY,
            eligibility=_reference_eligibility(simulation=False),
            rights=_york_rights(),
            outcome_access=OutcomeAccess.KNOWN_BEFORE_MANIFEST,
            assignment_rationale="Invalid fixture.",
            exclusion_reasons=("invalid",),
            claim_boundary="Invalid fixture.",
        )
