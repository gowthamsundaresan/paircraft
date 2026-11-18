from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256

import pytest

import paircraft.audit.model as audit_model
from paircraft.audit import (
    AuditablePrediction,
    AuditDisposition,
    AuditInterval,
    AuditReason,
    ConditionEvidenceStatus,
    ContentAuthenticatedWetLabResult,
    IntervalCalibrationStatus,
    ObservedCondition,
    PhysicalEvidenceLayer,
    ProspectiveAuditRegistration,
    SourceArtifact,
    WetLabProvenance,
    WetLabResultCandidate,
    audit_wet_lab_result,
    content_authenticate_wet_lab_result,
    content_authenticated_wet_lab_result_document,
    create_prospective_audit_registration,
    wet_lab_audit_report_document,
)
from paircraft.oracle import CalibrationTier, ObservableKind, SpinModelFamily


def _digest(label: str) -> str:
    return sha256(label.encode()).hexdigest()


def _conditions() -> tuple[ObservedCondition, ...]:
    return (
        ObservedCondition(
            name="temperature",
            value=298.15,
            unit="K",
            status=ConditionEvidenceStatus.MEASURED,
            exact_source_location="result.json#/conditions/temperature",
            standard_uncertainty=0.1,
        ),
    )


def _target(interval: AuditInterval | None) -> AuditablePrediction:
    return AuditablePrediction(
        case_sha256=_digest("case"),
        query_sha256=_digest("query"),
        prediction_sha256=_digest("prediction"),
        execution_sha256=_digest("execution"),
        oracle_artifact_sha256=_digest("oracle"),
        model_family=SpinModelFamily.HABERKORN_STATIC_PAIR_WITH_ST_DEPHASING,
        model_id="auditable-model",
        calibration_tier=(
            CalibrationTier.SIMULATOR_ONLY
            if interval is None
            else CalibrationTier.PUBLIC_RETROSPECTIVE
        ),
        observable=ObservableKind.SECONDARY_RADICAL_YIELD,
        evidence_layer=PhysicalEvidenceLayer.SPIN_CHEMICAL_YIELD,
        value=0.45,
        unit="1",
        interval=interval,
        _construction_key=audit_model._PREDICTION_TOKEN,
    )


def _registered_result(
    target: AuditablePrediction,
    *,
    result_interval: AuditInterval | None,
    value: float = 0.46,
    protocol_sha256: str | None = None,
    conditions: tuple[ObservedCondition, ...] | None = None,
    evidence_layer: PhysicalEvidenceLayer = PhysicalEvidenceLayer.SPIN_CHEMICAL_YIELD,
    unit: str = "1",
    independent_replicates: int = 3,
) -> tuple[ProspectiveAuditRegistration, ContentAuthenticatedWetLabResult]:
    registered_conditions = _conditions()
    protocol = _digest("protocol")
    registration = create_prospective_audit_registration(
        target,
        protocol_sha256=protocol,
        conditions=registered_conditions,
        minimum_independent_replicates=3,
        seal_authority="external-registration-service",
        seal_receipt=b"sealed-before-outcome",
        seal_scope=(
            "Receipt byte identity only; seal chronology and authority are unverified in v1."
        ),
    )
    payload = b"exact wet-lab result payload"
    analysis_payload = b"exact analysis artifact"
    verifier_payload = b"exact content-verifier artifact"
    provenance = WetLabProvenance(
        result_id="result-1",
        acquisition_id="acquisition-1",
        source_uri="lab://result-1",
        source_artifacts=(
            SourceArtifact(
                role="scalar_result",
                exact_location="result.json",
                sha256=sha256(payload).hexdigest(),
            ),
        ),
        analysis_artifact_sha256=sha256(analysis_payload).hexdigest(),
        verifier_id="paircraft-content-verifier",
        verifier_version="1",
        verifier_artifact_sha256=sha256(verifier_payload).hexdigest(),
        authentication_scope=(
            "Exact artifact bytes and registration binding only; lab authorship, analysis or "
            "extraction validity, replicate independence, interval calibration, and condition "
            "completeness are unverified in v1."
        ),
    )
    candidate = WetLabResultCandidate(
        registration_sha256=registration.registration_sha256,
        protocol_sha256=protocol_sha256 or protocol,
        conditions=conditions or registered_conditions,
        observable=ObservableKind.SECONDARY_RADICAL_YIELD,
        evidence_layer=evidence_layer,
        value=value,
        unit=unit,
        interval=result_interval,
        independent_replicates=independent_replicates,
        dependency_group_sha256=_digest("acquisition-family"),
        provenance=provenance,
    )
    authenticated = content_authenticate_wet_lab_result(
        candidate,
        registration=registration,
        source_payloads_by_role={"scalar_result": payload},
        analysis_artifact_payload=analysis_payload,
        verifier_artifact_payload=verifier_payload,
    )
    return registration, authenticated


def _calibrated_interval(lower: float, upper: float) -> AuditInterval:
    return AuditInterval(
        lower=lower,
        upper=upper,
        coverage_probability=0.95,
        method="predeclared 95% interval",
        calibration_status=IntervalCalibrationStatus.DECLARED_CALIBRATED,
    )


def test_simulator_only_target_is_insufficient_even_at_exact_numeric_agreement() -> None:
    target = _target(None)
    registration, result = _registered_result(
        target,
        result_interval=_calibrated_interval(0.44, 0.46),
        value=target.value,
    )

    report = audit_wet_lab_result(target, registration, result)

    assert report.disposition is AuditDisposition.INSUFFICIENT_EVIDENCE
    assert AuditReason.PREDICTION_INTERVAL_UNAVAILABLE in report.reasons
    assert report.calibration_tier is CalibrationTier.SIMULATOR_ONLY
    assert report.interval_overlap is None


@pytest.mark.parametrize(
    "measurement_interval",
    [
        _calibrated_interval(0.44, 0.47),
        _calibrated_interval(0.60, 0.70),
    ],
)
def test_self_asserted_intervals_and_replicates_never_produce_match_or_drift(
    measurement_interval: AuditInterval,
) -> None:
    target = _target(_calibrated_interval(0.40, 0.50))
    value = (measurement_interval.lower + measurement_interval.upper) / 2
    registration, result = _registered_result(
        target,
        result_interval=measurement_interval,
        value=value,
    )

    report = audit_wet_lab_result(target, registration, result)

    assert report.disposition is AuditDisposition.INSUFFICIENT_EVIDENCE
    assert report.interval_overlap is None
    assert {
        AuditReason.SEAL_CHRONOLOGY_AUTHORITY_UNVERIFIED,
        AuditReason.ANALYSIS_EXTRACTION_UNVERIFIED,
        AuditReason.REPLICATE_INDEPENDENCE_UNVERIFIED,
        AuditReason.INTERVAL_CALIBRATION_PROVENANCE_UNVERIFIED,
        AuditReason.REQUIRED_CONDITION_SCHEMA_UNAVAILABLE_OR_UNKNOWN,
    }.issubset(report.reasons)


def test_layer_unit_protocol_conditions_replicates_and_interval_fail_closed() -> None:
    target = _target(_calibrated_interval(0.40, 0.50))
    other_conditions = (replace(_conditions()[0], value=301.0),)
    uncalibrated = replace(
        _calibrated_interval(0.43, 0.49),
        calibration_status=IntervalCalibrationStatus.DECLARED_UNCALIBRATED,
    )
    registration, result = _registered_result(
        target,
        result_interval=uncalibrated,
        protocol_sha256=_digest("different-protocol"),
        conditions=other_conditions,
        evidence_layer=PhysicalEvidenceLayer.FLUORESCENCE_READOUT,
        unit="count",
        independent_replicates=1,
    )

    report = audit_wet_lab_result(target, registration, result)

    assert report.disposition is AuditDisposition.INSUFFICIENT_EVIDENCE
    assert {
        AuditReason.PROTOCOL_MISMATCH,
        AuditReason.CONDITIONS_MISMATCH,
        AuditReason.EVIDENCE_LAYER_MISMATCH,
        AuditReason.UNIT_MISMATCH,
        AuditReason.INSUFFICIENT_INDEPENDENT_REPLICATES,
        AuditReason.MEASUREMENT_INTERVAL_UNCALIBRATED,
    }.issubset(report.reasons)


def test_authentication_rejects_tampered_bytes_and_constructor_forgery() -> None:
    target = _target(None)
    registration, result = _registered_result(target, result_interval=None)
    candidate = result.candidate

    with pytest.raises(ValueError, match="source payload bytes"):
        content_authenticate_wet_lab_result(
            candidate,
            registration=registration,
            source_payloads_by_role={"scalar_result": b"tampered"},
            analysis_artifact_payload=b"exact analysis artifact",
            verifier_artifact_payload=b"exact content-verifier artifact",
        )
    with pytest.raises(ValueError, match="analysis artifact bytes"):
        content_authenticate_wet_lab_result(
            candidate,
            registration=registration,
            source_payloads_by_role={"scalar_result": b"exact wet-lab result payload"},
            analysis_artifact_payload=b"tampered",
            verifier_artifact_payload=b"exact content-verifier artifact",
        )
    with pytest.raises(ValueError, match="verifier artifact bytes"):
        content_authenticate_wet_lab_result(
            candidate,
            registration=registration,
            source_payloads_by_role={"scalar_result": b"exact wet-lab result payload"},
            analysis_artifact_payload=b"exact analysis artifact",
            verifier_artifact_payload=b"tampered",
        )
    with pytest.raises(ValueError, match="evidence authenticator"):
        ContentAuthenticatedWetLabResult(
            candidate=candidate,
            authentication_receipt_sha256=_digest("forged"),
            _token=object(),
        )
    with pytest.raises(ValueError, match="exact trusted adapter"):
        replace(target, _construction_key=object())
    with pytest.raises(ValueError, match="content-binding factory"):
        replace(registration, _construction_key=object())


def test_report_serialization_preserves_provenance_identity_and_claim_boundary() -> None:
    target = _target(None)
    registration, result = _registered_result(target, result_interval=None)
    report = audit_wet_lab_result(target, registration, result)

    document = wet_lab_audit_report_document(report)
    result_document = content_authenticated_wet_lab_result_document(result)
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)

    assert document["report_sha256"] == report.report_sha256
    assert document["model_family"] == target.model_family
    assert document["model_id"] == target.model_id
    assert document["calibration_tier"] == CalibrationTier.SIMULATOR_ONLY
    assert "lab://result-1" in json.dumps(result_document, sort_keys=True)
    assert result_document["result_sha256"] == result.result_sha256
    assert "never changes an oracle's calibration tier" in raw
    with pytest.raises(ValueError, match="authenticated comparison"):
        replace(
            report,
            disposition=AuditDisposition.DRIFT_OUTSIDE_DECLARED_INTERVAL,
            _construction_key=object(),
        )
    with pytest.raises(ValueError, match="must remain insufficient"):
        replace(
            report,
            disposition=AuditDisposition.DRIFT_OUTSIDE_DECLARED_INTERVAL,
            _construction_key=audit_model._REPORT_TOKEN,
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"lower": float("nan")}, "finite real scalar"),
        ({"lower": 2.0, "upper": 1.0}, "bounds or coverage"),
        ({"coverage_probability": 1.0}, "bounds or coverage"),
        ({"method": ""}, "method must be explicit"),
        ({"calibration_status": "calibrated"}, "status must be explicit"),
    ],
)
def test_interval_refuses_ambiguous_or_invalid_uncertainty_claims(
    changes: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "lower": 0.4,
        "upper": 0.5,
        "coverage_probability": 0.95,
        "method": "declared interval",
        "calibration_status": IntervalCalibrationStatus.DECLARED_UNCALIBRATED,
    }
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        AuditInterval(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"status": "measured"}, "status must be explicit"),
        (
            {
                "status": ConditionEvidenceStatus.UNKNOWN,
                "value": 298.15,
                "standard_uncertainty": None,
            },
            "unknown condition",
        ),
        ({"value": None}, "known condition requires"),
        ({"value": True}, "known condition requires"),
        ({"value": object()}, "numeric or text"),
        ({"value": ""}, "value must be explicit"),
        ({"standard_uncertainty": -0.1}, "nonnegative numeric"),
        ({"value": "ambient", "standard_uncertainty": 0.1}, "nonnegative numeric"),
    ],
)
def test_condition_epistemic_state_refuses_contradictory_values(
    changes: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "name": "temperature",
        "value": 298.15,
        "unit": "K",
        "status": ConditionEvidenceStatus.MEASURED,
        "exact_source_location": "result.json#/temperature",
        "standard_uncertainty": 0.1,
    }
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        ObservedCondition(**values)  # type: ignore[arg-type]


def test_unknown_condition_is_represented_without_numeric_placeholder() -> None:
    condition = ObservedCondition(
        name="oxygen",
        value=None,
        unit="1",
        status=ConditionEvidenceStatus.UNKNOWN,
        exact_source_location="result.json#/conditions",
    )

    assert condition.value is None
    assert condition.standard_uncertainty is None


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"source_artifacts": ()}, "typed source artifacts"),
        (
            {
                "source_artifacts": (
                    SourceArtifact("z", "z.bin", _digest("z")),
                    SourceArtifact("a", "a.bin", _digest("a")),
                )
            },
            "unique and sorted",
        ),
        ({"authentication_scope": "scientifically verified"}, "content-only"),
        ({"analysis_artifact_sha256": "bad"}, "lowercase SHA-256"),
    ],
)
def test_wet_lab_provenance_refuses_ambiguous_content_identity(
    changes: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "result_id": "result",
        "acquisition_id": "acquisition",
        "source_uri": "lab://result",
        "source_artifacts": (SourceArtifact("scalar", "result.bin", _digest("source")),),
        "analysis_artifact_sha256": _digest("analysis"),
        "verifier_id": "verifier",
        "verifier_version": "1",
        "verifier_artifact_sha256": _digest("verifier"),
        "authentication_scope": audit_model._CONTENT_AUTHENTICATION_SCOPE,
    }
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        WetLabProvenance(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"model_family": "haberkorn"}, "model family"),
        ({"calibration_tier": "simulator"}, "calibration tier"),
        ({"observable": "yield"}, "observable"),
        ({"evidence_layer": PhysicalEvidenceLayer.FLUORESCENCE_READOUT}, "spin-yield layer"),
        ({"interval": "0.4-0.5"}, "interval must be typed"),
        ({"value": float("inf")}, "finite real scalar"),
    ],
)
def test_trusted_prediction_refuses_scientific_type_erasure(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(
            _target(None),
            **changes,  # type: ignore[arg-type]
            _construction_key=audit_model._PREDICTION_TOKEN,
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"observable": "yield"}, "scientific identity"),
        ({"minimum_independent_replicates": True}, "positive integer"),
        ({"minimum_independent_replicates": 0}, "positive integer"),
        ({"schema_version": "future"}, "schema is unsupported"),
        ({"seal_scope": "verified prospectively"}, "content-only"),
    ],
)
def test_registration_refuses_untyped_or_overstated_evidence(
    changes: dict[str, object], message: str
) -> None:
    target = _target(None)
    registration, _ = _registered_result(target, result_interval=None)

    with pytest.raises(ValueError, match=message):
        replace(
            registration,
            **changes,  # type: ignore[arg-type]
            _construction_key=audit_model._REGISTRATION_TOKEN,
        )


@pytest.mark.parametrize(
    ("conditions", "receipt", "message"),
    [([_conditions()[0]], b"receipt", "immutable typed tuple"), ((), b"", "nonempty")],
)
def test_registration_factory_refuses_mutable_conditions_or_empty_receipt(
    conditions: object, receipt: bytes, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        create_prospective_audit_registration(
            _target(None),
            protocol_sha256=_digest("protocol"),
            conditions=conditions,  # type: ignore[arg-type]
            minimum_independent_replicates=1,
            seal_authority="authority",
            seal_receipt=receipt,
            seal_scope=audit_model._CONTENT_SEAL_SCOPE,
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"conditions": []}, "immutable typed tuple"),
        ({"observable": "yield"}, "scientific identity"),
        ({"interval": "0.4-0.5"}, "interval must be typed"),
        ({"interval": _calibrated_interval(0.1, 0.2)}, "contain the reported value"),
        ({"independent_replicates": True}, "positive integer"),
        ({"independent_replicates": 0}, "positive integer"),
        ({"provenance": object()}, "provenance must be typed"),
    ],
)
def test_result_candidate_refuses_unusable_scientific_evidence(
    changes: dict[str, object], message: str
) -> None:
    _, result = _registered_result(_target(None), result_interval=None)

    with pytest.raises(ValueError, match=message):
        replace(result.candidate, **changes)  # type: ignore[arg-type]


def test_authenticator_refuses_wrong_registration_roles_and_untyped_inputs() -> None:
    target = _target(None)
    registration, result = _registered_result(target, result_interval=None)
    candidate = result.candidate
    other_registration = replace(
        registration,
        seal_receipt_sha256=_digest("other receipt"),
        _construction_key=audit_model._REGISTRATION_TOKEN,
    )

    with pytest.raises(ValueError, match="typed inputs"):
        content_authenticate_wet_lab_result(
            object(),  # type: ignore[arg-type]
            registration=registration,
            source_payloads_by_role={},
            analysis_artifact_payload=b"",
            verifier_artifact_payload=b"",
        )
    with pytest.raises(ValueError, match="supplied registration"):
        content_authenticate_wet_lab_result(
            candidate,
            registration=other_registration,
            source_payloads_by_role={},
            analysis_artifact_payload=b"",
            verifier_artifact_payload=b"",
        )
    with pytest.raises(ValueError, match="roles contradict"):
        content_authenticate_wet_lab_result(
            candidate,
            registration=registration,
            source_payloads_by_role={"wrong": b"payload"},
            analysis_artifact_payload=b"exact analysis artifact",
            verifier_artifact_payload=b"exact content-verifier artifact",
        )


def test_audit_reports_identity_observable_and_coverage_mismatches() -> None:
    target = _target(_calibrated_interval(0.4, 0.5))
    registration, result = _registered_result(
        target,
        result_interval=AuditInterval(
            0.44,
            0.47,
            0.90,
            "declared 90 percent interval",
            IntervalCalibrationStatus.DECLARED_CALIBRATED,
        ),
        value=0.46,
    )
    candidate = replace(
        result.candidate,
        registration_sha256=_digest("different registration"),
        observable=ObservableKind.TRIPLET_REACTION_YIELD,
    )
    contradictory = ContentAuthenticatedWetLabResult(
        candidate=candidate,
        authentication_receipt_sha256=_digest("content receipt"),
        _token=audit_model._AUTHENTICATION_TOKEN,
    )
    other_registration = replace(
        registration,
        target_sha256=_digest("different target"),
        _construction_key=audit_model._REGISTRATION_TOKEN,
    )

    report = audit_wet_lab_result(target, other_registration, contradictory)

    assert {
        AuditReason.TARGET_IDENTITY_MISMATCH,
        AuditReason.REGISTRATION_BINDING_MISMATCH,
        AuditReason.OBSERVABLE_MISMATCH,
        AuditReason.COVERAGE_MISMATCH,
    }.issubset(report.reasons)


@pytest.mark.parametrize(
    ("serializer", "message"),
    [
        (content_authenticated_wet_lab_result_document, "result serialization"),
        (wet_lab_audit_report_document, "audit serialization"),
    ],
)
def test_audit_serializers_refuse_untyped_values(serializer: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        serializer(object())  # type: ignore[operator]
