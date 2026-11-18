from dataclasses import replace

import pytest

import paircraft.oracle.reference as reference_module
from paircraft.oracle import (
    AbstentionReason,
    ApplicabilityAssessment,
    ApplicabilityStatus,
    ConditionStatus,
    DeclaredCondition,
    DeclaredQuantity,
    FieldIntent,
    HaberkornReferenceOracle,
    IlluminationProtocol,
    ObservableKind,
    ObservationWindow,
    OracleAbstention,
    OraclePrediction,
    OracleQuery,
    PredictiveDistribution,
    SpinModelFamily,
    SupportBound,
    reference_protocol,
    reference_query,
    reference_system,
)


def test_exact_reference_query_returns_auditable_conserved_yields() -> None:
    oracle = HaberkornReferenceOracle()

    singlet = oracle.predict(reference_query(50e-6, query_id="singlet"))
    triplet = oracle.predict(
        reference_query(
            50e-6,
            ObservableKind.TRIPLET_REACTION_YIELD,
            query_id="triplet",
        )
    )

    assert isinstance(singlet, OraclePrediction)
    assert isinstance(triplet, OraclePrediction)
    assert singlet.query.query_id == "singlet"
    assert singlet.oracle == oracle.identity
    assert singlet.distribution.unit == "1"
    assert singlet.distribution.coverage_probability is None
    assert singlet.applicability.status is ApplicabilityStatus.IN_DOMAIN
    assert singlet.query.system.parameter_set_sha256 == reference_system().parameter_set_sha256
    assert len(singlet.lineage) == 5
    assert all(artifact.exact_location for artifact in singlet.lineage)
    assert all(artifact.source_uri for artifact in singlet.lineage)
    hyperfine = next(
        artifact
        for artifact in singlet.lineage
        if artifact.artifact_id == "effective-flavin-h8-hyperfine"
    )
    assert hyperfine.source_uri == "https://doi.org/10.1021/acs.jpcb.7b07672"
    assert hyperfine.source_artifact_sha256 is None
    assert "not stated" in hyperfine.license
    assert singlet.distribution.mean + triplet.distribution.mean == pytest.approx(1.0, abs=1e-12)


def test_out_of_domain_field_abstains_without_a_numeric_result() -> None:
    result = HaberkornReferenceOracle().predict(reference_query(0.100001))

    assert isinstance(result, OracleAbstention)
    assert result.reason is AbstentionReason.OUTSIDE_SUPPORTED_DOMAIN
    assert result.applicability.status is ApplicabilityStatus.OUT_OF_DOMAIN
    assert result.unsupported_dimensions == ("target_protocol",)


def test_relative_observable_requires_an_exact_reference_protocol() -> None:
    with pytest.raises(ValueError, match="requires a complete reference"):
        OracleQuery(
            schema_version="paircraft-oracle-query-v1",
            query_id="missing-reference",
            system=reference_system(),
            target_protocol=reference_protocol(50e-6),
            reference_protocol=None,
            observable=ObservableKind.SINGLET_REACTION_YIELD_MFE,
            model_family=HaberkornReferenceOracle().identity.supported_model_families[0],
        )


def test_predictive_distribution_rejects_an_impossible_interval() -> None:
    with pytest.raises(ValueError, match="must contain its mean"):
        PredictiveDistribution(
            mean=0.5,
            lower=0.6,
            upper=0.7,
            coverage_probability=0.95,
            unit="1",
            method="invalid test fixture",
            epistemic_standard_uncertainty=None,
            aleatoric_standard_uncertainty=None,
            simulator_discrepancy_standard_uncertainty=None,
            numerical_standard_uncertainty=None,
            decomposition_status="invalid fixture",
            uncertainty_assumptions="invalid fixture",
            calibration_status="invalid",
            note="This interval is deliberately impossible.",
        )


def test_parameter_hash_mismatch_cannot_return_a_prediction() -> None:
    query = reference_query(50e-6, query_id="wrong-parameter-hash")
    wrong_system = replace(query.system, parameter_set_sha256="0" * 64)

    result = HaberkornReferenceOracle().predict(replace(query, system=wrong_system))

    assert isinstance(result, OracleAbstention)
    assert result.reason is AbstentionReason.UNRESOLVED_PARAMETER_SET
    assert "parameter_set" in result.unsupported_dimensions


def test_unknown_query_schema_cannot_return_a_prediction() -> None:
    query = replace(reference_query(50e-6), schema_version="unknown-schema-v999")

    result = HaberkornReferenceOracle().predict(query)

    assert isinstance(result, OracleAbstention)
    assert result.reason is AbstentionReason.OUTSIDE_SUPPORTED_DOMAIN
    assert result.unsupported_dimensions == ("schema_version",)


def test_unsupported_secondary_yield_abstains() -> None:
    query = replace(
        reference_query(50e-6),
        observable=ObservableKind.SECONDARY_RADICAL_YIELD,
    )

    result = HaberkornReferenceOracle().predict(query)

    assert isinstance(result, OracleAbstention)
    assert result.reason is AbstentionReason.UNSUPPORTED_OBSERVABLE
    assert result.unsupported_dimensions == ("observable",)


def test_isotropic_reference_model_is_invariant_to_field_direction() -> None:
    oracle = HaberkornReferenceOracle()
    z_query = reference_query(1e-3, query_id="z-directed")
    x_field = replace(z_query.target_protocol.field, x_tesla=1e-3, z_tesla=0.0)
    x_query = replace(
        z_query,
        query_id="x-directed",
        target_protocol=replace(z_query.target_protocol, field=x_field),
    )

    z_result = oracle.predict(z_query)
    x_result = oracle.predict(x_query)

    assert isinstance(z_result, OraclePrediction)
    assert isinstance(x_result, OraclePrediction)
    assert x_result.distribution.mean == pytest.approx(z_result.distribution.mean, abs=1e-12)


def test_calibrated_field_requires_calibration_and_covariance() -> None:
    field = reference_protocol(50e-6).field

    with pytest.raises(ValueError, match="require a calibration ID"):
        replace(field, intent=FieldIntent.CALIBRATED_APPLIED_FIELD)

    with pytest.raises(ValueError, match="require uncertainty and covariance"):
        replace(
            field,
            intent=FieldIntent.CALIBRATED_APPLIED_FIELD,
            calibration_id="coil-map-v1",
        )

    calibrated = replace(
        field,
        intent=FieldIntent.CALIBRATED_APPLIED_FIELD,
        calibration_id="coil-map-v1",
        component_standard_uncertainty_tesla=(1e-9, 2e-9, 3e-9),
        covariance_tesla_squared=(
            (1e-18, 0.0, 0.0),
            (0.0, 4e-18, 0.0),
            (0.0, 0.0, 9e-18),
        ),
    )
    assert calibrated.calibration_id == "coil-map-v1"

    with pytest.raises(ValueError, match="diagonal must match"):
        replace(
            calibrated,
            covariance_tesla_squared=(
                (2e-18, 0.0, 0.0),
                (0.0, 4e-18, 0.0),
                (0.0, 0.0, 9e-18),
            ),
        )

    with pytest.raises(ValueError, match="must be symmetric"):
        replace(
            calibrated,
            covariance_tesla_squared=(
                (1e-18, 1e-18, 0.0),
                (0.0, 4e-18, 0.0),
                (0.0, 0.0, 9e-18),
            ),
        )

    with pytest.raises(ValueError, match="positive semidefinite"):
        replace(
            calibrated,
            component_standard_uncertainty_tesla=(1e-9, 1e-9, 1e-9),
            covariance_tesla_squared=(
                (1e-18, 2e-18, 0.0),
                (2e-18, 1e-18, 0.0),
                (0.0, 0.0, 1e-18),
            ),
        )

    with pytest.raises(ValueError, match="three components and a 3x3 covariance"):
        replace(
            calibrated,
            component_standard_uncertainty_tesla=(1e-9, 1e-9),  # type: ignore[arg-type]
            covariance_tesla_squared=(  # type: ignore[arg-type]
                (1e-18, 0.0),
                (0.0, 1e-18),
            ),
        )


def test_relative_query_rejects_nonfield_condition_changes() -> None:
    target = reference_protocol(50e-6)
    reference = replace(reference_protocol(0.0), temperature_kelvin=297.15)

    with pytest.raises(ValueError, match="differ only in magnetic flux density"):
        OracleQuery(
            schema_version="paircraft-oracle-query-v1",
            query_id="changed-temperature",
            system=reference_system(),
            target_protocol=target,
            reference_protocol=reference,
            observable=ObservableKind.SINGLET_REACTION_YIELD_MFE,
            model_family=HaberkornReferenceOracle().identity.supported_model_families[0],
        )


def test_numerical_failure_returns_abstention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_solver(_field: tuple[float, float, float]) -> tuple[float, float]:
        raise ArithmeticError("deliberate solver failure")

    monkeypatch.setattr(reference_module, "_reaction_yields", fail_solver)

    result = HaberkornReferenceOracle().predict(reference_query(50e-6))

    assert isinstance(result, OracleAbstention)
    assert result.reason is AbstentionReason.NUMERICAL_FAILURE
    assert result.unsupported_dimensions == ("solver",)


def test_condition_statuses_cannot_hide_missing_or_invented_values() -> None:
    valued = DeclaredCondition(ConditionStatus.VALUE, 7.0, "1", "Measured pH.")
    assert valued.value == 7.0

    with pytest.raises(ValueError, match="requires a value"):
        DeclaredCondition(ConditionStatus.VALUE, None, "1", "Missing pH.")

    with pytest.raises(ValueError, match="cannot carry a value"):
        DeclaredCondition(ConditionStatus.UNKNOWN, "air", "1", "Invented oxygen state.")

    with pytest.raises(ValueError, match="ConditionStatus"):
        DeclaredCondition("partial", None, "1", "Invalid raw status.")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="ConditionStatus"):
        DeclaredQuantity("partial", None, "Pa s", "Invalid raw status.")  # type: ignore[arg-type]


def test_illumination_and_observation_windows_are_structured() -> None:
    illumination = IlluminationProtocol(
        wavelength=DeclaredQuantity(ConditionStatus.VALUE, 450e-9, "m", "Known wavelength."),
        irradiance=DeclaredQuantity(
            ConditionStatus.UNKNOWN,
            None,
            "W m^-2",
            "Irradiance was not reported.",
        ),
        duration=DeclaredQuantity(ConditionStatus.VALUE, 0.1, "s", "Known duration."),
        dose=DeclaredQuantity(
            ConditionStatus.UNKNOWN,
            None,
            "J m^-2",
            "Areal dose was not reported.",
        ),
        timing="pulse begins at time zero",
        note="Partially known toy optical preparation.",
    )
    window = ObservationWindow(1e-6, 2e-6, "One-microsecond integration window.")
    assert illumination.wavelength.value == 450e-9
    assert illumination.irradiance.status is ConditionStatus.UNKNOWN
    assert window.end_seconds == 2e-6

    with pytest.raises(ValueError, match="strictly positive"):
        replace(
            illumination,
            wavelength=DeclaredQuantity(ConditionStatus.VALUE, 0.0, "m", "Invalid zero."),
        )

    with pytest.raises(ValueError, match="cannot carry a value"):
        DeclaredQuantity(ConditionStatus.UNKNOWN, 1.0, "W m^-2", "Contradiction.")

    with pytest.raises(ValueError, match="later than its start"):
        ObservationWindow(2e-6, 1e-6, "Backwards window.")


def test_absolute_query_rejects_a_reference_protocol() -> None:
    with pytest.raises(ValueError, match="absolute observable cannot carry"):
        OracleQuery(
            schema_version="paircraft-oracle-query-v1",
            query_id="absolute-with-reference",
            system=reference_system(),
            target_protocol=reference_protocol(50e-6),
            reference_protocol=reference_protocol(0.0),
            observable=ObservableKind.SINGLET_REACTION_YIELD,
            model_family=HaberkornReferenceOracle().identity.supported_model_families[0],
        )


def test_numeric_prediction_cannot_claim_out_of_domain_applicability() -> None:
    result = HaberkornReferenceOracle().predict(reference_query(50e-6))
    assert isinstance(result, OraclePrediction)
    out_of_domain = ApplicabilityAssessment(
        status=ApplicabilityStatus.OUT_OF_DOMAIN,
        reasons=("Deliberate contradiction.",),
        unsupported_dimensions=("field",),
        support_evidence_ids=(),
        support_bounds=(),
        model_family_disagreement="not assessed",
    )

    with pytest.raises(ValueError, match="cannot carry a numeric prediction"):
        replace(result, applicability=out_of_domain)


def test_distribution_rejects_invalid_coverage_and_uncertainty() -> None:
    result = HaberkornReferenceOracle().predict(reference_query(50e-6))
    assert isinstance(result, OraclePrediction)

    with pytest.raises(ValueError, match="coverage probability"):
        replace(result.distribution, coverage_probability=1.0)

    with pytest.raises(ValueError, match="standard uncertainties"):
        replace(result.distribution, epistemic_standard_uncertainty=-0.1)


def test_relative_mfe_matches_independent_target_reference_ratio() -> None:
    oracle = HaberkornReferenceOracle()
    target = oracle.predict(reference_query(50e-6, query_id="target"))
    reference = oracle.predict(reference_query(0.0, query_id="reference"))
    relative = oracle.predict(
        reference_query(
            50e-6,
            ObservableKind.SINGLET_REACTION_YIELD_MFE,
            reference_z_tesla=0.0,
            query_id="relative",
        )
    )

    assert isinstance(target, OraclePrediction)
    assert isinstance(reference, OraclePrediction)
    assert isinstance(relative, OraclePrediction)
    expected = (
        target.distribution.mean - reference.distribution.mean
    ) / reference.distribution.mean
    assert relative.distribution.mean == pytest.approx(expected, rel=1e-12)


def test_declared_values_and_protocols_reject_nonphysical_or_ambiguous_inputs() -> None:
    protocol = reference_protocol(50e-6)
    with pytest.raises(ValueError, match="numeric condition values must be finite"):
        DeclaredCondition(ConditionStatus.VALUE, float("nan"), "1", "Invalid numeric condition.")
    with pytest.raises(ValueError, match="condition value"):
        DeclaredCondition(ConditionStatus.VALUE, " ", "1", "Invalid blank condition.")
    with pytest.raises(ValueError, match="finite numeric value"):
        DeclaredQuantity(ConditionStatus.VALUE, float("nan"), "s", "Invalid duration.")
    with pytest.raises(ValueError, match="simulation field cannot claim"):
        replace(protocol.field, calibration_id="invented-calibration")
    with pytest.raises(ValueError, match="calibration ID must be explicit"):
        replace(
            protocol.field,
            intent=FieldIntent.MEASURED_FIELD,
            calibration_id=" ",
        )
    with pytest.raises(ValueError, match="component uncertainties must be finite and nonnegative"):
        replace(
            protocol.field,
            intent=FieldIntent.MEASURED_FIELD,
            calibration_id="field-map",
            component_standard_uncertainty_tesla=(-1.0, 0.0, 0.0),
            covariance_tesla_squared=((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        )
    with pytest.raises(ValueError, match="covariance values must be finite"):
        replace(
            protocol.field,
            intent=FieldIntent.MEASURED_FIELD,
            calibration_id="field-map",
            component_standard_uncertainty_tesla=(1.0, 1.0, 1.0),
            covariance_tesla_squared=(
                (1.0, float("nan"), 0.0),
                (float("nan"), 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
        )
    with pytest.raises(ValueError, match="illumination wavelength must use"):
        replace(
            protocol.illumination,
            wavelength=replace(protocol.illumination.wavelength, unit="nm"),
        )
    with pytest.raises(ValueError, match="observation start"):
        replace(protocol.observation_window, start_seconds=-1.0)
    with pytest.raises(ValueError, match="temperature"):
        replace(protocol, temperature_kelvin=0.0)
    with pytest.raises(ValueError, match="viscosity must use"):
        replace(protocol, viscosity=replace(protocol.viscosity, unit="cP"))
    with pytest.raises(ValueError, match="valued viscosity"):
        replace(protocol, viscosity=replace(protocol.viscosity, value=-1.0))


def test_oracle_result_contracts_reject_identity_and_support_contradictions() -> None:
    oracle = HaberkornReferenceOracle()
    result = oracle.predict(reference_query(50e-6))
    assert isinstance(result, OraclePrediction)

    with pytest.raises(ValueError, match="explicit supported families"):
        replace(result.oracle, supported_observables=())
    with pytest.raises(ValueError, match="finite and ordered"):
        SupportBound("field", 1.0, 0.0, "T", "Invalid reversed bound.")
    with pytest.raises(ValueError, match="at least one reason"):
        replace(result.applicability, reasons=())
    with pytest.raises(ValueError, match="source artifact sha256"):
        replace(result.lineage[0], source_artifact_sha256="bad")
    with pytest.raises(ValueError, match="exact resolved query"):
        replace(
            result,
            resolved_setup=replace(
                result.resolved_setup,
                target_protocol=reference_protocol(1e-3),
            ),
        )
    with pytest.raises(ValueError, match="executed model family"):
        replace(result, executed_model_family=SpinModelFamily.STOCHASTIC_DIFFUSIVE_PAIR)
    with pytest.raises(ValueError, match="at least one lineage"):
        replace(result, lineage=())
    with pytest.raises(ValueError, match="intervals must lie in"):
        replace(result, distribution=replace(result.distribution, upper=1.1))

    abstention = oracle.predict(reference_query(0.100001))
    assert isinstance(abstention, OracleAbstention)
    with pytest.raises(ValueError, match="unsupported or unknown applicability"):
        replace(
            abstention,
            applicability=replace(
                abstention.applicability,
                status=ApplicabilityStatus.IN_DOMAIN,
            ),
        )
