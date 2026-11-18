from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

import paircraft.oracle.maeda2012 as maeda_oracle_module
import paircraft.spin._maeda_kernel as kernel_module
import paircraft.spin.maeda2012 as spin_maeda_module
from paircraft.oracle import (
    AbstentionReason,
    ApplicabilityStatus,
    CalibrationTier,
    ConditionStatus,
    DeclaredCondition,
    FieldIntent,
    Maeda2012SimulatorOracle,
    ObservableKind,
    OracleAbstention,
    OraclePrediction,
    SpinModelFamily,
    maeda_2012_protocol,
    maeda_2012_query,
    maeda_2012_system,
)
from paircraft.spin import (
    MaedaProteinSystem,
    maeda_2012_specification,
    solve_maeda_secondary_pair_yield,
)


def test_system_catalog_uses_exact_spin_parameter_hashes() -> None:
    atcry = maeda_2012_system(MaedaProteinSystem.ATCRY1)
    ecpl = maeda_2012_system(MaedaProteinSystem.ECPL_E109A)

    assert (
        atcry.parameter_set_sha256
        == maeda_2012_specification(MaedaProteinSystem.ATCRY1).parameter_set_sha256
    )
    assert (
        ecpl.parameter_set_sha256
        == maeda_2012_specification(MaedaProteinSystem.ECPL_E109A).parameter_set_sha256
    )
    assert atcry.parameter_catalog_sha256 == ecpl.parameter_catalog_sha256
    assert atcry.system_id != ecpl.system_id


def test_oracle_artifact_identity_includes_neutral_kernel() -> None:
    digest = sha256()
    oracle_path = Path(maeda_oracle_module.__file__ or "")
    paths = (
        oracle_path,
        oracle_path.with_name("model.py"),
        Path(spin_maeda_module.__file__ or ""),
        Path(kernel_module.__file__ or ""),
    )
    for path in paths:
        digest.update(path.read_bytes())
    for system in MaedaProteinSystem:
        digest.update(maeda_2012_specification(system).parameter_set_sha256.encode())
    digest.update(maeda_oracle_module._CATALOG_VERSION.encode())

    assert Maeda2012SimulatorOracle().identity.artifact_sha256 == digest.hexdigest()


def test_protocol_retains_partial_conditions_without_invention() -> None:
    protocol = maeda_2012_protocol(MaedaProteinSystem.ATCRY1, (0.0, 0.0, 1e-3))

    assert protocol.temperature_kelvin == 270.0
    assert protocol.viscosity.status is ConditionStatus.UNKNOWN
    assert protocol.viscosity.value is None
    assert protocol.illumination.wavelength.value == 460e-9
    assert protocol.illumination.duration.value == 5e-9
    assert protocol.illumination.irradiance.status is ConditionStatus.UNKNOWN
    assert protocol.illumination.dose.status is ConditionStatus.UNKNOWN
    assert "5-7 mJ" in protocol.illumination.dose.note


def test_absolute_prediction_equals_direct_spin_solve() -> None:
    oracle = Maeda2012SimulatorOracle()
    query = maeda_2012_query(
        MaedaProteinSystem.ATCRY1,
        (0.0, 0.0, 1e-3),
        query_id="absolute",
    )

    prediction = oracle.predict(query)
    direct = solve_maeda_secondary_pair_yield(
        maeda_2012_specification(MaedaProteinSystem.ATCRY1),
        (0.0, 0.0, 1e-3),
    )

    assert isinstance(prediction, OraclePrediction)
    assert prediction.distribution.mean == pytest.approx(direct.secondary_pair, abs=1e-12)
    assert prediction.oracle.calibration_tier is CalibrationTier.SIMULATOR_ONLY
    assert prediction.applicability.status is ApplicabilityStatus.IN_DOMAIN
    assert prediction.distribution.coverage_probability is None
    assert prediction.execution is not None
    assert prediction.execution.achieved_relative_residual <= 1e-8
    assert prediction.execution.model_family is query.model_family
    assert prediction.execution.code_artifact_sha256 == direct.code_artifact_sha256
    assert len(prediction.lineage) >= 15
    lineage_ids = {artifact.artifact_id for artifact in prediction.lineage}
    assert "rp2-forward-rate" in lineage_ids
    assert "singlet-back-recombination-rate" in lineage_ids
    assert "singlet-triplet-dephasing-rate" in lineage_ids
    assert "initial-nuclear-state-assumption" in lineage_ids
    assert "non-identifiability" in prediction.distribution.uncertainty_assumptions


def test_relative_prediction_matches_explicit_target_reference_ratio() -> None:
    query = maeda_2012_query(
        MaedaProteinSystem.ECPL_E109A,
        (0.0, 0.0, 1e-3),
        ObservableKind.SECONDARY_RADICAL_YIELD_MFE,
        reference_field_tesla=(0.0, 0.0, 0.0),
        query_id="relative",
    )

    result = Maeda2012SimulatorOracle().predict(query)
    specification = maeda_2012_specification(MaedaProteinSystem.ECPL_E109A)
    target = solve_maeda_secondary_pair_yield(specification, (0.0, 0.0, 1e-3))
    reference = solve_maeda_secondary_pair_yield(specification, (0.0, 0.0, 0.0))

    assert isinstance(result, OraclePrediction)
    expected = (target.secondary_pair - reference.secondary_pair) / reference.secondary_pair
    assert result.distribution.mean == pytest.approx(expected, abs=1e-12)


def test_parameter_tampering_and_unsupported_observable_abstain() -> None:
    oracle = Maeda2012SimulatorOracle()
    query = maeda_2012_query(MaedaProteinSystem.ATCRY1, (0.0, 0.0, 1e-3))
    tampered = replace(query, system=replace(query.system, parameter_set_sha256="0" * 64))
    wrong_observable = replace(query, observable=ObservableKind.SINGLET_REACTION_YIELD)

    tampered_result = oracle.predict(tampered)
    observable_result = oracle.predict(wrong_observable)

    assert isinstance(tampered_result, OracleAbstention)
    assert tampered_result.reason is AbstentionReason.UNRESOLVED_PARAMETER_SET
    assert isinstance(observable_result, OracleAbstention)
    assert observable_result.reason is AbstentionReason.UNSUPPORTED_OBSERVABLE

    unknown_schema = oracle.predict(replace(query, schema_version="unknown-schema"))
    unknown_system = oracle.predict(
        replace(query, system=replace(query.system, system_id="unknown-system"))
    )
    assert isinstance(unknown_schema, OracleAbstention)
    assert unknown_schema.reason is AbstentionReason.OUTSIDE_SUPPORTED_DOMAIN
    assert isinstance(unknown_system, OracleAbstention)
    assert unknown_system.reason is AbstentionReason.UNSUPPORTED_SYSTEM


def test_changed_protocol_family_and_out_of_range_field_abstain() -> None:
    oracle = Maeda2012SimulatorOracle()
    query = maeda_2012_query(MaedaProteinSystem.ATCRY1, (0.0, 0.0, 1e-3))
    changed_protocol = replace(
        query,
        target_protocol=replace(query.target_protocol, temperature_kelvin=271.0),
    )
    wrong_family = replace(query, model_family=SpinModelFamily.HABERKORN_STATIC_PAIR)
    out_of_range = maeda_2012_query(
        MaedaProteinSystem.ATCRY1,
        (0.0, 0.0, 0.050001),
    )

    for changed in (changed_protocol, out_of_range):
        result = oracle.predict(changed)
        assert isinstance(result, OracleAbstention)
        assert result.reason is AbstentionReason.OUTSIDE_SUPPORTED_DOMAIN
    wrong_family_result = oracle.predict(wrong_family)
    assert isinstance(wrong_family_result, OracleAbstention)
    assert wrong_family_result.reason is AbstentionReason.MODEL_FAMILY_DISAGREEMENT

    incomplete = replace(
        query,
        target_protocol=replace(
            query.target_protocol,
            ph=DeclaredCondition(
                ConditionStatus.UNKNOWN,
                None,
                "1",
                "Required published pH removed for test.",
            ),
        ),
    )
    incomplete_result = oracle.predict(incomplete)
    assert isinstance(incomplete_result, OracleAbstention)
    assert incomplete_result.reason is AbstentionReason.INCOMPLETE_PROTOCOL


def test_relative_reference_on_domain_edge_marks_prediction_as_boundary() -> None:
    result = Maeda2012SimulatorOracle().predict(
        maeda_2012_query(
            MaedaProteinSystem.ATCRY1,
            (0.0, 0.0, 1e-3),
            ObservableKind.SECONDARY_RADICAL_YIELD_MFE,
            reference_field_tesla=(0.0, 0.0, 0.05),
        )
    )

    assert isinstance(result, OraclePrediction)
    assert result.applicability.status is ApplicabilityStatus.BOUNDARY


def test_nonideal_field_and_relative_condition_change_are_rejected() -> None:
    oracle = Maeda2012SimulatorOracle()
    query = maeda_2012_query(MaedaProteinSystem.ATCRY1, (0.0, 0.0, 1e-3))
    calibrated_field = replace(
        query.target_protocol.field,
        intent=FieldIntent.MEASURED_FIELD,
        calibration_id="field-map-v1",
        component_standard_uncertainty_tesla=(1e-9, 1e-9, 1e-9),
        covariance_tesla_squared=(
            (1e-18, 0.0, 0.0),
            (0.0, 1e-18, 0.0),
            (0.0, 0.0, 1e-18),
        ),
    )

    result = oracle.predict(
        replace(query, target_protocol=replace(query.target_protocol, field=calibrated_field))
    )
    assert isinstance(result, OracleAbstention)

    with pytest.raises(ValueError, match="differ only in magnetic flux density"):
        relative = maeda_2012_query(
            MaedaProteinSystem.ATCRY1,
            (0.0, 0.0, 1e-3),
            ObservableKind.SECONDARY_RADICAL_YIELD_MFE,
            reference_field_tesla=(0.0, 0.0, 0.0),
        )
        assert relative.reference_protocol is not None
        replace(
            relative,
            reference_protocol=replace(relative.reference_protocol, temperature_kelvin=269.0),
        )


def test_numerical_failure_becomes_abstention(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_solver(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise ArithmeticError("deliberate solver failure")

    monkeypatch.setattr(maeda_oracle_module, "solve_maeda_secondary_pair_yield", fail_solver)
    result = Maeda2012SimulatorOracle().predict(
        maeda_2012_query(MaedaProteinSystem.ATCRY1, (0.0, 0.0, 1e-3))
    )

    assert isinstance(result, OracleAbstention)
    assert result.reason is AbstentionReason.NUMERICAL_FAILURE
