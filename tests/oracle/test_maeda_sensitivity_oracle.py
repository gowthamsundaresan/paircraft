from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

import paircraft.oracle.maeda2012 as published_oracle_module
import paircraft.oracle.maeda_sensitivity as oracle_module
import paircraft.oracle.model as oracle_model_module
import paircraft.spin._maeda_kernel as kernel_module
import paircraft.spin.maeda2012 as published_spin_module
import paircraft.spin.maeda_sensitivity as sensitivity_spin_module
import paircraft.sweeps.maeda_sensitivity as sensitivity_sweep_module
from paircraft.oracle.maeda2012 import Maeda2012SimulatorOracle
from paircraft.oracle.maeda_sensitivity import (
    MaedaSensitivitySimulatorOracle,
    execute_maeda_sensitivity_oracle_job,
    execute_maeda_sensitivity_oracle_plan,
    iter_maeda_sensitivity_oracle_jobs,
    maeda_sensitivity_query,
    maeda_sensitivity_system,
)
from paircraft.oracle.model import (
    AbstentionReason,
    ApplicabilityStatus,
    CalibrationTier,
    ObservableKind,
    OracleAbstention,
    OraclePrediction,
    OracleQuery,
    OracleResult,
    SpinModelFamily,
)
from paircraft.oracle.simulation import CorpusMembership, EvidenceOrigin
from paircraft.sweeps.maeda_sensitivity import (
    MaedaSensitivityPlan,
    default_maeda_local_sensitivity,
    materialize_maeda_local_sensitivity,
)


@pytest.fixture(scope="module")
def plan() -> MaedaSensitivityPlan:
    return materialize_maeda_local_sensitivity(default_maeda_local_sensitivity())


def test_identity_is_distinct_simulator_only_catalog(plan: MaedaSensitivityPlan) -> None:
    sensitivity = MaedaSensitivitySimulatorOracle(plan).identity
    reference = Maeda2012SimulatorOracle().identity

    assert sensitivity.oracle_id != reference.oracle_id
    assert sensitivity.parameter_catalog_version != reference.parameter_catalog_version
    assert sensitivity.parameter_catalog_sha256 != reference.parameter_catalog_sha256
    assert sensitivity.calibration_tier is CalibrationTier.SIMULATOR_ONLY
    assert sensitivity.supported_model_families == (
        SpinModelFamily.HABERKORN_STATIC_PAIR_WITH_ST_DEPHASING,
    )
    assert sensitivity.supported_observables == (ObservableKind.SECONDARY_RADICAL_YIELD,)


def test_oracle_artifact_identity_includes_all_inherited_implementations(
    plan: MaedaSensitivityPlan,
) -> None:
    digest = sha256()
    for module in (
        oracle_module,
        oracle_model_module,
        sensitivity_spin_module,
        kernel_module,
        published_spin_module,
        published_oracle_module,
        sensitivity_sweep_module,
    ):
        digest.update(Path(module.__file__ or "").read_bytes())
    digest.update(plan.plan_sha256.encode())
    digest.update(oracle_module._CATALOG_VERSION.encode())

    assert MaedaSensitivitySimulatorOracle(plan).identity.artifact_sha256 == digest.hexdigest()


def test_exact_job_prediction_is_point_only_and_empirically_unknown(
    plan: MaedaSensitivityPlan,
) -> None:
    job = plan.jobs[3]
    query = maeda_sensitivity_query(plan, job)
    result = MaedaSensitivitySimulatorOracle(plan).predict(query)

    assert isinstance(result, OraclePrediction)
    assert result.query.system.parameter_set_sha256 == job.parameter_family_sha256
    assert result.executed_model_id == job.execution_input.model_id
    assert result.distribution.mean == result.distribution.lower == result.distribution.upper
    assert result.distribution.coverage_probability is None
    assert result.distribution.epistemic_standard_uncertainty is None
    assert result.distribution.aleatoric_standard_uncertainty is None
    assert result.distribution.simulator_discrepancy_standard_uncertainty is None
    assert result.distribution.numerical_standard_uncertainty is None
    assert result.applicability.status is ApplicabilityStatus.IN_DOMAIN
    assert any(
        "Empirical applicability: UNKNOWN" in reason for reason in result.applicability.reasons
    )
    assert "not parameter uncertainty" in result.claim_boundary


def test_lineage_retains_baseline_transform_family_dependency_job_and_execution(
    plan: MaedaSensitivityPlan,
) -> None:
    job = plan.jobs[6]
    result = MaedaSensitivitySimulatorOracle(plan).predict(maeda_sensitivity_query(plan, job))
    assert isinstance(result, OraclePrediction)
    lineage = {artifact.artifact_id: artifact for artifact in result.lineage}

    assert (
        lineage[f"{job.family.parent.protein_system.value}-published-baseline-parameter-set"].sha256
        == job.family.parent.parent_parameter_set_sha256
    )
    assert lineage["hypothetical-parameter-transform"].evidence_status == (
        "assumed_hypothetical_design"
    )
    assert lineage["hypothetical-parameter-family"].sha256 == job.parameter_family_sha256
    assert lineage["sensitivity-design-dependency-group"].sha256 == (job.design_dependency_sha256)
    assert lineage["sensitivity-plan-job"].sha256 == job.job_sha256
    assert "maeda-sensitivity-numerical-execution" in lineage
    assert len(result.lineage) >= 16
    assert all("not training data" in artifact.membership for artifact in result.lineage)


def test_exact_plan_execution_returns_reference_only_simulation_datum(
    plan: MaedaSensitivityPlan,
) -> None:
    job = plan.jobs[0]
    datum = execute_maeda_sensitivity_oracle_job(plan, job)

    assert datum.evidence_origin is EvidenceOrigin.SIMULATION
    assert datum.membership is CorpusMembership.REFERENCE_ONLY_NOT_TRAINING
    assert datum.prediction.query.system.parameter_set_sha256 == job.parameter_family_sha256
    lineage = {artifact.artifact_id: artifact.sha256 for artifact in datum.prediction.lineage}
    assert lineage["sensitivity-design-dependency-group"] == job.design_dependency_sha256
    assert lineage["sensitivity-plan-job"] == job.job_sha256


def test_plan_execution_does_not_accept_a_same_plan_fake_oracle(
    plan: MaedaSensitivityPlan,
) -> None:
    class SamePlanFakeOracle(MaedaSensitivitySimulatorOracle):
        def predict(self, query: OracleQuery) -> OracleResult:
            del query
            raise AssertionError("a fake oracle must never be called")

    fake = SamePlanFakeOracle(plan)
    with pytest.raises(TypeError):
        execute_maeda_sensitivity_oracle_job(plan, plan.jobs[0], fake)  # type: ignore[call-arg]


def test_parameter_field_observable_and_family_tampering_abstain(
    plan: MaedaSensitivityPlan,
) -> None:
    job = plan.jobs[0]
    oracle = MaedaSensitivitySimulatorOracle(plan)
    query = maeda_sensitivity_query(plan, job)
    cases = (
        (
            replace(query, system=replace(query.system, parameter_set_sha256="0" * 64)),
            AbstentionReason.UNRESOLVED_PARAMETER_SET,
        ),
        (
            replace(
                query,
                target_protocol=replace(
                    query.target_protocol,
                    field=replace(query.target_protocol.field, z_tesla=30e-6),
                ),
            ),
            AbstentionReason.OUTSIDE_SUPPORTED_DOMAIN,
        ),
        (
            replace(query, observable=ObservableKind.SINGLET_REACTION_YIELD),
            AbstentionReason.UNSUPPORTED_OBSERVABLE,
        ),
        (
            replace(query, model_family=SpinModelFamily.HABERKORN_STATIC_PAIR),
            AbstentionReason.MODEL_FAMILY_DISAGREEMENT,
        ),
    )
    for changed, expected_reason in cases:
        result = oracle.predict(changed)
        assert isinstance(result, OracleAbstention)
        assert result.reason is expected_reason
        assert result.applicability.status is ApplicabilityStatus.OUT_OF_DOMAIN


def test_query_constructor_rejects_job_from_another_plan(
    plan: MaedaSensitivityPlan,
) -> None:
    alien = replace(plan.jobs[0], specification_sha256="0" * 64)
    with pytest.raises(ValueError, match="exact job"):
        maeda_sensitivity_query(plan, alien)


def test_numerical_failure_abstains_without_result(
    plan: MaedaSensitivityPlan, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise ArithmeticError("injected solver failure")

    monkeypatch.setattr(oracle_module, "execute_maeda_sensitivity_job", fail)
    result = MaedaSensitivitySimulatorOracle(plan).predict(
        maeda_sensitivity_query(plan, plan.jobs[0])
    )

    assert isinstance(result, OracleAbstention)
    assert result.reason is AbstentionReason.NUMERICAL_FAILURE


def test_oracle_public_boundaries_reject_untyped_or_foreign_plan_members(
    plan: MaedaSensitivityPlan,
) -> None:
    alien_job = replace(plan.jobs[0], specification_sha256="0" * 64)

    with pytest.raises(ValueError, match="MaedaSensitivityPlan"):
        MaedaSensitivitySimulatorOracle("plan")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exact family"):
        maeda_sensitivity_system(plan, "family")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="typed sensitivity plan"):
        execute_maeda_sensitivity_oracle_plan("plan")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="typed plan and job tuple"):
        tuple(iter_maeda_sensitivity_oracle_jobs(plan, [plan.jobs[0]]))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="outside its plan"):
        tuple(iter_maeda_sensitivity_oracle_jobs(plan, (alien_job,)))
    with pytest.raises(ValueError, match="exact sensitivity plan job"):
        execute_maeda_sensitivity_oracle_job(plan, alien_job)


@pytest.mark.parametrize(
    ("change", "expected_reason", "expected_dimension"),
    (
        (
            lambda query: replace(query, schema_version="foreign-schema"),
            AbstentionReason.OUTSIDE_SUPPORTED_DOMAIN,
            "schema_version",
        ),
        (
            lambda query: replace(
                query,
                system=replace(query.system, system_id="unknown-system"),
            ),
            AbstentionReason.UNSUPPORTED_SYSTEM,
            "system",
        ),
        (
            lambda query: replace(
                query,
                observable=ObservableKind.SECONDARY_RADICAL_YIELD_MFE,
                reference_protocol=query.target_protocol,
            ),
            AbstentionReason.UNSUPPORTED_OBSERVABLE,
            "reference_protocol",
        ),
    ),
)
def test_nonmember_query_dimensions_abstain_without_execution(
    plan: MaedaSensitivityPlan,
    change: Callable[[OracleQuery], OracleQuery],
    expected_reason: AbstentionReason,
    expected_dimension: str,
) -> None:
    query = maeda_sensitivity_query(plan, plan.jobs[0])
    changed = change(query)
    result = MaedaSensitivitySimulatorOracle(plan).predict(changed)

    assert isinstance(result, OracleAbstention)
    assert result.reason is expected_reason
    assert expected_dimension in result.unsupported_dimensions
