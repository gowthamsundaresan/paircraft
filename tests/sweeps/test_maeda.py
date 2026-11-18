from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from math import copysign

import pytest

from paircraft.oracle import (
    Maeda2012SimulatorOracle,
    ObservableKind,
    OracleIdentity,
    SimulationDatum,
)
from paircraft.spin import MaedaProteinSystem, maeda_2012_specification
from paircraft.sweeps import (
    IdealFieldPoint,
    MaedaSweepJob,
    MaedaSweepPlan,
    MaedaSweepSpecification,
    ParameterSetPolicy,
    RandomnessStatus,
    SweepSamplingMethod,
    default_maeda_reference_sweep,
    execute_maeda_sweep_job,
    maeda_sweep_plan_json,
    maeda_sweep_target,
    materialize_maeda_sweep,
    validate_maeda_job_result,
)


def _small_spec() -> MaedaSweepSpecification:
    return MaedaSweepSpecification(
        sweep_id="maeda-test-sweep",
        targets=(maeda_sweep_target(MaedaProteinSystem.ATCRY1),),
        field_points=(
            IdealFieldPoint("zero", 0.0),
            IdealFieldPoint("fifty-ut", 50e-6),
        ),
        claim_boundary="Exact published-model field sensitivity; reference-only, not prediction.",
    )


def test_default_sweep_materializes_exact_28_job_grid() -> None:
    specification = default_maeda_reference_sweep()
    plan = materialize_maeda_sweep(specification)

    assert len(specification.targets) == 2
    assert len(specification.field_points) == 14
    assert len(plan.jobs) == 28
    assert tuple(point.magnitude_tesla for point in specification.field_points) == (
        0.0,
        10e-6,
        20e-6,
        50e-6,
        100e-6,
        200e-6,
        500e-6,
        1e-3,
        2e-3,
        5e-3,
        10e-3,
        20e-3,
        30e-3,
        50e-3,
    )
    assert specification.observable is ObservableKind.SECONDARY_RADICAL_YIELD
    assert specification.parameter_set_policy is ParameterSetPolicy.PUBLISHED_POINT_ESTIMATES_ONLY
    assert specification.sampling_method is SweepSamplingMethod.ENUMERATED_CARTESIAN_PRODUCT
    assert specification.randomness_status is RandomnessStatus.NOT_APPLICABLE_DETERMINISTIC
    assert specification.random_seed is None
    assert all(job.query_sha256 != job.job_sha256 for job in plan.jobs)


def test_input_order_does_not_change_specification_or_plan_identity() -> None:
    baseline = default_maeda_reference_sweep()
    reordered = replace(
        baseline,
        targets=tuple(reversed(baseline.targets)),
        field_points=tuple(reversed(baseline.field_points)),
    )
    first = materialize_maeda_sweep(baseline)
    second = materialize_maeda_sweep(reordered)

    assert baseline.specification_sha256 == reordered.specification_sha256
    assert first.jobs == second.jobs
    assert first.plan_sha256 == second.plan_sha256
    assert maeda_sweep_plan_json(first) == maeda_sweep_plan_json(second)
    encoded = maeda_sweep_plan_json(first)
    assert b'"prediction"' not in encoded
    assert b'"distribution"' not in encoded
    document = json.loads(encoded)
    assert all("scientific_family_sha256" in job for job in document["jobs"])
    assert all("job_sha256" in job for job in document["jobs"])
    assert document["execution_oracle"]["artifact_sha256"] == first.execution_oracle.artifact_sha256


def test_all_fields_of_one_target_share_scientific_family() -> None:
    plan = materialize_maeda_sweep(default_maeda_reference_sweep())
    families: dict[MaedaProteinSystem, set[str]] = {}
    for job in plan.jobs:
        families.setdefault(job.target.protein_system, set()).add(job.scientific_family_sha256)

    assert all(len(values) == 1 for values in families.values())
    assert families[MaedaProteinSystem.ATCRY1] != families[MaedaProteinSystem.ECPL_E109A]


def test_target_resolves_exact_catalog_and_rejects_tampering() -> None:
    target = maeda_sweep_target(MaedaProteinSystem.ATCRY1)
    source_hash = maeda_2012_specification(MaedaProteinSystem.ATCRY1).parameter_set_sha256

    assert target.parameter_set_sha256 == source_hash
    with pytest.raises(ValueError, match="exact published Maeda catalog"):
        replace(target, parameter_set_sha256="0" * 64)
    assert maeda_2012_specification(MaedaProteinSystem.ATCRY1).parameter_set_sha256 == source_hash


@pytest.mark.parametrize("field", [-1.0, float("nan"), 0.050001])
def test_field_domain_is_fail_closed(field: float) -> None:
    with pytest.raises(ValueError, match=r"0-0\.05 T"):
        IdealFieldPoint("bad", field)


def test_field_numbers_canonicalize_without_accepting_booleans() -> None:
    integer_zero = IdealFieldPoint("zero", 0)
    negative_zero = IdealFieldPoint("zero", -0.0)
    positive_zero = IdealFieldPoint("zero", 0.0)

    assert integer_zero == positive_zero == negative_zero
    assert isinstance(integer_zero.magnitude_tesla, float)
    assert copysign(1.0, negative_zero.magnitude_tesla) == 1.0
    with pytest.raises(ValueError, match="not a boolean"):
        IdealFieldPoint("bad", False)


def test_specification_rejects_random_relative_and_ambiguous_axes() -> None:
    specification = _small_spec()
    invalid: tuple[Callable[[], object], ...] = (
        lambda: replace(specification, targets=list(specification.targets)),  # type: ignore[arg-type]
        lambda: replace(specification, field_points=list(specification.field_points)),  # type: ignore[arg-type]
        lambda: replace(specification, field_points=specification.field_points[1:]),
        lambda: replace(
            specification,
            field_points=(IdealFieldPoint("a", 0.0), IdealFieldPoint("b", 0.0)),
        ),
        lambda: replace(specification, observable=ObservableKind.SECONDARY_RADICAL_YIELD_MFE),
        lambda: replace(specification, random_seed=7),  # type: ignore[arg-type]
        lambda: replace(specification, parameter_set_policy="perturbed"),  # type: ignore[arg-type]
        lambda: replace(specification, sampling_method="random"),  # type: ignore[arg-type]
        lambda: replace(specification, randomness_status="seeded"),  # type: ignore[arg-type]
    )
    for constructor in invalid:
        with pytest.raises(ValueError):
            constructor()


def test_plan_requires_exact_cartesian_product_and_authenticated_jobs() -> None:
    plan = materialize_maeda_sweep(_small_spec())
    with pytest.raises(ValueError, match="Cartesian product"):
        MaedaSweepPlan(plan.specification, plan.jobs[:-1], plan.execution_oracle)
    with pytest.raises(ValueError, match="query hash"):
        replace(plan.jobs[0], query_sha256="0" * 64)
    with pytest.raises(ValueError, match="query ID"):
        replace(plan.jobs[0], query_id="wrong")
    with pytest.raises(ValueError, match="hexadecimal"):
        replace(plan.jobs[0], specification_sha256="z" * 64)


class _DifferentArtifactOracle(Maeda2012SimulatorOracle):
    @property
    def identity(self) -> OracleIdentity:
        return replace(super().identity, artifact_sha256="f" * 64)


def test_execution_is_bound_to_parent_plan_and_pinned_oracle() -> None:
    plan = materialize_maeda_sweep(_small_spec())
    job = plan.jobs[0]
    forged_job = replace(job, specification_sha256="e" * 64)
    changed_plan = replace(plan, execution_oracle=_DifferentArtifactOracle().identity)

    with pytest.raises(ValueError, match="exact member"):
        execute_maeda_sweep_job(plan, forged_job)
    with pytest.raises(ValueError, match="oracle identity"):
        execute_maeda_sweep_job(plan, job, _DifferentArtifactOracle())
    with pytest.raises(ValueError, match="oracle identity"):
        execute_maeda_sweep_job(changed_plan, job)
    assert changed_plan.plan_sha256 != plan.plan_sha256


@pytest.fixture(scope="module")
def executed_first_job() -> tuple[MaedaSweepPlan, MaedaSweepJob, SimulationDatum]:
    plan = materialize_maeda_sweep(_small_spec())
    job = plan.jobs[0]
    return plan, job, execute_maeda_sweep_job(plan, job)


def test_one_real_job_executes_and_authenticates(
    executed_first_job: tuple[MaedaSweepPlan, MaedaSweepJob, SimulationDatum],
) -> None:
    plan, job, datum = executed_first_job
    validate_maeda_job_result(plan, job, datum)
    assert datum.query_sha256 == job.query_sha256
    assert datum.membership.value == "reference_only_not_training"


def test_result_validation_rejects_a_different_field(
    executed_first_job: tuple[MaedaSweepPlan, MaedaSweepJob, SimulationDatum],
) -> None:
    plan, _, datum = executed_first_job
    other_job = materialize_maeda_sweep(_small_spec()).jobs[1]
    with pytest.raises(ValueError, match="exact sweep query"):
        validate_maeda_job_result(plan, other_job, datum)


def test_result_validation_rejects_mutated_oracle_artifact(
    executed_first_job: tuple[MaedaSweepPlan, MaedaSweepJob, SimulationDatum],
) -> None:
    plan, job, datum = executed_first_job
    mutated_identity = replace(datum.prediction.oracle, artifact_sha256="f" * 64)
    mutated_prediction = replace(datum.prediction, oracle=mutated_identity)
    mutated_datum = SimulationDatum(mutated_prediction, datum.execution)

    with pytest.raises(ValueError, match="oracle identity"):
        validate_maeda_job_result(plan, job, mutated_datum)
