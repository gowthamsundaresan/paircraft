from __future__ import annotations

import json
from dataclasses import replace

import pytest

import paircraft.sweeps.maeda_sensitivity as sweep_module
from paircraft.spin import MaedaProteinSystem
from paircraft.spin.maeda_sensitivity import (
    SensitivityDimension,
    SensitivityDirection,
    maeda_sensitivity_specification,
)
from paircraft.sweeps.maeda_sensitivity import (
    MaedaSensitivityPlan,
    default_maeda_local_sensitivity,
    execute_maeda_sensitivity_job,
    maeda_local_sensitivity_families,
    maeda_sensitivity_parent,
    maeda_sensitivity_plan_document,
    maeda_sensitivity_plan_json,
    materialize_maeda_local_sensitivity,
)


@pytest.fixture(scope="module")
def plan() -> MaedaSensitivityPlan:
    return materialize_maeda_local_sensitivity(default_maeda_local_sensitivity())


def test_default_design_materializes_exact_204_job_product(plan: MaedaSensitivityPlan) -> None:
    assert len(plan.specification.parents) == 2
    assert len(plan.specification.dimensions) == 8
    assert len(plan.specification.field_points) == 6
    assert len(plan.jobs) == 204
    assert len({job.job_sha256 for job in plan.jobs}) == 204
    assert plan.specification.random_seed is None
    assert plan.specification.corpus_membership == "reference_only_not_training"
    assert "not parameter uncertainty" in plan.specification.claim_boundary


def test_each_parent_has_baseline_and_low_high_oat_families() -> None:
    for system in MaedaProteinSystem:
        families = maeda_local_sensitivity_families(maeda_sensitivity_parent(system))
        assert len(families) == 17
        assert families[0].family_id == "baseline"
        assert families[0].spin_specification.override.direction is SensitivityDirection.BASELINE
        for family in families[1:]:
            override = family.spin_specification.override
            assert override.direction in {
                SensitivityDirection.LOWER,
                SensitivityDirection.HIGHER,
            }
            assert override.log_scale_delta in {-0.05, 0.05}


def test_job_order_and_family_dependencies_are_canonical(plan: MaedaSensitivityPlan) -> None:
    first = plan.jobs[0]
    assert first.family.parent.protein_system is MaedaProteinSystem.ATCRY1
    assert first.family.family_id == "baseline"
    assert first.field_point.magnitude_tesla == 0.0
    assert [job.field_point.magnitude_tesla for job in plan.jobs[:6]] == [
        0.0,
        10e-6,
        20e-6,
        50e-6,
        100e-6,
        200e-6,
    ]
    assert len({job.parameter_family_sha256 for job in plan.jobs}) == 34
    assert len({job.design_dependency_sha256 for job in plan.jobs}) == 2
    for offset in range(0, len(plan.jobs), 6):
        family_jobs = plan.jobs[offset : offset + 6]
        assert len({job.parameter_family_sha256 for job in family_jobs}) == 1
        assert len({job.design_dependency_sha256 for job in family_jobs}) == 1
    for offset in range(0, len(plan.jobs), 102):
        parent_jobs = plan.jobs[offset : offset + 102]
        assert len({job.parameter_family_sha256 for job in parent_jobs}) == 17
        assert len({job.design_dependency_sha256 for job in parent_jobs}) == 1


def test_parameter_family_and_design_dependency_are_distinct(plan: MaedaSensitivityPlan) -> None:
    job = plan.jobs[0]
    assert job.parameter_family_sha256 != job.design_dependency_sha256
    assert job.execution_input.baseline_parameter_set_sha256 == (
        job.family.parent.parent_parameter_set_sha256
    )
    assert job.execution_input == job.family.spin_specification


def test_plan_serialization_is_canonical_and_contains_no_results(
    plan: MaedaSensitivityPlan,
) -> None:
    document = maeda_sensitivity_plan_document(plan)
    encoded = maeda_sensitivity_plan_json(plan)

    assert (
        encoded
        == json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    assert document["hashes"]["plan_sha256"] == plan.plan_sha256  # type: ignore[index]
    assert len(document["jobs"]) == 204  # type: ignore[arg-type]
    assert set(document) == {"schema_version", "specification", "jobs", "hashes"}
    jobs = document["jobs"]
    assert isinstance(jobs, list)
    assert all(
        isinstance(job, dict) and not {"result", "outcome", "yield"}.intersection(job)
        for job in jobs
    )


def test_hashes_are_reproducible() -> None:
    first = materialize_maeda_local_sensitivity(default_maeda_local_sensitivity())
    second = materialize_maeda_local_sensitivity(default_maeda_local_sensitivity())

    assert first.specification.specification_sha256 == second.specification.specification_sha256
    assert first.plan_sha256 == second.plan_sha256
    assert [job.job_sha256 for job in first.jobs] == [job.job_sha256 for job in second.jobs]


def test_family_rejects_noncanonical_duplicate_baseline() -> None:
    with pytest.raises(ValueError, match="canonical only for the forward-rate"):
        maeda_sensitivity_specification(
            MaedaProteinSystem.ATCRY1,
            SensitivityDimension.FAD_N5_HYPERFINE,
            SensitivityDirection.BASELINE,
        )


def test_plan_rejects_missing_or_reordered_jobs(plan: MaedaSensitivityPlan) -> None:
    with pytest.raises(ValueError, match="canonical materialization"):
        MaedaSensitivityPlan(plan.specification, plan.jobs[:-1])
    with pytest.raises(ValueError, match="canonical materialization"):
        MaedaSensitivityPlan(plan.specification, tuple(reversed(plan.jobs)))


def test_job_rejects_execution_input_from_another_field(plan: MaedaSensitivityPlan) -> None:
    first = plan.jobs[0]
    another_family = plan.jobs[6]
    with pytest.raises(ValueError, match="contradicts its family"):
        replace(first, execution_input=another_family.execution_input)


def test_one_planned_job_executes_through_hypothetical_spin_layer(
    plan: MaedaSensitivityPlan,
) -> None:
    job = plan.jobs[0]
    result = execute_maeda_sensitivity_job(plan, job)

    assert result.specification == job.execution_input
    assert result.field_tesla == job.field_point.vector_tesla
    assert result.parameter_set_sha256 == job.parameter_family_sha256
    assert "not experimental calibration" in result.claim_boundary


def test_design_cannot_be_relabelled_as_uncertainty() -> None:
    specification = default_maeda_local_sensitivity()
    with pytest.raises(ValueError, match="claim semantics"):
        replace(specification, claim_boundary="parameter uncertainty ensemble")


def test_parent_and_family_reject_identity_and_claim_contradictions() -> None:
    parent = maeda_sensitivity_parent(MaedaProteinSystem.ATCRY1)
    other_parent = maeda_sensitivity_parent(MaedaProteinSystem.ECPL_E109A)
    family = maeda_local_sensitivity_families(parent)[1]

    with pytest.raises(ValueError, match="Maeda protein system"):
        replace(parent, protein_system="atcry1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exact published specification"):
        replace(parent, parent_model_id="other")
    with pytest.raises(ValueError, match="typed parent"):
        replace(family, parent="parent")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="family ID must be explicit"):
        replace(family, family_id="")
    with pytest.raises(ValueError, match="hypothetical spin specification"):
        replace(family, spin_specification="specification")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="contradicts its parent"):
        replace(family, parent=other_parent)
    with pytest.raises(ValueError, match="family ID contradicts"):
        replace(family, family_id="different")
    with pytest.raises(ValueError, match="claim semantics"):
        replace(family, corpus_membership="training")
    with pytest.raises(ValueError, match="typed parent"):
        maeda_local_sensitivity_families("parent")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"design_id": ""}, "design ID must be explicit"),
        ({"schema_version": "v2"}, "schema is unsupported"),
        ({"parents": ()}, "both exact published parents"),
        ({"field_points": ()}, "exact six low-field points"),
        ({"dimensions": ()}, "all eight canonical dimensions"),
        ({"log_scale_deltas": ()}, "centered log-scale offsets"),
        ({"random_seed": None, "design_method": "random"}, "deterministic centered OAT"),
        ({"evidence_status": "measured"}, "claim semantics"),
    ],
)
def test_sweep_specification_rejects_noncanonical_design(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(default_maeda_local_sensitivity(), **changes)  # type: ignore[arg-type]


def test_job_plan_and_serialization_reject_untyped_or_foreign_values(
    plan: MaedaSensitivityPlan,
) -> None:
    job = plan.jobs[0]
    with pytest.raises(ValueError, match="design ID must be explicit"):
        replace(job, design_id="")
    with pytest.raises(ValueError, match="64-character hexadecimal"):
        replace(job, specification_sha256="bad")
    with pytest.raises(ValueError, match="typed family"):
        replace(job, family="family")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ideal field point"):
        replace(job, field_point="field")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="typed specification"):
        sweep_module.MaedaSensitivityPlan("specification", ())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="immutable typed tuple"):
        sweep_module.MaedaSensitivityPlan(plan.specification, [])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="typed specification"):
        materialize_maeda_local_sensitivity("specification")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="typed plan"):
        maeda_sensitivity_plan_document("plan")  # type: ignore[arg-type]


def test_execution_and_validation_reject_foreign_job_result_or_field(
    plan: MaedaSensitivityPlan,
) -> None:
    job = plan.jobs[0]
    result = execute_maeda_sensitivity_job(plan, job)
    foreign_job = replace(job, design_id="foreign-design")

    with pytest.raises(ValueError, match="typed plan and job"):
        execute_maeda_sensitivity_job("plan", job)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not an exact member"):
        execute_maeda_sensitivity_job(plan, foreign_job)
    with pytest.raises(ValueError, match="HypotheticalSecondaryPairYield"):
        sweep_module.validate_maeda_sensitivity_job_result(plan, job, "result")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="field contradicts"):
        sweep_module.validate_maeda_sensitivity_job_result(
            plan,
            job,
            replace(result, field_tesla=(0.0, 0.0, 10e-6)),
        )
