from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

import paircraft.sweeps.maeda_sensitivity_qa as qa_module
from paircraft.sweeps.maeda_sensitivity import (
    MaedaSensitivityPlan,
    default_maeda_local_sensitivity,
    materialize_maeda_local_sensitivity,
)
from paircraft.sweeps.maeda_sensitivity_qa import (
    MaedaSensitivityQaReport,
    build_maeda_sensitivity_qa_report,
    maeda_sensitivity_qa_json,
)


@pytest.fixture(scope="module")
def plan() -> MaedaSensitivityPlan:
    return materialize_maeda_local_sensitivity(default_maeda_local_sensitivity())


@pytest.fixture
def authenticated_grid(
    monkeypatch: pytest.MonkeyPatch,
    plan: MaedaSensitivityPlan,
) -> None:
    family_order = {
        (job.family.parent.protein_system.value, job.family.family_id): index
        for index, job in enumerate(plan.jobs[::6])
    }

    def digest(value: str) -> str:
        return sha256(value.encode()).hexdigest()

    documents: list[dict[str, object]] = []
    for index, job in enumerate(plan.jobs):
        parent = job.family.parent.protein_system.value
        family_index = family_order[(parent, job.family.family_id)]
        zero = 0.4 + (0.05 if parent == "ecpl_e109a" else 0.0) + family_index / 10_000
        slope = 100.0 + family_index
        value = zero * (1.0 + slope * job.field_point.magnitude_tesla)
        documents.append(
            {
                "membership": "reference_only_not_training",
                "prediction": {
                    "query": {"model_family": "haberkorn_static_pair_with_st_dephasing"},
                    "executed_model_family": ("haberkorn_static_pair_with_st_dephasing"),
                    "executed_model_id": job.execution_input.model_id,
                    "distribution": {"mean": value},
                },
                "hashes": {
                    "datum_sha256": digest(f"datum:{index}"),
                    "configuration_sha256": digest("configuration"),
                },
            }
        )

    def load(
        path: Path,
        *,
        expected_plan: MaedaSensitivityPlan,
    ) -> tuple[SimpleNamespace, tuple[dict[str, object], ...]]:
        del path
        assert expected_plan is plan
        return SimpleNamespace(bundle_sha256=digest("bundle")), tuple(documents)

    monkeypatch.setattr(qa_module, "load_authenticated_maeda_sensitivity_bundle", load)


def test_report_has_exact_grid_comparisons_without_ranking_or_interpolation(
    tmp_path: Path,
    plan: MaedaSensitivityPlan,
    authenticated_grid: None,
) -> None:
    del authenticated_grid
    report = build_maeda_sensitivity_qa_report(
        tmp_path / "authenticated.bundle.json", expected_plan=plan
    )

    assert len(report.rows) == 204
    assert len(report.curves) == 34
    assert len(report.design_dependency_sha256) == 2
    assert all(
        row.within_family_zero_relative_effect == 0.0
        for row in report.rows
        if row.field_tesla == 0.0
    )
    baseline = next(curve for curve in report.curves if curve.family_id == "baseline")
    assert baseline.relative_effect_at_10_microtesla == pytest.approx(0.001)
    assert baseline.relative_effect_at_20_microtesla == pytest.approx(0.002)
    assert baseline.relative_effect_at_50_microtesla == pytest.approx(0.005)
    assert baseline.delta_from_parent_at_10_microtesla == pytest.approx(0.0)
    assert baseline.delta_from_parent_at_20_microtesla == pytest.approx(0.0)
    assert baseline.delta_from_parent_at_50_microtesla == pytest.approx(0.0)
    assert baseline.physical_model_family == "haberkorn_static_pair_with_st_dephasing"
    assert baseline.executed_model_id == plan.jobs[0].execution_input.model_id
    assert "No interpolation" in report.claim_boundary
    assert "ranking" in report.claim_boundary


def test_signed_h23_perturbation_exports_value_not_ambiguous_direction(
    tmp_path: Path,
    plan: MaedaSensitivityPlan,
    authenticated_grid: None,
) -> None:
    del authenticated_grid
    report = build_maeda_sensitivity_qa_report(
        tmp_path / "authenticated.bundle.json", expected_plan=plan
    )
    h23_higher = next(
        curve
        for curve in report.curves
        if curve.dimension == "trph-h23-isotropic-hyperfine"
        and curve.direction == "higher-exp-plus-0.05"
    )

    assert h23_higher.baseline_value_si < 0.0
    assert h23_higher.proposed_value_si < h23_higher.baseline_value_si
    assert h23_higher.perturbation_delta_si < 0.0
    assert h23_higher.log_scale_delta > 0.0
    assert h23_higher.parameter_unit == "T"


def test_nonbaseline_family_pairs_each_field_to_its_own_zero_and_parent_field(
    tmp_path: Path,
    plan: MaedaSensitivityPlan,
    authenticated_grid: None,
) -> None:
    del authenticated_grid
    report = build_maeda_sensitivity_qa_report(
        tmp_path / "authenticated.bundle.json", expected_plan=plan
    )
    family = next(
        curve
        for curve in report.curves
        if curve.dimension == "singlet-back-recombination-rate"
        and curve.direction == "higher-exp-plus-0.05"
        and curve.parent_system == "atcry1"
    )
    rows = tuple(
        row for row in report.rows if row.parameter_family_sha256 == family.parameter_family_sha256
    )
    zero = next(row for row in rows if row.field_tesla == 0.0)
    parent_rows = {
        row.field_tesla: row
        for row in report.rows
        if row.parent_system == family.parent_system and row.family_id == "baseline"
    }
    parent_zero = parent_rows[0.0]

    assert len(rows) == 6
    for row in rows:
        parent = parent_rows[row.field_tesla]
        expected_relative = (
            row.absolute_secondary_yield - zero.absolute_secondary_yield
        ) / zero.absolute_secondary_yield
        parent_relative = (
            parent.absolute_secondary_yield - parent_zero.absolute_secondary_yield
        ) / parent_zero.absolute_secondary_yield
        assert row.zero_reference_job_sha256 == zero.job_sha256
        assert row.parent_baseline_job_sha256 == parent.job_sha256
        assert row.within_family_zero_relative_effect == pytest.approx(expected_relative)
        assert row.absolute_yield_delta_from_parent_baseline == pytest.approx(
            row.absolute_secondary_yield - parent.absolute_secondary_yield
        )
        assert row.relative_effect_delta_from_parent_baseline == pytest.approx(
            expected_relative - parent_relative
        )


def test_qa_json_is_canonical_and_contains_no_rank_or_uncertainty_field(
    tmp_path: Path,
    plan: MaedaSensitivityPlan,
    authenticated_grid: None,
) -> None:
    del authenticated_grid
    report = build_maeda_sensitivity_qa_report(
        tmp_path / "authenticated.bundle.json", expected_plan=plan
    )
    raw = maeda_sensitivity_qa_json(report)
    document = json.loads(raw)

    assert document["report_sha256"] == report.report_sha256
    assert (
        raw == json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    assert all("rank" not in key and "uncertainty" not in key for key in document)


def test_report_cannot_be_constructed_without_authenticated_loader() -> None:
    with pytest.raises(ValueError, match="authenticated bundle"):
        MaedaSensitivityQaReport(
            plan_sha256="0" * 64,
            bundle_sha256="1" * 64,
            rows=(),
            curves=(),
            design_dependency_sha256=(),
            _construction_key=object(),
        )
