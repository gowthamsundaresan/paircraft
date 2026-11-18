"""Grid-only QA for exact replay-authenticated Maeda sensitivity bundles."""

from __future__ import annotations

from dataclasses import InitVar, asdict, dataclass
from pathlib import Path
from typing import cast

from paircraft.sweeps.maeda_sensitivity import MaedaSensitivityJob, MaedaSensitivityPlan
from paircraft.sweeps.maeda_sensitivity_bundle import (
    load_authenticated_maeda_sensitivity_bundle,
)
from paircraft.sweeps.storage import canonical_json_bytes, canonical_sha256

_SCHEMA = "paircraft-maeda-sensitivity-grid-qa-v1"
_CONSTRUCTION_KEY = object()
_CLAIM_BOUNDARY = (
    "Grid-only descriptive QA of exact simulator-only hypothetical OAT point results. "
    "Within-family effects use only each family's enumerated zero-field result, and parent-"
    "baseline deltas compare only exact co-enumerated fields. No interpolation, uncertainty, "
    "ranking, training-data status, mutation effect, calibration, or biological claim is made."
)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    return cast(dict[str, object], value)


def _real(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a real scalar")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must lie in [0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class MaedaSensitivityQaRow:
    """One exact family/field point with two explicit deterministic comparisons."""

    job_sha256: str
    parent_system: str
    family_id: str
    dimension: str
    direction: str
    physical_model_family: str
    executed_model_id: str
    baseline_parameter_id: str
    baseline_value_si: float
    proposed_value_si: float
    perturbation_delta_si: float
    parameter_unit: str
    log_scale_delta: float
    parameter_family_sha256: str
    design_dependency_sha256: str
    field_tesla: float
    absolute_secondary_yield: float
    within_family_zero_relative_effect: float
    absolute_yield_delta_from_parent_baseline: float
    relative_effect_delta_from_parent_baseline: float
    zero_reference_job_sha256: str
    parent_baseline_job_sha256: str
    datum_sha256: str
    configuration_sha256: str
    corpus_membership: str


@dataclass(frozen=True, slots=True)
class MaedaSensitivityCurveQa:
    """Exact-grid summary for one hypothetical parameter family."""

    parent_system: str
    family_id: str
    dimension: str
    direction: str
    physical_model_family: str
    executed_model_id: str
    baseline_parameter_id: str
    baseline_value_si: float
    proposed_value_si: float
    perturbation_delta_si: float
    parameter_unit: str
    log_scale_delta: float
    parameter_family_sha256: str
    design_dependency_sha256: str
    point_count: int
    zero_field_yield: float
    relative_effect_at_10_microtesla: float
    relative_effect_at_20_microtesla: float
    relative_effect_at_50_microtesla: float
    delta_from_parent_at_10_microtesla: float
    delta_from_parent_at_20_microtesla: float
    delta_from_parent_at_50_microtesla: float


@dataclass(frozen=True, slots=True)
class MaedaSensitivityQaReport:
    """Canonical report constructible only from an authenticated complete bundle."""

    plan_sha256: str
    bundle_sha256: str
    rows: tuple[MaedaSensitivityQaRow, ...]
    curves: tuple[MaedaSensitivityCurveQa, ...]
    design_dependency_sha256: tuple[str, ...]
    _construction_key: InitVar[object]
    schema_version: str = _SCHEMA
    claim_boundary: str = _CLAIM_BOUNDARY

    def __post_init__(self, _construction_key: object) -> None:
        if _construction_key is not _CONSTRUCTION_KEY:
            raise ValueError("sensitivity QA requires authenticated bundle construction")
        if len(self.rows) != 204 or len(self.curves) != 34:
            raise ValueError("sensitivity QA requires 204 rows and 34 family curves")
        if len(self.design_dependency_sha256) != 2:
            raise ValueError("sensitivity QA requires exactly two design dependencies")

    @property
    def report_sha256(self) -> str:
        """Hash the complete grid-only report payload."""

        return canonical_sha256(_payload(self))


def _payload(report: MaedaSensitivityQaReport) -> dict[str, object]:
    return {
        "schema_version": report.schema_version,
        "plan_sha256": report.plan_sha256,
        "bundle_sha256": report.bundle_sha256,
        "claim_boundary": report.claim_boundary,
        "design_dependency_sha256": list(report.design_dependency_sha256),
        "rows": [asdict(row) for row in report.rows],
        "curves": [asdict(curve) for curve in report.curves],
    }


def build_maeda_sensitivity_qa_report(
    bundle_path: Path,
    *,
    expected_plan: MaedaSensitivityPlan,
) -> MaedaSensitivityQaReport:
    """Derive exact-grid comparisons after full bundle replay authentication."""

    bundle, documents = load_authenticated_maeda_sensitivity_bundle(
        bundle_path, expected_plan=expected_plan
    )
    if len(documents) != len(expected_plan.jobs):
        raise ValueError("authenticated sensitivity bundle is incomplete")
    values: dict[str, float] = {}
    datum_hashes: dict[str, str] = {}
    configuration_hashes: dict[str, str] = {}
    memberships: dict[str, str] = {}
    model_families: dict[str, str] = {}
    model_ids: dict[str, str] = {}
    for job, document in zip(expected_plan.jobs, documents, strict=True):
        prediction = _mapping(document.get("prediction"), "prediction")
        query = _mapping(prediction.get("query"), "query")
        distribution = _mapping(prediction.get("distribution"), "distribution")
        hashes = _mapping(document.get("hashes"), "datum hashes")
        values[job.job_sha256] = _real(distribution.get("mean"), "absolute secondary yield")
        datum_hashes[job.job_sha256] = cast(str, hashes.get("datum_sha256"))
        configuration_hashes[job.job_sha256] = cast(str, hashes.get("configuration_sha256"))
        memberships[job.job_sha256] = cast(str, document.get("membership"))
        model_family = cast(str, query.get("model_family"))
        model_id = cast(str, prediction.get("executed_model_id"))
        if (
            model_family != "haberkorn_static_pair_with_st_dephasing"
            or prediction.get("executed_model_family") != model_family
            or model_id != job.execution_input.model_id
        ):
            raise ValueError("authenticated result contradicts its planned physical model")
        model_families[job.job_sha256] = model_family
        model_ids[job.job_sha256] = model_id

    by_parent_field: dict[tuple[str, float], MaedaSensitivityJob] = {}
    by_family_zero: dict[str, MaedaSensitivityJob] = {}
    for job in expected_plan.jobs:
        parent = job.family.parent.protein_system.value
        if job.family.family_id == "baseline":
            by_parent_field[(parent, job.field_point.magnitude_tesla)] = job
        if job.field_point.magnitude_tesla == 0.0:
            by_family_zero[job.parameter_family_sha256] = job
    if len(by_parent_field) != 12 or len(by_family_zero) != 34:
        raise ValueError("sensitivity plan lacks an exact baseline or zero-field grid member")

    rows: list[MaedaSensitivityQaRow] = []
    for job in expected_plan.jobs:
        parent = job.family.parent.protein_system.value
        zero = by_family_zero[job.parameter_family_sha256]
        baseline = by_parent_field[(parent, job.field_point.magnitude_tesla)]
        zero_hash = zero.job_sha256
        baseline_hash = baseline.job_sha256
        value = values[job.job_sha256]
        zero_value = values[zero_hash]
        baseline_value = values[baseline_hash]
        baseline_zero = by_parent_field[(parent, 0.0)]
        baseline_zero_value = values[baseline_zero.job_sha256]
        if zero_value == 0.0 or baseline_zero_value == 0.0:
            raise ValueError("zero-field yield cannot normalize a sensitivity curve")
        relative = (value - zero_value) / zero_value
        baseline_relative = (baseline_value - baseline_zero_value) / baseline_zero_value
        override = job.execution_input.override
        rows.append(
            MaedaSensitivityQaRow(
                job_sha256=job.job_sha256,
                parent_system=parent,
                family_id=job.family.family_id,
                dimension=override.dimension.value,
                direction=override.direction.value,
                physical_model_family=model_families[job.job_sha256],
                executed_model_id=model_ids[job.job_sha256],
                baseline_parameter_id=override.baseline_parameter_id,
                baseline_value_si=override.baseline_value_si,
                proposed_value_si=override.proposed_value_si,
                perturbation_delta_si=(override.proposed_value_si - override.baseline_value_si),
                parameter_unit=override.unit,
                log_scale_delta=override.log_scale_delta,
                parameter_family_sha256=job.parameter_family_sha256,
                design_dependency_sha256=job.design_dependency_sha256,
                field_tesla=job.field_point.magnitude_tesla,
                absolute_secondary_yield=value,
                within_family_zero_relative_effect=relative,
                absolute_yield_delta_from_parent_baseline=value - baseline_value,
                relative_effect_delta_from_parent_baseline=relative - baseline_relative,
                zero_reference_job_sha256=zero_hash,
                parent_baseline_job_sha256=baseline_hash,
                datum_sha256=datum_hashes[job.job_sha256],
                configuration_sha256=configuration_hashes[job.job_sha256],
                corpus_membership=memberships[job.job_sha256],
            )
        )

    curves: list[MaedaSensitivityCurveQa] = []
    for family in dict.fromkeys(job.parameter_family_sha256 for job in expected_plan.jobs):
        curve = tuple(row for row in rows if row.parameter_family_sha256 == family)
        if len(curve) != 6 or len({row.design_dependency_sha256 for row in curve}) != 1:
            raise ValueError("each sensitivity family must contain exactly six dependent fields")
        by_field = {row.field_tesla: row for row in curve}
        if set(by_field) != {0.0, 10e-6, 20e-6, 50e-6, 100e-6, 200e-6}:
            raise ValueError("sensitivity QA accepts only the exact enumerated field grid")
        representative = curve[0]
        curves.append(
            MaedaSensitivityCurveQa(
                parent_system=representative.parent_system,
                family_id=representative.family_id,
                dimension=representative.dimension,
                direction=representative.direction,
                physical_model_family=representative.physical_model_family,
                executed_model_id=representative.executed_model_id,
                baseline_parameter_id=representative.baseline_parameter_id,
                baseline_value_si=representative.baseline_value_si,
                proposed_value_si=representative.proposed_value_si,
                perturbation_delta_si=representative.perturbation_delta_si,
                parameter_unit=representative.parameter_unit,
                log_scale_delta=representative.log_scale_delta,
                parameter_family_sha256=representative.parameter_family_sha256,
                design_dependency_sha256=representative.design_dependency_sha256,
                point_count=6,
                zero_field_yield=by_field[0.0].absolute_secondary_yield,
                relative_effect_at_10_microtesla=(
                    by_field[10e-6].within_family_zero_relative_effect
                ),
                relative_effect_at_20_microtesla=(
                    by_field[20e-6].within_family_zero_relative_effect
                ),
                relative_effect_at_50_microtesla=(
                    by_field[50e-6].within_family_zero_relative_effect
                ),
                delta_from_parent_at_10_microtesla=(
                    by_field[10e-6].relative_effect_delta_from_parent_baseline
                ),
                delta_from_parent_at_20_microtesla=(
                    by_field[20e-6].relative_effect_delta_from_parent_baseline
                ),
                delta_from_parent_at_50_microtesla=(
                    by_field[50e-6].relative_effect_delta_from_parent_baseline
                ),
            )
        )
    dependencies = tuple(sorted({row.design_dependency_sha256 for row in rows}))
    return MaedaSensitivityQaReport(
        plan_sha256=expected_plan.plan_sha256,
        bundle_sha256=bundle.bundle_sha256,
        rows=tuple(rows),
        curves=tuple(curves),
        design_dependency_sha256=dependencies,
        _construction_key=_CONSTRUCTION_KEY,
    )


def maeda_sensitivity_qa_json(report: MaedaSensitivityQaReport) -> bytes:
    """Serialize an authenticated grid-only report as canonical JSON."""

    return canonical_json_bytes({**_payload(report), "report_sha256": report.report_sha256})
