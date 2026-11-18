"""Auditable summaries of authenticated deterministic Maeda sweep bundles."""

from __future__ import annotations

import json
import math
import os
import tempfile
from contextlib import suppress
from dataclasses import InitVar, asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

from paircraft.sweeps.bundle import load_authenticated_maeda_sweep_bundle
from paircraft.sweeps.maeda import MaedaSweepPlan

_SCHEMA_VERSION = "paircraft-maeda-sweep-qa-v1"
_AUTHENTICATED_CONSTRUCTION = object()
_CLAIM_BOUNDARY = (
    "Descriptive QA of deterministic simulator-only point results; relative effects are "
    "zero-field-normalized derived views, not measurements, uncertainty, calibration, "
    "biological variability, mutation effects, or prospective predictions."
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _canonical_sha256(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    return cast(dict[str, object], value)


def _real(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


@dataclass(frozen=True, slots=True)
class MaedaSweepQaRow:
    """One authenticated field calculation represented without reinterpretation."""

    job_sha256: str
    system_id: str
    parameter_set_id: str
    model_family: str
    field_magnitude_tesla: float
    absolute_secondary_yield: float
    achieved_relative_residual: float
    iteration_count: int
    applicability_status: str
    configuration_sha256: str
    dependency_group_sha256: str
    datum_sha256: str
    evidence_origin: str
    corpus_membership: str


@dataclass(frozen=True, slots=True)
class ZeroFieldRelativeEffect:
    """A derived comparison whose target and reference remain explicit."""

    system_id: str
    parameter_set_id: str
    model_family: str
    dependency_group_sha256: str
    configuration_sha256: str
    field_magnitude_tesla: float
    target_job_sha256: str
    reference_job_sha256: str
    target_datum_sha256: str
    reference_datum_sha256: str
    relative_effect: float
    unit: str = "1"
    uncertainty_status: str = (
        "not calculated; published parameter covariance and predictive discrepancy are unavailable"
    )


@dataclass(frozen=True, slots=True)
class MaedaCurveSummary:
    """Grid-only descriptive facts for one published point-estimate system."""

    system_id: str
    parameter_set_id: str
    model_family: str
    dependency_group_sha256: str
    configuration_sha256: str
    point_count: int
    zero_field_yield: float
    minimum_yield: float
    maximum_yield: float
    yield_span: float
    maximum_absolute_relative_effect: float
    relative_effect_at_10_microtesla: float | None
    relative_effect_at_20_microtesla: float | None
    relative_effect_at_50_microtesla: float | None
    maximum_achieved_relative_residual: float
    maximum_iteration_count: int


@dataclass(frozen=True, slots=True)
class MaedaSweepQaReport:
    """Canonical QA view bound to one fully authenticated sweep bundle."""

    plan_sha256: str
    bundle_sha256: str
    rows: tuple[MaedaSweepQaRow, ...]
    zero_field_relative_effects: tuple[ZeroFieldRelativeEffect, ...]
    curve_summaries: tuple[MaedaCurveSummary, ...]
    _construction_key: InitVar[object]
    schema_version: str = _SCHEMA_VERSION
    claim_boundary: str = _CLAIM_BOUNDARY

    def __post_init__(self, _construction_key: object) -> None:
        if _construction_key is not _AUTHENTICATED_CONSTRUCTION:
            raise ValueError("QA reports must be constructed from an authenticated sweep bundle")
        if not self.rows or not self.zero_field_relative_effects or not self.curve_summaries:
            raise ValueError("an authenticated QA report cannot be empty")

    @property
    def report_sha256(self) -> str:
        return _canonical_sha256(_report_payload(self))


def _report_payload(report: MaedaSweepQaReport) -> dict[str, object]:
    return {
        "schema_version": report.schema_version,
        "plan_sha256": report.plan_sha256,
        "bundle_sha256": report.bundle_sha256,
        "claim_boundary": report.claim_boundary,
        "rows": [asdict(row) for row in report.rows],
        "zero_field_relative_effects": [
            asdict(effect) for effect in report.zero_field_relative_effects
        ],
        "curve_summaries": [asdict(summary) for summary in report.curve_summaries],
    }


def _row_from_document(
    document: dict[str, object],
    *,
    job_sha256: str,
    expected_field_tesla: float,
) -> MaedaSweepQaRow:
    prediction = _mapping(document["prediction"], "prediction")
    query = _mapping(prediction["query"], "query")
    system = _mapping(query["system"], "system")
    distribution = _mapping(prediction["distribution"], "distribution")
    applicability = _mapping(prediction["applicability"], "applicability")
    execution = _mapping(document["execution"], "execution")
    hashes = _mapping(document["hashes"], "hashes")
    value = _real(distribution["mean"], "absolute secondary yield")
    field = expected_field_tesla
    if value < 0.0 or value > 1.0:
        raise ValueError("absolute secondary yield must lie in [0, 1]")
    return MaedaSweepQaRow(
        job_sha256=job_sha256,
        system_id=cast(str, system["system_id"]),
        parameter_set_id=cast(str, system["parameter_set_id"]),
        model_family=cast(str, query["model_family"]),
        field_magnitude_tesla=field,
        absolute_secondary_yield=value,
        achieved_relative_residual=_real(
            execution["achieved_relative_residual"], "achieved relative residual"
        ),
        iteration_count=_integer(execution["iteration_count"], "iteration count"),
        applicability_status=cast(str, applicability["status"]),
        configuration_sha256=cast(str, hashes["configuration_sha256"]),
        dependency_group_sha256=cast(str, hashes["dependency_group_sha256"]),
        datum_sha256=cast(str, hashes["datum_sha256"]),
        evidence_origin=cast(str, document["evidence_origin"]),
        corpus_membership=cast(str, document["membership"]),
    )


def build_maeda_sweep_qa_report(
    bundle_path: Path,
    *,
    expected_plan: MaedaSweepPlan,
) -> MaedaSweepQaReport:
    """Build a grid-only QA report after complete plan-bound bundle authentication."""

    bundle, documents = load_authenticated_maeda_sweep_bundle(
        bundle_path, expected_plan=expected_plan
    )
    if len(documents) != len(expected_plan.jobs):
        raise ValueError("authenticated bundle and plan job counts differ")
    rows = tuple(
        _row_from_document(
            document,
            job_sha256=job.job_sha256,
            expected_field_tesla=job.field_point.magnitude_tesla,
        )
        for job, document in zip(expected_plan.jobs, documents, strict=True)
    )
    tolerance = 1e-8
    if any(row.achieved_relative_residual > tolerance for row in rows):
        raise ValueError("a solver residual exceeds the authenticated requested tolerance")

    effects: list[ZeroFieldRelativeEffect] = []
    summaries: list[MaedaCurveSummary] = []
    for target in expected_plan.specification.targets:
        curve = tuple(
            row
            for row in rows
            if row.system_id == target.system_id and row.parameter_set_id == target.parameter_set_id
        )
        dependency_groups = {row.dependency_group_sha256 for row in curve}
        configurations = {row.configuration_sha256 for row in curve}
        model_families = {row.model_family for row in curve}
        if len(dependency_groups) != 1 or len(configurations) != 1 or len(model_families) != 1:
            raise ValueError("a curve cannot mix scientific families or execution configurations")
        references = tuple(row for row in curve if row.field_magnitude_tesla == 0.0)
        if len(references) != 1:
            raise ValueError("each system requires exactly one zero-field reference")
        reference = references[0]
        if reference.absolute_secondary_yield == 0.0:
            raise ValueError("zero-field yield cannot normalize relative effects")
        curve_effects: list[ZeroFieldRelativeEffect] = []
        for row in curve:
            effect = ZeroFieldRelativeEffect(
                system_id=row.system_id,
                parameter_set_id=row.parameter_set_id,
                model_family=row.model_family,
                dependency_group_sha256=row.dependency_group_sha256,
                configuration_sha256=row.configuration_sha256,
                field_magnitude_tesla=row.field_magnitude_tesla,
                target_job_sha256=row.job_sha256,
                reference_job_sha256=reference.job_sha256,
                target_datum_sha256=row.datum_sha256,
                reference_datum_sha256=reference.datum_sha256,
                relative_effect=(row.absolute_secondary_yield - reference.absolute_secondary_yield)
                / reference.absolute_secondary_yield,
            )
            curve_effects.append(effect)
            effects.append(effect)
        by_field = {effect.field_magnitude_tesla: effect for effect in curve_effects}
        yields = tuple(row.absolute_secondary_yield for row in curve)
        summaries.append(
            MaedaCurveSummary(
                system_id=target.system_id,
                parameter_set_id=target.parameter_set_id,
                model_family=reference.model_family,
                dependency_group_sha256=reference.dependency_group_sha256,
                configuration_sha256=reference.configuration_sha256,
                point_count=len(curve),
                zero_field_yield=reference.absolute_secondary_yield,
                minimum_yield=min(yields),
                maximum_yield=max(yields),
                yield_span=max(yields) - min(yields),
                maximum_absolute_relative_effect=max(
                    abs(effect.relative_effect) for effect in curve_effects
                ),
                relative_effect_at_10_microtesla=(
                    by_field[10e-6].relative_effect if 10e-6 in by_field else None
                ),
                relative_effect_at_20_microtesla=(
                    by_field[20e-6].relative_effect if 20e-6 in by_field else None
                ),
                relative_effect_at_50_microtesla=(
                    by_field[50e-6].relative_effect if 50e-6 in by_field else None
                ),
                maximum_achieved_relative_residual=max(
                    row.achieved_relative_residual for row in curve
                ),
                maximum_iteration_count=max(row.iteration_count for row in curve),
            )
        )
    return MaedaSweepQaReport(
        plan_sha256=expected_plan.plan_sha256,
        bundle_sha256=bundle.bundle_sha256,
        rows=rows,
        zero_field_relative_effects=tuple(effects),
        curve_summaries=tuple(summaries),
        _construction_key=_AUTHENTICATED_CONSTRUCTION,
    )


def maeda_sweep_qa_json(report: MaedaSweepQaReport) -> bytes:
    """Serialize a report previously built through the authenticated bundle boundary."""

    payload = _report_payload(report)
    return _canonical_bytes({**payload, "report_sha256": report.report_sha256})


def write_maeda_sweep_qa_report(
    bundle_path: Path,
    *,
    expected_plan: MaedaSweepPlan,
    output_path: Path,
) -> Path:
    """Authenticate a bundle, derive its QA view, and publish it exactly once."""

    if not isinstance(output_path, Path):
        raise ValueError("QA report path must be a pathlib.Path")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"QA report already exists: {output_path}")
    report = build_maeda_sweep_qa_report(bundle_path, expected_plan=expected_plan)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(maeda_sweep_qa_json(report))
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(output_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.link(temporary, output_path, follow_symlinks=False)
            with suppress(OSError):
                os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    with suppress(OSError):
        temporary.unlink(missing_ok=True)
    return output_path
