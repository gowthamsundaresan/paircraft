"""Deterministic plans for the separate hypothetical Maeda sensitivity model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from string import hexdigits
from typing import Any, Final, cast

from paircraft.spin import MaedaProteinSystem, maeda_2012_specification
from paircraft.spin.maeda_sensitivity import (
    HypotheticalSecondaryPairYield,
    SensitivityDimension,
    SensitivityDirection,
    maeda_sensitivity_specification,
    solve_maeda_sensitivity_yield,
)
from paircraft.spin.maeda_sensitivity import (
    MaedaSensitivitySpecification as SpinSensitivitySpecification,
)
from paircraft.sweeps.maeda import IdealFieldPoint

_SCHEMA: Final = "paircraft-maeda-local-sensitivity-plan-v1"
_METHOD: Final = "deterministic_centered_one_at_a_time_log_elasticity"
_EVIDENCE: Final = "assumed_hypothetical_design"
_MEMBERSHIP: Final = "reference_only_not_training"
_CLAIM: Final = (
    "Deterministic local numerical sensitivity of a separately identified hypothetical model "
    "around published Maeda point estimates; not parameter uncertainty, a posterior, biological "
    "variability, mutation or sequence prediction, experimental calibration, training data, or "
    "prospective evidence."
)
_FIELDS: Final = (
    ("b-0-t", 0.0),
    ("b-10-ut", 10e-6),
    ("b-20-ut", 20e-6),
    ("b-50-ut", 50e-6),
    ("b-100-ut", 100e-6),
    ("b-200-ut", 200e-6),
)
_PERTURBED_DIRECTIONS: Final = (
    SensitivityDirection.LOWER,
    SensitivityDirection.HIGHER,
)


def _sha(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return sha256(raw).hexdigest()


def _text(value: str, label: str) -> None:
    if not value.strip():
        raise ValueError(f"{label} must be explicit")


def _digest(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in hexdigits for character in value):
        raise ValueError(f"{label} must be a 64-character hexadecimal digest")


@dataclass(frozen=True, slots=True)
class MaedaSensitivityParent:
    """Exact published parent retained by every hypothetical family."""

    protein_system: MaedaProteinSystem
    parent_model_id: str
    parent_parameter_set_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.protein_system, MaedaProteinSystem):
            raise ValueError("sensitivity parent requires a Maeda protein system")
        baseline = maeda_2012_specification(self.protein_system)
        if (self.parent_model_id, self.parent_parameter_set_sha256) != (
            baseline.model_id,
            baseline.parameter_set_sha256,
        ):
            raise ValueError("sensitivity parent must equal the exact published specification")


def maeda_sensitivity_parent(system: MaedaProteinSystem) -> MaedaSensitivityParent:
    """Resolve one exact published parent without modifying it."""

    baseline = maeda_2012_specification(system)
    return MaedaSensitivityParent(system, baseline.model_id, baseline.parameter_set_sha256)


@dataclass(frozen=True, slots=True)
class MaedaSensitivityFamily:
    """One hypothetical parameter family shared by all its field calculations."""

    parent: MaedaSensitivityParent
    family_id: str
    spin_specification: SpinSensitivitySpecification
    evidence_status: str = _EVIDENCE
    corpus_membership: str = _MEMBERSHIP

    def __post_init__(self) -> None:
        if not isinstance(self.parent, MaedaSensitivityParent):
            raise ValueError("sensitivity family requires a typed parent")
        _text(self.family_id, "sensitivity family ID")
        if not isinstance(self.spin_specification, SpinSensitivitySpecification):
            raise ValueError("sensitivity family requires a hypothetical spin specification")
        if (
            self.spin_specification.protein_system is not self.parent.protein_system
            or self.spin_specification.baseline_parameter_set_sha256
            != self.parent.parent_parameter_set_sha256
        ):
            raise ValueError("sensitivity spin specification contradicts its parent")
        override = self.spin_specification.override
        if override.direction is SensitivityDirection.BASELINE:
            if override.dimension is not SensitivityDimension.FORWARD_RATE:
                raise ValueError("only forward-rate BASELINE is the canonical control")
            expected_id = "baseline"
        else:
            expected_id = f"{override.dimension.value}:{override.direction.value}"
        if self.family_id != expected_id:
            raise ValueError("sensitivity family ID contradicts its spin specification")
        if self.evidence_status != _EVIDENCE or self.corpus_membership != _MEMBERSHIP:
            raise ValueError("sensitivity family claim semantics are unsupported")

    @property
    def parameter_family_sha256(self) -> str:
        """Return the spin layer's complete hypothetical parameter identity."""

        return self.spin_specification.parameter_set_sha256


def _family(
    parent: MaedaSensitivityParent,
    dimension: SensitivityDimension,
    direction: SensitivityDirection,
) -> MaedaSensitivityFamily:
    specification = maeda_sensitivity_specification(parent.protein_system, dimension, direction)
    family_id = (
        "baseline"
        if direction is SensitivityDirection.BASELINE
        else f"{dimension.value}:{direction.value}"
    )
    return MaedaSensitivityFamily(parent, family_id, specification)


def maeda_local_sensitivity_families(
    parent: MaedaSensitivityParent,
) -> tuple[MaedaSensitivityFamily, ...]:
    """Return baseline plus lower/higher OAT families in canonical order."""

    if not isinstance(parent, MaedaSensitivityParent):
        raise ValueError("local sensitivity families require a typed parent")
    return (
        _family(
            parent,
            SensitivityDimension.FORWARD_RATE,
            SensitivityDirection.BASELINE,
        ),
        *(
            _family(parent, dimension, direction)
            for dimension in SensitivityDimension
            for direction in _PERTURBED_DIRECTIONS
        ),
    )


@dataclass(frozen=True, slots=True)
class MaedaSensitivitySweepSpecification:
    """Complete deterministic OAT design without generated results."""

    design_id: str
    parents: tuple[MaedaSensitivityParent, ...]
    field_points: tuple[IdealFieldPoint, ...]
    dimensions: tuple[SensitivityDimension, ...]
    log_scale_deltas: tuple[float, ...]
    claim_boundary: str
    design_method: str = _METHOD
    random_seed: None = None
    evidence_status: str = _EVIDENCE
    corpus_membership: str = _MEMBERSHIP
    schema_version: str = _SCHEMA

    def __post_init__(self) -> None:
        _text(self.design_id, "sensitivity design ID")
        expected_parents = tuple(
            maeda_sensitivity_parent(system)
            for system in sorted(MaedaProteinSystem, key=lambda item: item.value)
        )
        expected_fields = tuple(IdealFieldPoint(name, value) for name, value in _FIELDS)
        expected_deltas = tuple(
            direction.log_scale_delta
            for direction in (
                SensitivityDirection.LOWER,
                SensitivityDirection.BASELINE,
                SensitivityDirection.HIGHER,
            )
        )
        if self.schema_version != _SCHEMA:
            raise ValueError("sensitivity design schema is unsupported")
        if self.parents != expected_parents:
            raise ValueError("sensitivity design requires both exact published parents")
        if self.field_points != expected_fields:
            raise ValueError("sensitivity design requires the exact six low-field points")
        if self.dimensions != tuple(SensitivityDimension):
            raise ValueError("sensitivity design requires all eight canonical dimensions")
        if self.log_scale_deltas != expected_deltas:
            raise ValueError("sensitivity design requires exact centered log-scale offsets")
        if self.design_method != _METHOD or self.random_seed is not None:
            raise ValueError("sensitivity design must remain deterministic centered OAT")
        if (
            self.claim_boundary != _CLAIM
            or self.evidence_status != _EVIDENCE
            or self.corpus_membership != _MEMBERSHIP
        ):
            raise ValueError("sensitivity design claim semantics are unsupported")

    @property
    def specification_sha256(self) -> str:
        """Hash the complete claim-bounded design."""

        return _sha(asdict(self))


@dataclass(frozen=True, slots=True)
class MaedaSensitivityJob:
    """One content-addressed hypothetical family/field calculation."""

    design_id: str
    specification_sha256: str
    family: MaedaSensitivityFamily
    field_point: IdealFieldPoint
    execution_input: SpinSensitivitySpecification

    def __post_init__(self) -> None:
        _text(self.design_id, "sensitivity job design ID")
        _digest(self.specification_sha256, "sensitivity job specification sha256")
        if not isinstance(self.family, MaedaSensitivityFamily):
            raise ValueError("sensitivity job requires a typed family")
        if not isinstance(self.field_point, IdealFieldPoint):
            raise ValueError("sensitivity job requires an ideal field point")
        if self.execution_input != self.family.spin_specification:
            raise ValueError("sensitivity execution input contradicts its family")

    @property
    def parameter_family_sha256(self) -> str:
        """Identify the hypothetical parameter family independently of field."""

        return self.family.parameter_family_sha256

    @property
    def design_dependency_sha256(self) -> str:
        """Group all baseline/OAT families sharing one parent and design policy."""

        return _sha(
            {
                "design_id": self.design_id,
                "specification_sha256": self.specification_sha256,
                "design_method": _METHOD,
                "parent_parameter_set_sha256": (self.family.parent.parent_parameter_set_sha256),
                "model_id": self.execution_input.model_id,
            }
        )

    @property
    def job_sha256(self) -> str:
        """Hash one exact family/field job without a generated result."""

        return _sha(asdict(self))


@dataclass(frozen=True, slots=True)
class MaedaSensitivityPlan:
    """Canonical 2-system x 17-family x 6-field materialization."""

    specification: MaedaSensitivitySweepSpecification
    jobs: tuple[MaedaSensitivityJob, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.specification, MaedaSensitivitySweepSpecification):
            raise ValueError("sensitivity plan requires a typed specification")
        if not isinstance(self.jobs, tuple) or any(
            not isinstance(job, MaedaSensitivityJob) for job in self.jobs
        ):
            raise ValueError("sensitivity plan jobs must be an immutable typed tuple")
        if self.jobs != _materialized_jobs(self.specification):
            raise ValueError("sensitivity plan jobs must equal the canonical materialization")

    @property
    def plan_sha256(self) -> str:
        """Hash the design and exact ordered job identities."""

        return _sha(
            {
                "specification_sha256": self.specification.specification_sha256,
                "job_sha256": [job.job_sha256 for job in self.jobs],
            }
        )


def default_maeda_local_sensitivity() -> MaedaSensitivitySweepSpecification:
    """Return the fixed deterministic v1 local-sensitivity design."""

    return MaedaSensitivitySweepSpecification(
        design_id="maeda-local-log-elasticity-v1",
        parents=tuple(
            maeda_sensitivity_parent(system)
            for system in sorted(MaedaProteinSystem, key=lambda item: item.value)
        ),
        field_points=tuple(IdealFieldPoint(name, value) for name, value in _FIELDS),
        dimensions=tuple(SensitivityDimension),
        log_scale_deltas=tuple(
            direction.log_scale_delta
            for direction in (
                SensitivityDirection.LOWER,
                SensitivityDirection.BASELINE,
                SensitivityDirection.HIGHER,
            )
        ),
        claim_boundary=_CLAIM,
    )


def _materialized_jobs(
    specification: MaedaSensitivitySweepSpecification,
) -> tuple[MaedaSensitivityJob, ...]:
    return tuple(
        MaedaSensitivityJob(
            design_id=specification.design_id,
            specification_sha256=specification.specification_sha256,
            family=family,
            field_point=field,
            execution_input=family.spin_specification,
        )
        for parent in specification.parents
        for family in maeda_local_sensitivity_families(parent)
        for field in specification.field_points
    )


def materialize_maeda_local_sensitivity(
    specification: MaedaSensitivitySweepSpecification,
) -> MaedaSensitivityPlan:
    """Materialize the exact deterministic family/field product."""

    if not isinstance(specification, MaedaSensitivitySweepSpecification):
        raise ValueError("sensitivity materialization requires a typed specification")
    return MaedaSensitivityPlan(specification, _materialized_jobs(specification))


def _require_plan_job(plan: MaedaSensitivityPlan, job: MaedaSensitivityJob) -> None:
    if not isinstance(plan, MaedaSensitivityPlan) or not isinstance(job, MaedaSensitivityJob):
        raise ValueError("sensitivity execution requires a typed plan and job")
    if job.specification_sha256 != plan.specification.specification_sha256 or job not in plan.jobs:
        raise ValueError("sensitivity job is not an exact member of its parent plan")


def validate_maeda_sensitivity_job_result(
    plan: MaedaSensitivityPlan,
    job: MaedaSensitivityJob,
    result: HypotheticalSecondaryPairYield,
) -> None:
    """Bind one spin-layer result to its exact planned family and field."""

    _require_plan_job(plan, job)
    if not isinstance(result, HypotheticalSecondaryPairYield):
        raise ValueError("sensitivity result must be a HypotheticalSecondaryPairYield")
    if result.specification != job.execution_input:
        raise ValueError("sensitivity result specification contradicts its family")
    if result.field_tesla != job.field_point.vector_tesla:
        raise ValueError("sensitivity result field contradicts its planned field")
    if result.parameter_set_sha256 != job.parameter_family_sha256:
        raise ValueError("sensitivity result parameter identity contradicts its family")


def execute_maeda_sensitivity_job(
    plan: MaedaSensitivityPlan,
    job: MaedaSensitivityJob,
) -> HypotheticalSecondaryPairYield:
    """Execute one exact planned job through the hypothetical spin model."""

    _require_plan_job(plan, job)
    result = solve_maeda_sensitivity_yield(job.execution_input, job.field_point.vector_tesla)
    validate_maeda_sensitivity_job_result(plan, job, result)
    return result


def maeda_sensitivity_plan_document(plan: MaedaSensitivityPlan) -> dict[str, object]:
    """Return canonical design metadata without generated results."""

    if not isinstance(plan, MaedaSensitivityPlan):
        raise ValueError("sensitivity serialization requires a typed plan")
    document: dict[str, Any] = {
        "schema_version": plan.specification.schema_version,
        "specification": asdict(plan.specification),
        "jobs": [
            {
                **asdict(job),
                "parameter_family_sha256": job.parameter_family_sha256,
                "design_dependency_sha256": job.design_dependency_sha256,
                "job_sha256": job.job_sha256,
            }
            for job in plan.jobs
        ],
        "hashes": {
            "specification_sha256": plan.specification.specification_sha256,
            "plan_sha256": plan.plan_sha256,
        },
    }
    normalized = json.loads(
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    if not isinstance(normalized, dict):
        raise AssertionError("sensitivity plan normalization did not produce an object")
    return cast(dict[str, object], normalized)


def maeda_sensitivity_plan_json(plan: MaedaSensitivityPlan) -> bytes:
    """Return canonical compact JSON; no persistence policy is implied."""

    return json.dumps(
        maeda_sensitivity_plan_document(plan),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
