"""Deterministic field-sweep plans for the exact published Maeda systems."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from string import hexdigits
from typing import Any, cast

from paircraft.oracle.maeda2012 import (
    Maeda2012SimulatorOracle,
    maeda_2012_query,
    maeda_2012_system,
)
from paircraft.oracle.model import (
    ObservableKind,
    OracleIdentity,
    OraclePrediction,
    OracleQuery,
    SpinModelFamily,
)
from paircraft.oracle.simulation import SimulationDatum
from paircraft.spin import MaedaProteinSystem

_MAX_FIELD_TESLA = 0.05
_PLAN_SCHEMA_VERSION = "paircraft-maeda-sweep-plan-v1"
_MODEL_FAMILY = SpinModelFamily.HABERKORN_STATIC_PAIR_WITH_ST_DEPHASING
_DEFAULT_FIELDS: tuple[tuple[str, float], ...] = (
    ("b-0-t", 0.0),
    ("b-10-ut", 10e-6),
    ("b-20-ut", 20e-6),
    ("b-50-ut", 50e-6),
    ("b-100-ut", 100e-6),
    ("b-200-ut", 200e-6),
    ("b-500-ut", 500e-6),
    ("b-1-mt", 1e-3),
    ("b-2-mt", 2e-3),
    ("b-5-mt", 5e-3),
    ("b-10-mt", 10e-3),
    ("b-20-mt", 20e-3),
    ("b-30-mt", 30e-3),
    ("b-50-mt", 50e-3),
)


def _require_text(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be explicit")


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return sha256(encoded).hexdigest()


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in hexdigits for character in value):
        raise ValueError(f"{name} must be a 64-character hexadecimal digest")


class ParameterSetPolicy(StrEnum):
    """Which internal model parameters a sweep may use."""

    PUBLISHED_POINT_ESTIMATES_ONLY = "published_point_estimates_only"


class SweepSamplingMethod(StrEnum):
    """How sweep jobs are enumerated."""

    ENUMERATED_CARTESIAN_PRODUCT = "enumerated_cartesian_product"


class RandomnessStatus(StrEnum):
    """Randomness policy of a scientific sweep design."""

    NOT_APPLICABLE_DETERMINISTIC = "not_applicable_deterministic"


@dataclass(frozen=True, slots=True)
class MaedaSweepTarget:
    """Exact identity of one published point-estimate system."""

    protein_system: MaedaProteinSystem
    system_id: str
    parameter_set_id: str
    parameter_set_sha256: str
    parameter_catalog_version: str
    parameter_catalog_sha256: str
    model_family: SpinModelFamily

    def __post_init__(self) -> None:
        if not isinstance(self.protein_system, MaedaProteinSystem):
            raise ValueError("sweep protein system must be a MaedaProteinSystem")
        if self.model_family is not _MODEL_FAMILY:
            raise ValueError("Maeda sweep target has the wrong physical model family")
        expected = maeda_2012_system(self.protein_system)
        actual = (
            self.system_id,
            self.parameter_set_id,
            self.parameter_set_sha256,
            self.parameter_catalog_version,
            self.parameter_catalog_sha256,
        )
        required = (
            expected.system_id,
            expected.parameter_set_id,
            expected.parameter_set_sha256,
            expected.parameter_catalog_version,
            expected.parameter_catalog_sha256,
        )
        if actual != required:
            raise ValueError("sweep target must resolve to the exact published Maeda catalog")

    @property
    def scientific_family_sha256(self) -> str:
        """Hash the scientific parameter family independently of field and execution code."""

        return _canonical_sha256(
            {
                "system_id": self.system_id,
                "parameter_set_id": self.parameter_set_id,
                "parameter_set_sha256": self.parameter_set_sha256,
                "model_family": self.model_family,
            }
        )


def maeda_sweep_target(system: MaedaProteinSystem) -> MaedaSweepTarget:
    """Resolve one exact published Maeda catalog target."""

    catalog = maeda_2012_system(system)
    return MaedaSweepTarget(
        protein_system=system,
        system_id=catalog.system_id,
        parameter_set_id=catalog.parameter_set_id,
        parameter_set_sha256=catalog.parameter_set_sha256,
        parameter_catalog_version=catalog.parameter_catalog_version,
        parameter_catalog_sha256=catalog.parameter_catalog_sha256,
        model_family=_MODEL_FAMILY,
    )


@dataclass(frozen=True, slots=True)
class IdealFieldPoint:
    """One ideal isotropic field conventionally represented along the z axis."""

    point_id: str
    magnitude_tesla: float
    coordinate_frame: str = "maeda-2012-isotropic-lab-frame"
    field_intent: str = "ideal_simulation_target"

    def __post_init__(self) -> None:
        _require_text(self.point_id, "field point ID")
        if isinstance(self.magnitude_tesla, bool) or not isinstance(
            self.magnitude_tesla, int | float
        ):
            raise ValueError("field magnitude must be a real scalar, not a boolean")
        magnitude = float(self.magnitude_tesla)
        if not isfinite(magnitude) or not 0 <= magnitude <= _MAX_FIELD_TESLA:
            raise ValueError("field magnitude must be finite and lie in the 0-0.05 T domain")
        object.__setattr__(self, "magnitude_tesla", 0.0 if magnitude == 0.0 else magnitude)
        if self.coordinate_frame != "maeda-2012-isotropic-lab-frame":
            raise ValueError("field point must use the exact Maeda isotropic lab frame")
        if self.field_intent != "ideal_simulation_target":
            raise ValueError("field point must remain an ideal simulation target")

    @property
    def vector_tesla(self) -> tuple[float, float, float]:
        """Return the conventional z-directed ideal field vector in tesla."""

        return (0.0, 0.0, self.magnitude_tesla)


@dataclass(frozen=True, slots=True)
class MaedaSweepSpecification:
    """Scientific design for an exact deterministic Maeda field sweep."""

    sweep_id: str
    targets: tuple[MaedaSweepTarget, ...]
    field_points: tuple[IdealFieldPoint, ...]
    claim_boundary: str
    observable: ObservableKind = ObservableKind.SECONDARY_RADICAL_YIELD
    parameter_set_policy: ParameterSetPolicy = ParameterSetPolicy.PUBLISHED_POINT_ESTIMATES_ONLY
    sampling_method: SweepSamplingMethod = SweepSamplingMethod.ENUMERATED_CARTESIAN_PRODUCT
    randomness_status: RandomnessStatus = RandomnessStatus.NOT_APPLICABLE_DETERMINISTIC
    random_seed: None = None
    schema_version: str = _PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.sweep_id, "sweep ID")
        _require_text(self.claim_boundary, "sweep claim boundary")
        if self.schema_version != _PLAN_SCHEMA_VERSION:
            raise ValueError("Maeda sweep schema version is unsupported")
        if not isinstance(self.targets, tuple) or not self.targets:
            raise ValueError("sweep targets must be a nonempty immutable tuple")
        if not isinstance(self.field_points, tuple) or not self.field_points:
            raise ValueError("field points must be a nonempty immutable tuple")
        if any(not isinstance(target, MaedaSweepTarget) for target in self.targets):
            raise ValueError("sweep targets must be MaedaSweepTarget values")
        if any(not isinstance(point, IdealFieldPoint) for point in self.field_points):
            raise ValueError("field points must be IdealFieldPoint values")
        targets = tuple(sorted(self.targets, key=lambda item: item.protein_system.value))
        points = tuple(
            sorted(self.field_points, key=lambda item: (item.magnitude_tesla, item.point_id))
        )
        if len({target.protein_system for target in targets}) != len(targets):
            raise ValueError("sweep target systems must be unique")
        if len({point.point_id for point in points}) != len(points):
            raise ValueError("field point IDs must be unique")
        if len({point.magnitude_tesla for point in points}) != len(points):
            raise ValueError("field magnitudes must be unique")
        if sum(point.magnitude_tesla == 0.0 for point in points) != 1:
            raise ValueError("a sweep requires exactly one zero-field reference point")
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "field_points", points)
        if self.observable is not ObservableKind.SECONDARY_RADICAL_YIELD:
            raise ValueError("Maeda sweep supports only absolute secondary-pair yield")
        if self.parameter_set_policy is not ParameterSetPolicy.PUBLISHED_POINT_ESTIMATES_ONLY:
            raise ValueError("Maeda sweep cannot perturb published parameter sets")
        if self.sampling_method is not SweepSamplingMethod.ENUMERATED_CARTESIAN_PRODUCT:
            raise ValueError("Maeda sweep must use the enumerated Cartesian product")
        if self.randomness_status is not RandomnessStatus.NOT_APPLICABLE_DETERMINISTIC:
            raise ValueError("Maeda sweep does not permit random sampling")
        if self.random_seed is not None:
            raise ValueError("deterministic Maeda sweep cannot carry a random seed")

    @property
    def specification_sha256(self) -> str:
        """Hash the canonical scientific sweep design without results."""

        return _canonical_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class MaedaSweepJob:
    """One content-addressed field calculation in a Maeda sweep plan."""

    sweep_id: str
    specification_sha256: str
    target: MaedaSweepTarget
    field_point: IdealFieldPoint
    query_id: str
    query_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.sweep_id, "job sweep ID")
        _require_text(self.query_id, "job query ID")
        _require_sha256(self.specification_sha256, "job specification sha256")
        _require_sha256(self.query_sha256, "job query sha256")
        object.__setattr__(self, "specification_sha256", self.specification_sha256.lower())
        object.__setattr__(self, "query_sha256", self.query_sha256.lower())
        if not isinstance(self.target, MaedaSweepTarget):
            raise ValueError("job target must be a MaedaSweepTarget")
        if not isinstance(self.field_point, IdealFieldPoint):
            raise ValueError("job field point must be an IdealFieldPoint")
        expected_id = _query_id(self.sweep_id, self.target, self.field_point)
        if self.query_id != expected_id:
            raise ValueError("job query ID contradicts its sweep target and field")
        expected_query = _job_query(self.target, self.field_point, self.query_id)
        if self.query_sha256 != _canonical_sha256(asdict(expected_query)):
            raise ValueError("job query hash contradicts the exact resolved query")

    @property
    def scientific_family_sha256(self) -> str:
        """Return the parameter-family grouping shared by every field of this target."""

        return self.target.scientific_family_sha256

    @property
    def job_sha256(self) -> str:
        """Hash this exact field job without any numerical result."""

        return _canonical_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class MaedaSweepPlan:
    """Canonical Cartesian materialization of one scientific sweep specification."""

    specification: MaedaSweepSpecification
    jobs: tuple[MaedaSweepJob, ...]
    execution_oracle: OracleIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.specification, MaedaSweepSpecification):
            raise ValueError("plan specification must be a MaedaSweepSpecification")
        if not isinstance(self.execution_oracle, OracleIdentity):
            raise ValueError("plan execution oracle must be an OracleIdentity")
        if (
            self.execution_oracle.parameter_catalog_version
            != self.specification.targets[0].parameter_catalog_version
            or self.execution_oracle.parameter_catalog_sha256
            != self.specification.targets[0].parameter_catalog_sha256
            or _MODEL_FAMILY not in self.execution_oracle.supported_model_families
            or ObservableKind.SECONDARY_RADICAL_YIELD
            not in self.execution_oracle.supported_observables
        ):
            raise ValueError("plan oracle identity contradicts the sweep catalog or observable")
        if not isinstance(self.jobs, tuple) or any(
            not isinstance(job, MaedaSweepJob) for job in self.jobs
        ):
            raise ValueError("plan jobs must be an immutable tuple of MaedaSweepJob values")
        jobs = tuple(
            sorted(
                self.jobs,
                key=lambda item: (
                    item.target.protein_system.value,
                    item.field_point.magnitude_tesla,
                ),
            )
        )
        object.__setattr__(self, "jobs", jobs)
        expected = _materialized_jobs(self.specification)
        if jobs != expected:
            raise ValueError("plan jobs must equal the exact target/field Cartesian product")

    @property
    def plan_sha256(self) -> str:
        """Hash the scientific specification and its canonical job identities."""

        return _canonical_sha256(
            {
                "specification_sha256": self.specification.specification_sha256,
                "execution_oracle": asdict(self.execution_oracle),
                "job_sha256": [job.job_sha256 for job in self.jobs],
            }
        )


def _query_id(sweep_id: str, target: MaedaSweepTarget, point: IdealFieldPoint) -> str:
    return f"{sweep_id}:{target.protein_system.value}:{point.point_id}"


def _job_query(target: MaedaSweepTarget, point: IdealFieldPoint, query_id: str) -> OracleQuery:
    return maeda_2012_query(
        target.protein_system,
        point.vector_tesla,
        observable=ObservableKind.SECONDARY_RADICAL_YIELD,
        query_id=query_id,
    )


def _materialized_jobs(specification: MaedaSweepSpecification) -> tuple[MaedaSweepJob, ...]:
    jobs: list[MaedaSweepJob] = []
    for target in specification.targets:
        for point in specification.field_points:
            query_id = _query_id(specification.sweep_id, target, point)
            query = _job_query(target, point, query_id)
            jobs.append(
                MaedaSweepJob(
                    sweep_id=specification.sweep_id,
                    specification_sha256=specification.specification_sha256,
                    target=target,
                    field_point=point,
                    query_id=query_id,
                    query_sha256=_canonical_sha256(asdict(query)),
                )
            )
    return tuple(jobs)


def materialize_maeda_sweep(specification: MaedaSweepSpecification) -> MaedaSweepPlan:
    """Materialize the exact deterministic target/field Cartesian product."""

    return MaedaSweepPlan(
        specification,
        _materialized_jobs(specification),
        Maeda2012SimulatorOracle().identity,
    )


def default_maeda_reference_sweep() -> MaedaSweepSpecification:
    """Return the two-system, 14-field reference sweep without executing it."""

    return MaedaSweepSpecification(
        sweep_id="maeda-2012-reference-field-sweep-v1",
        targets=tuple(maeda_sweep_target(system) for system in MaedaProteinSystem),
        field_points=tuple(IdealFieldPoint(point_id, field) for point_id, field in _DEFAULT_FIELDS),
        claim_boundary=(
            "Deterministic field sensitivity of two exact published point-estimate reduced models; "
            "not parameter uncertainty, biological variability, mutation or sequence prediction, "
            "MagLOV evidence, surrogate training data, or experimental low-field validation."
        ),
    )


def _require_plan_job(plan: MaedaSweepPlan, job: MaedaSweepJob) -> None:
    if not isinstance(plan, MaedaSweepPlan) or not isinstance(job, MaedaSweepJob):
        raise ValueError("sweep operation requires a typed plan and job")
    if job.specification_sha256 != plan.specification.specification_sha256 or job not in plan.jobs:
        raise ValueError("sweep job is not an exact member of the parent plan")


def validate_maeda_job_result(
    plan: MaedaSweepPlan,
    job: MaedaSweepJob,
    datum: SimulationDatum,
) -> None:
    """Authenticate a result before treating a previously executed job as complete."""

    _require_plan_job(plan, job)
    if not isinstance(datum, SimulationDatum):
        raise ValueError("job-result validation requires a SimulationDatum")
    prediction = datum.prediction
    if prediction.oracle != plan.execution_oracle:
        raise ValueError("simulation datum oracle identity contradicts the parent plan")
    query = prediction.query
    if datum.query_sha256 != job.query_sha256 or query.query_id != job.query_id:
        raise ValueError("simulation datum does not match the exact sweep query")
    target = job.target
    system = query.system
    actual_system = (
        system.system_id,
        system.parameter_set_id,
        system.parameter_set_sha256,
        system.parameter_catalog_version,
        system.parameter_catalog_sha256,
        query.model_family,
    )
    expected_system = (
        target.system_id,
        target.parameter_set_id,
        target.parameter_set_sha256,
        target.parameter_catalog_version,
        target.parameter_catalog_sha256,
        target.model_family,
    )
    if actual_system != expected_system:
        raise ValueError("simulation datum contradicts the sweep target catalog identity")
    if query.observable is not ObservableKind.SECONDARY_RADICAL_YIELD:
        raise ValueError("sweep result must be an absolute secondary-pair yield")
    if query.target_protocol.field.components_tesla != job.field_point.vector_tesla:
        raise ValueError("simulation datum field contradicts the sweep job")
    dependency_payload = {
        "system_id": system.system_id,
        "parameter_set_id": system.parameter_set_id,
        "parameter_set_sha256": system.parameter_set_sha256,
        "parameter_catalog_version": system.parameter_catalog_version,
        "parameter_catalog_sha256": system.parameter_catalog_sha256,
        "model_family": query.model_family,
        "oracle_artifact_sha256": plan.execution_oracle.artifact_sha256,
    }
    if datum.dependency_group_sha256 != _canonical_sha256(dependency_payload):
        raise ValueError("simulation datum dependency group is not authentic")


def execute_maeda_sweep_job(
    plan: MaedaSweepPlan,
    job: MaedaSweepJob,
    oracle: Maeda2012SimulatorOracle | None = None,
) -> SimulationDatum:
    """Execute one exact sweep job through the provenance-bearing simulator oracle."""

    _require_plan_job(plan, job)
    executor = Maeda2012SimulatorOracle() if oracle is None else oracle
    if executor.identity != plan.execution_oracle:
        raise ValueError("execution oracle identity contradicts the parent sweep plan")
    query = _job_query(job.target, job.field_point, job.query_id)
    if _canonical_sha256(asdict(query)) != job.query_sha256:
        raise ValueError("sweep job query authentication failed")
    result = executor.predict(query)
    if not isinstance(result, OraclePrediction) or result.execution is None:
        raise ArithmeticError("Maeda sweep oracle did not return an executable prediction")
    datum = SimulationDatum(result, result.execution)
    validate_maeda_job_result(plan, job, datum)
    return datum


def maeda_sweep_plan_document(plan: MaedaSweepPlan) -> dict[str, object]:
    """Return a canonical JSON-compatible plan without numerical results."""

    document: dict[str, Any] = {
        "schema_version": plan.specification.schema_version,
        "specification": asdict(plan.specification),
        "execution_oracle": asdict(plan.execution_oracle),
        "jobs": [
            {
                **asdict(job),
                "scientific_family_sha256": job.scientific_family_sha256,
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
        raise AssertionError("Maeda sweep plan normalization did not produce an object")
    return cast(dict[str, object], normalized)


def maeda_sweep_plan_json(plan: MaedaSweepPlan) -> bytes:
    """Return canonical compact plan JSON bytes; no persistence policy is implied."""

    return json.dumps(
        maeda_sweep_plan_document(plan),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
