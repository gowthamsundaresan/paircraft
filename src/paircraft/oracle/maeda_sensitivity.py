"""Simulator-only oracle for the predeclared hypothetical Maeda OAT design."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from typing import Final

import numpy as np
import scipy

from paircraft import __version__
from paircraft.oracle.maeda2012 import maeda_2012_protocol, maeda_2012_system
from paircraft.oracle.model import (
    AbstentionReason,
    ApplicabilityAssessment,
    ApplicabilityStatus,
    CalibrationTier,
    ExperimentProtocol,
    LineageArtifact,
    NumericalExecution,
    ObservableKind,
    OracleAbstention,
    OracleIdentity,
    OraclePrediction,
    OracleQuery,
    OracleResult,
    PredictiveDistribution,
    RadicalPairSystemRef,
    ResolvedExperimentSetup,
    SolverSetting,
    SpinModelFamily,
    SupportBound,
)
from paircraft.oracle.simulation import SimulationDatum
from paircraft.spin import Maeda2012Specification
from paircraft.spin.maeda2012 import PublishedParameter
from paircraft.spin.maeda_sensitivity import HypotheticalSecondaryPairYield
from paircraft.sweeps.maeda_sensitivity import (
    MaedaSensitivityFamily,
    MaedaSensitivityJob,
    MaedaSensitivityPlan,
    default_maeda_local_sensitivity,
    execute_maeda_sensitivity_job,
    materialize_maeda_local_sensitivity,
)

_QUERY_SCHEMA: Final = "paircraft-maeda-sensitivity-query-v1"
_CATALOG_VERSION: Final = "paircraft-maeda-hypothetical-oat-catalog-v1"
_MODEL_FAMILY: Final = SpinModelFamily.HABERKORN_STATIC_PAIR_WITH_ST_DEPHASING
_TRAINING_MANIFEST_ID: Final = "none-simulator-only-hypothetical-design"
_TRAINING_MANIFEST_SHA256: Final = sha256(b"").hexdigest()
_CLAIM_BOUNDARY: Final = (
    "simulator-only result for one predeclared hypothetical OAT design point; not parameter "
    "uncertainty, calibration, biological variability, mutation or sequence prediction, "
    "training data, prospective evidence, or sub-Earth-field validation"
)


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return sha256(encoded).hexdigest()


def _default_plan() -> MaedaSensitivityPlan:
    return materialize_maeda_local_sensitivity(default_maeda_local_sensitivity())


def _system_id(family: MaedaSensitivityFamily) -> str:
    family_slug = family.family_id.replace(":", "--")
    return f"maeda-hypothetical-oat-{family.parent.protein_system.value}-{family_slug}"


def _catalog_sha256(plan: MaedaSensitivityPlan) -> str:
    families = _families(plan)
    return _canonical_sha256(
        {
            "catalog_version": _CATALOG_VERSION,
            "plan_sha256": plan.plan_sha256,
            "systems": {_system_id(family): family.parameter_family_sha256 for family in families},
        }
    )


def _artifact_sha256(plan: MaedaSensitivityPlan) -> str:
    digest = sha256()
    for path in (
        Path(__file__),
        Path(__file__).with_name("model.py"),
        Path(__file__).parents[1] / "spin" / "maeda_sensitivity.py",
        Path(__file__).parents[1] / "spin" / "_maeda_kernel.py",
        Path(__file__).parents[1] / "spin" / "maeda2012.py",
        Path(__file__).with_name("maeda2012.py"),
        Path(__file__).parents[1] / "sweeps" / "maeda_sensitivity.py",
    ):
        digest.update(path.read_bytes())
    digest.update(plan.plan_sha256.encode())
    digest.update(_CATALOG_VERSION.encode())
    return digest.hexdigest()


def _families(plan: MaedaSensitivityPlan) -> tuple[MaedaSensitivityFamily, ...]:
    ordered: dict[tuple[str, str], MaedaSensitivityFamily] = {}
    for job in plan.jobs:
        key = (job.family.parent.protein_system.value, job.family.family_id)
        ordered.setdefault(key, job.family)
    return tuple(ordered.values())


def maeda_sensitivity_system(
    plan: MaedaSensitivityPlan,
    family: MaedaSensitivityFamily,
) -> RadicalPairSystemRef:
    """Return the distinct catalog identity of one exact hypothetical family."""

    if not isinstance(plan, MaedaSensitivityPlan) or family not in _families(plan):
        raise ValueError("sensitivity system requires an exact family from its plan")
    return _sensitivity_system(plan, family, _catalog_sha256(plan))


def _sensitivity_system(
    plan: MaedaSensitivityPlan,
    family: MaedaSensitivityFamily,
    catalog_sha256: str,
) -> RadicalPairSystemRef:
    baseline = maeda_2012_system(family.parent.protein_system)
    return RadicalPairSystemRef(
        system_id=_system_id(family),
        radical_one=baseline.radical_one,
        radical_two=baseline.radical_two,
        initial_spin_state=baseline.initial_spin_state,
        parameter_set_id=f"hypothetical-oat:{family.family_id}",
        parameter_set_sha256=family.parameter_family_sha256,
        parameter_catalog_version=_CATALOG_VERSION,
        parameter_catalog_sha256=catalog_sha256,
    )


def _protocol(job: MaedaSensitivityJob) -> ExperimentProtocol:
    inherited = maeda_2012_protocol(
        job.family.parent.protein_system,
        job.field_point.vector_tesla,
    )
    return replace(
        inherited,
        reaction_model_id="paircraft-maeda-hypothetical-oat-reaction-v1",
        relaxation_model_id="paircraft-maeda-hypothetical-oat-st-dephasing-v1",
    )


def maeda_sensitivity_query(
    plan: MaedaSensitivityPlan,
    job: MaedaSensitivityJob,
    *,
    query_id: str | None = None,
) -> OracleQuery:
    """Construct a query only for one exact predeclared plan job."""

    if not isinstance(plan, MaedaSensitivityPlan) or job not in plan.jobs:
        raise ValueError("sensitivity query requires an exact job from its parent plan")
    return OracleQuery(
        schema_version=_QUERY_SCHEMA,
        query_id=(
            query_id if query_id is not None else f"{plan.specification.design_id}:{job.job_sha256}"
        ),
        system=maeda_sensitivity_system(plan, job.family),
        target_protocol=_protocol(job),
        reference_protocol=None,
        observable=ObservableKind.SECONDARY_RADICAL_YIELD,
        model_family=_MODEL_FAMILY,
    )


def _oracle_identity(plan: MaedaSensitivityPlan) -> OracleIdentity:
    return OracleIdentity(
        oracle_id="paircraft-maeda-hypothetical-oat-simulator-oracle",
        oracle_version="1",
        artifact_sha256=_artifact_sha256(plan),
        code_version=__version__,
        training_manifest_id=_TRAINING_MANIFEST_ID,
        training_manifest_sha256=_TRAINING_MANIFEST_SHA256,
        parameter_catalog_version=_CATALOG_VERSION,
        parameter_catalog_sha256=_catalog_sha256(plan),
        supported_model_families=(_MODEL_FAMILY,),
        supported_observables=(ObservableKind.SECONDARY_RADICAL_YIELD,),
        calibration_tier=CalibrationTier.SIMULATOR_ONLY,
    )


class MaedaSensitivitySimulatorOracle:
    """Execute only exact members of one immutable sensitivity plan."""

    def __init__(self, plan: MaedaSensitivityPlan | None = None) -> None:
        resolved = _default_plan() if plan is None else plan
        if not isinstance(resolved, MaedaSensitivityPlan):
            raise ValueError("sensitivity oracle requires a MaedaSensitivityPlan")
        self._plan = resolved
        self._jobs_by_system: dict[str, tuple[MaedaSensitivityJob, ...]] = {}
        self._systems: dict[str, RadicalPairSystemRef] = {}
        catalog_sha256 = _catalog_sha256(resolved)
        for family in _families(resolved):
            identifier = _system_id(family)
            self._jobs_by_system[identifier] = tuple(
                job for job in resolved.jobs if job.family == family
            )
            self._systems[identifier] = _sensitivity_system(resolved, family, catalog_sha256)
        self._identity = _oracle_identity(resolved)

    @property
    def plan(self) -> MaedaSensitivityPlan:
        """Return the exact immutable plan served by this oracle."""

        return self._plan

    @property
    def identity(self) -> OracleIdentity:
        """Return the distinct content-addressed hypothetical oracle identity."""

        return self._identity

    def predict(self, query: OracleQuery) -> OracleResult:
        """Return one exact design result or abstain without interpreting arbitrary overrides."""

        job, unsupported = self._resolve_job(query)
        if job is None:
            reason = _abstention_reason(unsupported)
            return self._abstain(query, reason, unsupported)
        try:
            result = execute_maeda_sensitivity_job(self._plan, job)
        except (ArithmeticError, ValueError):
            return self._abstain(query, AbstentionReason.NUMERICAL_FAILURE, ("solver",))
        execution = _execution(result)
        return OraclePrediction(
            query=query,
            resolved_setup=ResolvedExperimentSetup(query.system, query.target_protocol, None),
            oracle=self.identity,
            executed_model_family=_MODEL_FAMILY,
            distribution=_distribution(result),
            applicability=_applicability(),
            lineage=_lineage(self._plan, job, result, execution),
            claim_boundary=_CLAIM_BOUNDARY,
            execution=execution,
            executed_model_id=result.model_id,
        )

    def _query_for_exact_job(self, job: MaedaSensitivityJob) -> OracleQuery:
        if job not in self._plan.jobs:
            raise ValueError("sensitivity query requires an exact oracle-plan job")
        return OracleQuery(
            schema_version=_QUERY_SCHEMA,
            query_id=f"{self._plan.specification.design_id}:{job.job_sha256}",
            system=self._systems[_system_id(job.family)],
            target_protocol=_protocol(job),
            reference_protocol=None,
            observable=ObservableKind.SECONDARY_RADICAL_YIELD,
            model_family=_MODEL_FAMILY,
        )

    def _resolve_job(
        self, query: OracleQuery
    ) -> tuple[MaedaSensitivityJob | None, tuple[str, ...]]:
        unsupported: list[str] = []
        candidates = self._jobs_by_system.get(query.system.system_id)
        if query.schema_version != _QUERY_SCHEMA:
            unsupported.append("schema_version")
        if candidates is None:
            unsupported.append("system")
        else:
            expected_system = self._systems[query.system.system_id]
            if (
                query.system.parameter_set_id != expected_system.parameter_set_id
                or query.system.parameter_set_sha256 != expected_system.parameter_set_sha256
            ):
                unsupported.append("parameter_set")
            if query.system != expected_system:
                unsupported.append("system")
        if query.observable is not ObservableKind.SECONDARY_RADICAL_YIELD:
            unsupported.append("observable")
        if query.model_family is not _MODEL_FAMILY:
            unsupported.append("model_family")
        if query.reference_protocol is not None:
            unsupported.append("reference_protocol")
        if unsupported or candidates is None:
            return None, tuple(dict.fromkeys(unsupported))
        matches = tuple(job for job in candidates if query.target_protocol == _protocol(job))
        if len(matches) != 1:
            return None, ("target_protocol",)
        return matches[0], ()

    def _abstain(
        self,
        query: OracleQuery,
        reason: AbstentionReason,
        unsupported: tuple[str, ...],
    ) -> OracleAbstention:
        return OracleAbstention(
            query_id=query.query_id,
            oracle=self.identity,
            reason=reason,
            unsupported_dimensions=unsupported,
            applicability=ApplicabilityAssessment(
                status=ApplicabilityStatus.OUT_OF_DOMAIN,
                reasons=("Query is not an exact member of the predeclared OAT design.",),
                unsupported_dimensions=unsupported,
                support_evidence_ids=(self._plan.specification.specification_sha256,),
                support_bounds=_support_bounds(),
                model_family_disagreement=(
                    "not assessed because no supported computational prediction was produced"
                ),
            ),
            note="No numeric prediction was produced and no arbitrary override was executed.",
        )


def _abstention_reason(unsupported: tuple[str, ...]) -> AbstentionReason:
    if "parameter_set" in unsupported:
        return AbstentionReason.UNRESOLVED_PARAMETER_SET
    if "system" in unsupported:
        return AbstentionReason.UNSUPPORTED_SYSTEM
    if "observable" in unsupported:
        return AbstentionReason.UNSUPPORTED_OBSERVABLE
    if "model_family" in unsupported:
        return AbstentionReason.MODEL_FAMILY_DISAGREEMENT
    return AbstentionReason.OUTSIDE_SUPPORTED_DOMAIN


def _support_bounds() -> tuple[SupportBound, ...]:
    return (
        SupportBound(
            "magnetic flux-density magnitude",
            0.0,
            200e-6,
            "T",
            "Exact enumerated ideal z-directed B values; no interpolation is claimed.",
        ),
        SupportBound(
            "hypothetical log-scale delta",
            -0.05,
            0.05,
            "1",
            "Exact OAT design levels, not an uncertainty interval or probability distribution.",
        ),
    )


def _applicability() -> ApplicabilityAssessment:
    return ApplicabilityAssessment(
        status=ApplicabilityStatus.IN_DOMAIN,
        reasons=(
            "Computational-design applicability: IN_DOMAIN for this exact enumerated family/field.",
            "Empirical applicability: UNKNOWN because no wet-lab calibration supports "
            "the transform.",
        ),
        unsupported_dimensions=(),
        support_evidence_ids=("paircraft-maeda-hypothetical-oat-design-v1",),
        support_bounds=_support_bounds(),
        model_family_disagreement="not assessed; this oracle executes one physical model family",
    )


def _distribution(result: HypotheticalSecondaryPairYield) -> PredictiveDistribution:
    value = result.secondary_pair
    return PredictiveDistribution(
        mean=value,
        lower=value,
        upper=value,
        coverage_probability=None,
        unit="1",
        method=(
            "deterministic hypothetical Maeda OAT solve; "
            f"residual={result.relative_residual:.6g}; iterations={result.iterations}"
        ),
        epistemic_standard_uncertainty=None,
        aleatoric_standard_uncertainty=None,
        simulator_discrepancy_standard_uncertainty=None,
        numerical_standard_uncertainty=None,
        decomposition_status="not calculated; no uncertainty distribution is defined",
        uncertainty_assumptions=(
            "The exp(plus or minus 0.05) levels are deterministic design points, not draws, "
            "confidence limits, priors, or biological variation."
        ),
        calibration_status="uncalibrated simulator-only hypothetical point result",
        note=(
            "GMRES residual is a convergence diagnostic only. Empirical applicability remains "
            "unknown and the result is reference-only, not training data."
        ),
    )


def _execution(result: HypotheticalSecondaryPairYield) -> NumericalExecution:
    return NumericalExecution(
        model_id=result.model_id,
        model_family=_MODEL_FAMILY,
        solver_name=result.solver,
        solver_versions=(("numpy", np.__version__), ("scipy", scipy.__version__)),
        code_version=result.code_version,
        code_artifact_sha256=result.code_artifact_sha256,
        settings=(
            SolverSetting(
                "maximum_restart_cycles",
                str(result.maximum_restart_cycles),
                "1",
                "Maximum number of restarted GMRES cycles.",
            ),
            SolverSetting(
                "relative_tolerance",
                str(result.requested_relative_tolerance),
                "1",
                "Requested true relative residual tolerance.",
            ),
            SolverSetting(
                "restart",
                str(result.restart),
                "1",
                "Krylov subspace dimension before GMRES restart.",
            ),
        ),
        achieved_relative_residual=result.relative_residual,
        iteration_count=result.iterations,
        deterministic=True,
        random_seed=None,
        randomness_note="Deterministic OAT design and linear solve; no random sampling is used.",
    )


def _parameter_artifact(parameter: PublishedParameter, *, role: str) -> LineageArtifact:
    return LineageArtifact(
        artifact_id=parameter.parameter_id,
        sha256=_canonical_sha256(asdict(parameter)),
        evidence_role=role,
        evidence_status=parameter.evidence_status.value,
        reuse_status="independent numerical representation of cited fact",
        membership="inherited baseline parameter; not training data",
        exact_location="; ".join(citation.exact_location for citation in parameter.citation_chain),
        conditions=parameter.conditions,
        uncertainty_note=f"{parameter.uncertainty_status}: {parameter.uncertainty_note}",
        source_uri=parameter.source_uri,
        source_artifact_sha256=parameter.source_artifact_sha256,
        license=parameter.license,
    )


def _baseline_lineage(baseline: Maeda2012Specification) -> tuple[LineageArtifact, ...]:
    artifacts: list[LineageArtifact] = [
        LineageArtifact(
            artifact_id=f"{baseline.protein_system.value}-published-baseline-parameter-set",
            sha256=baseline.parameter_set_sha256,
            evidence_role="complete inherited published point-estimate baseline",
            evidence_status="calculated, fitted, fixed, and assumed inputs retained separately",
            reuse_status="independent numerical representation of cited facts",
            membership="inherited oracle baseline; not training data",
            exact_location="Paircraft Maeda2012Specification canonical payload",
            conditions=baseline.forward_rate.conditions,
            uncertainty_note=(
                "Fitted-rate covariance is unavailable; this baseline is not ground truth."
            ),
            source_uri="paircraft:maeda-2012-specification",
            source_artifact_sha256=None,
            license="Internal structured representation; source rights retained separately.",
        )
    ]
    for hyperfine in baseline.hyperfine_parameters:
        artifacts.append(
            replace(
                _parameter_artifact(
                    hyperfine.equivalent_field,
                    role=(
                        "inherited signed isotropic hyperfine input for "
                        f"{hyperfine.radical} {hyperfine.nucleus}"
                    ),
                ),
                sha256=_canonical_sha256(asdict(hyperfine)),
            )
        )
    for parameter in (
        baseline.forward_rate,
        baseline.back_recombination_rate,
        baseline.st_dephasing_rate,
        baseline.electron_g_factor,
        baseline.electron_gyromagnetic_ratio,
    ):
        artifacts.append(_parameter_artifact(parameter, role="inherited reduced-model scalar"))
    assumption = baseline.initial_nuclear_state_assumption
    artifacts.append(
        LineageArtifact(
            artifact_id="initial-nuclear-state-assumption",
            sha256=sha256(assumption.encode()).hexdigest(),
            evidence_role="inherited initial nuclear-state assumption",
            evidence_status="assumed",
            reuse_status="Paircraft model assumption",
            membership="inherited oracle assumption; not training data",
            exact_location="Maeda2012Specification.initial_nuclear_state_assumption",
            conditions=baseline.forward_rate.conditions,
            uncertainty_note="No nuclear-polarization uncertainty is propagated.",
            source_uri="paircraft:maeda-2012-specification",
            source_artifact_sha256=None,
            license="Internal structured assumption.",
        )
    )
    return tuple(artifacts)


def _lineage(
    plan: MaedaSensitivityPlan,
    job: MaedaSensitivityJob,
    result: HypotheticalSecondaryPairYield,
    execution: NumericalExecution,
) -> tuple[LineageArtifact, ...]:
    override = result.specification.override
    execution_sha = _canonical_sha256(
        {
            "configuration_sha256": execution.configuration_sha256,
            "residual": result.relative_residual,
            "iterations": result.iterations,
        }
    )
    return (
        *_baseline_lineage(result.specification.baseline),
        LineageArtifact(
            artifact_id="hypothetical-parameter-transform",
            sha256=_canonical_sha256(asdict(override)),
            evidence_role="one predeclared hypothetical OAT transform",
            evidence_status=override.evidence_status,
            reuse_status="Paircraft internal deterministic design",
            membership="hypothetical design input; not training data",
            exact_location=override.design_exact_location,
            conditions=override.conditions,
            uncertainty_note=override.uncertainty_status,
            source_uri=override.design_source_uri,
            source_artifact_sha256=None,
            license="Internal generated design metadata.",
        ),
        LineageArtifact(
            artifact_id="hypothetical-parameter-family",
            sha256=job.parameter_family_sha256,
            evidence_role="complete hypothetical parameter-family identity",
            evidence_status="assumed_hypothetical_design",
            reuse_status="Paircraft internal deterministic design",
            membership="reference-only parameter family; not training data",
            exact_location="MaedaSensitivityFamily.spin_specification",
            conditions=override.conditions,
            uncertainty_note="No uncertainty distribution is defined for this family.",
            source_uri="paircraft:maeda-sensitivity-specification",
            source_artifact_sha256=None,
            license="Internal generated design metadata.",
        ),
        LineageArtifact(
            artifact_id="sensitivity-design-dependency-group",
            sha256=job.design_dependency_sha256,
            evidence_role="parent-level dependency shared by all baseline/OAT curves",
            evidence_status="calculated",
            reuse_status="Paircraft internal deterministic design",
            membership="design dependency; not training data",
            exact_location="MaedaSensitivityJob.design_dependency_sha256",
            conditions=plan.specification.claim_boundary,
            uncertainty_note="Dependent families must not be split across corpus roles.",
            source_uri="paircraft:maeda-local-sensitivity-plan",
            source_artifact_sha256=None,
            license="Internal generated design metadata.",
        ),
        LineageArtifact(
            artifact_id="sensitivity-plan-job",
            sha256=job.job_sha256,
            evidence_role="exact predeclared family/field job",
            evidence_status="calculated",
            reuse_status="Paircraft internal deterministic design",
            membership="reference-only job; not training data",
            exact_location="MaedaSensitivityPlan.jobs",
            conditions=plan.specification.claim_boundary,
            uncertainty_note="The job is an enumerated calculation, not a random draw.",
            source_uri="paircraft:maeda-local-sensitivity-plan",
            source_artifact_sha256=None,
            license="Internal generated design metadata.",
        ),
        LineageArtifact(
            artifact_id="maeda-sensitivity-numerical-execution",
            sha256=execution_sha,
            evidence_role="numerical execution manifest",
            evidence_status="calculated",
            reuse_status="Paircraft internal execution evidence",
            membership="result execution; not training data",
            exact_location="NumericalExecution embedded in OraclePrediction",
            conditions=(
                f"configuration={execution.configuration_sha256}; "
                f"residual={result.relative_residual}; iterations={result.iterations}"
            ),
            uncertainty_note="Residual is convergence evidence, not predictive uncertainty.",
            source_uri="paircraft:maeda-sensitivity-execution",
            source_artifact_sha256=None,
            license="Internal generated metadata.",
        ),
    )


def execute_maeda_sensitivity_oracle_job(
    plan: MaedaSensitivityPlan,
    job: MaedaSensitivityJob,
) -> SimulationDatum:
    """Execute one exact plan job and retain reference-only simulation semantics."""

    if not isinstance(plan, MaedaSensitivityPlan) or job not in plan.jobs:
        raise ValueError("oracle execution requires an exact sensitivity plan job")
    executor = MaedaSensitivitySimulatorOracle(plan)
    return _execute_exact_job(executor, job)


def execute_maeda_sensitivity_oracle_plan(
    plan: MaedaSensitivityPlan,
) -> tuple[SimulationDatum, ...]:
    """Execute the complete immutable plan through one exact internal oracle."""

    if not isinstance(plan, MaedaSensitivityPlan):
        raise ValueError("oracle plan execution requires a typed sensitivity plan")
    return tuple(iter_maeda_sensitivity_oracle_jobs(plan, plan.jobs))


def iter_maeda_sensitivity_oracle_jobs(
    plan: MaedaSensitivityPlan,
    jobs: tuple[MaedaSensitivityJob, ...],
) -> Iterator[SimulationDatum]:
    """Yield selected exact plan jobs through one internally constructed oracle."""

    if not isinstance(plan, MaedaSensitivityPlan) or not isinstance(jobs, tuple):
        raise ValueError("oracle job iteration requires a typed plan and job tuple")
    if any(not isinstance(job, MaedaSensitivityJob) or job not in plan.jobs for job in jobs):
        raise ValueError("oracle job iteration contains a job outside its plan")
    executor = MaedaSensitivitySimulatorOracle(plan)
    for job in jobs:
        yield _execute_exact_job(executor, job)


def _execute_exact_job(
    executor: MaedaSensitivitySimulatorOracle,
    job: MaedaSensitivityJob,
) -> SimulationDatum:
    plan = executor.plan
    if job not in plan.jobs:
        raise ValueError("oracle execution requires an exact sensitivity plan job")
    result = executor.predict(executor._query_for_exact_job(job))
    if not isinstance(result, OraclePrediction) or result.execution is None:
        raise ArithmeticError("sensitivity oracle did not return an executable prediction")
    datum = SimulationDatum(result, result.execution)
    lineage = {artifact.artifact_id: artifact.sha256 for artifact in result.lineage}
    if (
        datum.prediction.query.system.parameter_set_sha256 != job.parameter_family_sha256
        or lineage.get("sensitivity-design-dependency-group") != job.design_dependency_sha256
        or lineage.get("sensitivity-plan-job") != job.job_sha256
    ):
        raise ValueError("sensitivity datum lost its planned family or dependency identity")
    return datum
