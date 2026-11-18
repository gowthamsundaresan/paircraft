"""Simulator-only oracle for the published Maeda 2012 reduced model."""

from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from math import isclose, sqrt
from pathlib import Path
from typing import Final

import numpy as np
import scipy

from paircraft import __version__
from paircraft.coupling import CoordinateFrame
from paircraft.oracle.model import (
    AbstentionReason,
    ApplicabilityAssessment,
    ApplicabilityStatus,
    CalibrationTier,
    ConditionStatus,
    DeclaredCondition,
    DeclaredQuantity,
    ExperimentProtocol,
    FieldIntent,
    IlluminationProtocol,
    InitialSpinState,
    LineageArtifact,
    NumericalExecution,
    ObservableKind,
    ObservationWindow,
    OracleAbstention,
    OracleIdentity,
    OraclePrediction,
    OracleQuery,
    OracleResult,
    PredictiveDistribution,
    RadicalPairSystemRef,
    RadicalSpecies,
    RequestedMagneticFluxDensity,
    ResolvedExperimentSetup,
    SolverSetting,
    SpinModelFamily,
    SupportBound,
)
from paircraft.spin import (
    Maeda2012Specification,
    MaedaProteinSystem,
    SecondaryPairYield,
    maeda_2012_specification,
    solve_maeda_secondary_pair_yield,
)

_QUERY_SCHEMA_VERSION: Final = "paircraft-oracle-query-v2"
_CATALOG_VERSION: Final = "paircraft-maeda-2012-catalog-v1"
_MAX_FIELD_TESLA: Final = 0.05
_REFERENCE_STABILITY_THRESHOLD: Final = 1e-12
_TRAINING_MANIFEST_ID: Final = "none-simulator-only"
_TRAINING_MANIFEST_SHA256: Final = sha256(b"").hexdigest()
_MODEL_FAMILY: Final = SpinModelFamily.HABERKORN_STATIC_PAIR_WITH_ST_DEPHASING
_SYSTEM_IDS: Final = {
    MaedaProteinSystem.ATCRY1: "maeda-2012-atcry1-rate-set-v1",
    MaedaProteinSystem.ECPL_E109A: "maeda-2012-ecpl-e109a-rate-set-v1",
}
_SYSTEM_BY_ID: Final = {identifier: system for system, identifier in _SYSTEM_IDS.items()}


def _catalog_sha256() -> str:
    payload = {
        "catalog_version": _CATALOG_VERSION,
        "systems": {
            _SYSTEM_IDS[system]: maeda_2012_specification(system).parameter_set_sha256
            for system in MaedaProteinSystem
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode()).hexdigest()


_CATALOG_SHA256: Final = _catalog_sha256()


def _artifact_sha256() -> str:
    digest = sha256()
    paths = (
        Path(__file__),
        Path(__file__).with_name("model.py"),
        Path(__file__).parents[1] / "spin" / "maeda2012.py",
        Path(__file__).parents[1] / "spin" / "_maeda_kernel.py",
    )
    for path in paths:
        digest.update(path.read_bytes())
    for system in MaedaProteinSystem:
        digest.update(maeda_2012_specification(system).parameter_set_sha256.encode())
    digest.update(_CATALOG_VERSION.encode())
    return digest.hexdigest()


def _frame() -> CoordinateFrame:
    return CoordinateFrame(
        name="maeda-2012-isotropic-lab-frame",
        definition="Origin at the radical pair; axes define the ideal Cartesian B vector.",
        axis_convention="Right-handed x, y, z; isotropic model has no molecular-frame axis.",
        handedness="right",
    )


def maeda_2012_system(system: MaedaProteinSystem) -> RadicalPairSystemRef:
    """Return the exact catalog identity for one published Maeda rate set."""

    specification = maeda_2012_specification(system)
    return RadicalPairSystemRef(
        system_id=_SYSTEM_IDS[system],
        radical_one=RadicalSpecies(
            identity="FAD radical anion in the Maeda five-nucleus approximation",
            formal_charge=-1,
            redox_state="one-electron-reduced radical anion",
            protonation_state="unprotonated flavin radical anion",
            isotopic_composition="explicit 14N5 plus one effective spin-1/2 pseudo-nucleus",
        ),
        radical_two=RadicalSpecies(
            identity="TrpH radical cation in the Maeda five-nucleus approximation",
            formal_charge=1,
            redox_state="one-electron-oxidized radical cation",
            protonation_state="TrpH radical cation",
            isotopic_composition=(
                "explicit 1H18 and 1H23 plus one effective spin-1/2 pseudo-nucleus"
            ),
        ),
        initial_spin_state=InitialSpinState.SINGLET,
        parameter_set_id=f"maeda-2012-{system.value}-point-estimates-v1",
        parameter_set_sha256=specification.parameter_set_sha256,
        parameter_catalog_version=_CATALOG_VERSION,
        parameter_catalog_sha256=_CATALOG_SHA256,
    )


def _quantity(
    status: ConditionStatus,
    value: float | None,
    unit: str,
    note: str,
) -> DeclaredQuantity:
    return DeclaredQuantity(status, value, unit, note)


def maeda_2012_protocol(
    system: MaedaProteinSystem,
    field_tesla: tuple[float, float, float],
) -> ExperimentProtocol:
    """Return the exact published context at one ideal simulation field."""

    if system is MaedaProteinSystem.ATCRY1:
        temperature = 270.0
        solvent = "60% glycerol; pH 7.5 Tris/HCl; 500 mM NaCl; 250 mM imidazole"
        ph_value = 7.5
    else:
        temperature = 260.0
        solvent = "50% glycerol; 50 mM HEPES; 100 mM KCl"
        ph_value = 7.0
    return ExperimentProtocol(
        field=RequestedMagneticFluxDensity(
            *field_tesla,
            coordinate_frame=_frame(),
            intent=FieldIntent.IDEAL_SIMULATION_TARGET,
        ),
        temperature_kelvin=temperature,
        solvent=solvent,
        viscosity=_quantity(
            ConditionStatus.UNKNOWN,
            None,
            "Pa s",
            "The glycerol fraction and temperature are reported, but viscosity is not.",
        ),
        orientation_regime=(
            "isotropic five-nucleus reduced Hamiltonian; protein orientation not modelled"
        ),
        ph=DeclaredCondition(
            ConditionStatus.VALUE,
            ph_value,
            "1",
            "Reported sample pH.",
        ),
        oxygen=DeclaredCondition(
            ConditionStatus.UNKNOWN,
            None,
            "1",
            "Sample oxygen state was not resolved for the published model input.",
        ),
        illumination=IlluminationProtocol(
            wavelength=_quantity(
                ConditionStatus.VALUE,
                460e-9,
                "m",
                "Reported excitation wavelength.",
            ),
            irradiance=_quantity(
                ConditionStatus.UNKNOWN,
                None,
                "W m^-2",
                "Irradiance was not reported.",
            ),
            duration=_quantity(
                ConditionStatus.VALUE,
                5e-9,
                "s",
                "Reported excitation pulse duration.",
            ),
            dose=_quantity(
                ConditionStatus.UNKNOWN,
                None,
                "J m^-2",
                "The reported 5-7 mJ pulse energy lacks the beam area needed for areal dose.",
            ),
            timing="instantaneous singlet radical-pair preparation at simulation time zero",
            note=(
                "Optical context is retained; the reduced spin model begins from the prepared "
                "radical pair and does not simulate excitation."
            ),
        ),
        observation_window=ObservationWindow(
            0.0,
            None,
            "Infinite-time integrated secondary radical-pair yield.",
        ),
        reaction_model_id="maeda-2012-singlet-back-recombination-plus-rp2-formation-v1",
        relaxation_model_id="maeda-2012-shushin-singlet-triplet-dephasing-v1",
    )


def maeda_2012_query(
    system: MaedaProteinSystem,
    target_field_tesla: tuple[float, float, float],
    observable: ObservableKind = ObservableKind.SECONDARY_RADICAL_YIELD,
    *,
    reference_field_tesla: tuple[float, float, float] | None = None,
    query_id: str = "maeda-2012-query",
) -> OracleQuery:
    """Construct one exact Maeda simulator-oracle query."""

    reference = (
        None
        if reference_field_tesla is None
        else maeda_2012_protocol(system, reference_field_tesla)
    )
    return OracleQuery(
        schema_version=_QUERY_SCHEMA_VERSION,
        query_id=query_id,
        system=maeda_2012_system(system),
        target_protocol=maeda_2012_protocol(system, target_field_tesla),
        reference_protocol=reference,
        observable=observable,
        model_family=_MODEL_FAMILY,
    )


class Maeda2012SimulatorOracle:
    """Expose the exact published point-estimate model through the oracle contract."""

    @property
    def identity(self) -> OracleIdentity:
        """Return the content-addressed simulator-oracle identity."""

        return OracleIdentity(
            oracle_id="paircraft-maeda-2012-simulator-oracle",
            oracle_version="1",
            artifact_sha256=_artifact_sha256(),
            code_version=__version__,
            training_manifest_id=_TRAINING_MANIFEST_ID,
            training_manifest_sha256=_TRAINING_MANIFEST_SHA256,
            parameter_catalog_version=_CATALOG_VERSION,
            parameter_catalog_sha256=_CATALOG_SHA256,
            supported_model_families=(_MODEL_FAMILY,),
            supported_observables=(
                ObservableKind.SECONDARY_RADICAL_YIELD,
                ObservableKind.SECONDARY_RADICAL_YIELD_MFE,
            ),
            calibration_tier=CalibrationTier.SIMULATOR_ONLY,
        )

    def predict(self, query: OracleQuery) -> OracleResult:
        """Return secondary-pair yield or a structured abstention."""

        unsupported = self._unsupported_dimensions(query)
        if unsupported:
            if "parameter_set" in unsupported:
                reason = AbstentionReason.UNRESOLVED_PARAMETER_SET
            elif "system" in unsupported:
                reason = AbstentionReason.UNSUPPORTED_SYSTEM
            elif "observable" in unsupported:
                reason = AbstentionReason.UNSUPPORTED_OBSERVABLE
            elif "model_family" in unsupported:
                reason = AbstentionReason.MODEL_FAMILY_DISAGREEMENT
            elif _query_has_incomplete_protocol(query):
                reason = AbstentionReason.INCOMPLETE_PROTOCOL
            else:
                reason = AbstentionReason.OUTSIDE_SUPPORTED_DOMAIN
            return self._abstain(query, reason, unsupported)

        system = _SYSTEM_BY_ID[query.system.system_id]
        specification = maeda_2012_specification(system)
        try:
            target_result = solve_maeda_secondary_pair_yield(
                specification,
                query.target_protocol.field.components_tesla,
            )
            target = target_result.secondary_pair
            execution = _execution(target_result)
            if query.observable.is_relative:
                reference_protocol = query.reference_protocol
                if reference_protocol is None:
                    raise AssertionError("validated relative query lost its reference protocol")
                reference_result = solve_maeda_secondary_pair_yield(
                    specification,
                    reference_protocol.field.components_tesla,
                )
                reference = reference_result.secondary_pair
                if abs(reference) <= _REFERENCE_STABILITY_THRESHOLD:
                    return self._abstain(
                        query,
                        AbstentionReason.UNCALIBRATED_UNCERTAINTY,
                        ("reference_denominator",),
                    )
                prediction = (target - reference) / reference
            else:
                prediction = target
        except (ArithmeticError, ValueError):
            return self._abstain(query, AbstentionReason.NUMERICAL_FAILURE, ("solver",))

        magnitudes = [_field_magnitude(query.target_protocol.field)]
        if query.reference_protocol is not None:
            magnitudes.append(_field_magnitude(query.reference_protocol.field))
        status = (
            ApplicabilityStatus.BOUNDARY
            if any(
                isclose(value, _MAX_FIELD_TESLA, rel_tol=0.0, abs_tol=1e-15) for value in magnitudes
            )
            else ApplicabilityStatus.IN_DOMAIN
        )
        applicability = ApplicabilityAssessment(
            status=status,
            reasons=(
                "Exact published point-estimate parameter set and ideal protocol are supported.",
            ),
            unsupported_dimensions=(),
            support_evidence_ids=("maeda-2012-article", "maeda-2012-supporting-information"),
            support_bounds=(_field_support_bound(),),
            model_family_disagreement="not assessed; this oracle executes one model family",
        )
        distribution = PredictiveDistribution(
            mean=prediction,
            lower=prediction,
            upper=prediction,
            coverage_probability=None,
            unit="1",
            method=(
                "deterministic infinite-time Maeda RP2 solve; "
                f"residual={target_result.relative_residual:.6g}; "
                f"iterations={target_result.iterations}"
            ),
            epistemic_standard_uncertainty=None,
            aleatoric_standard_uncertainty=None,
            simulator_discrepancy_standard_uncertainty=None,
            numerical_standard_uncertainty=None,
            decomposition_status="unavailable for published point-estimate model",
            uncertainty_assumptions=(
                "Fitted-rate covariance is unavailable; common rate-scaling "
                "non-identifiability and model discrepancy are not propagated."
            ),
            calibration_status="uncalibrated simulator-only point result; no predictive coverage",
            note=(
                "GMRES residual is a convergence diagnostic, not predictive uncertainty. "
                "Millitesla model agreement does not validate sub-Earth-field biology."
            ),
        )
        return OraclePrediction(
            query=query,
            resolved_setup=ResolvedExperimentSetup(
                query.system,
                query.target_protocol,
                query.reference_protocol,
            ),
            oracle=self.identity,
            executed_model_family=_MODEL_FAMILY,
            distribution=distribution,
            applicability=applicability,
            lineage=_lineage(specification, target_result, execution),
            claim_boundary=(
                "simulator-only reproduction of a published reduced point-estimate model; not "
                "experimental calibration, sequence prediction, exact Figure 6 refitting, or "
                "sub-Earth-field validation"
            ),
            execution=execution,
            executed_model_id=target_result.model_id,
        )

    def _unsupported_dimensions(self, query: OracleQuery) -> tuple[str, ...]:
        unsupported: list[str] = []
        system = _SYSTEM_BY_ID.get(query.system.system_id)
        if query.schema_version != _QUERY_SCHEMA_VERSION:
            unsupported.append("schema_version")
        if system is None:
            unsupported.append("system")
        else:
            expected = maeda_2012_system(system)
            if (
                query.system.parameter_set_id != expected.parameter_set_id
                or query.system.parameter_set_sha256 != expected.parameter_set_sha256
            ):
                unsupported.append("parameter_set")
            if query.system != expected:
                unsupported.append("system")
            if not _protocol_is_supported(system, query.target_protocol):
                unsupported.append("target_protocol")
            if query.reference_protocol is not None and not _protocol_is_supported(
                system,
                query.reference_protocol,
            ):
                unsupported.append("reference_protocol")
        if query.model_family is not _MODEL_FAMILY:
            unsupported.append("model_family")
        if query.observable not in self.identity.supported_observables:
            unsupported.append("observable")
        return tuple(dict.fromkeys(unsupported))

    def _abstain(
        self,
        query: OracleQuery,
        reason: AbstentionReason,
        unsupported_dimensions: tuple[str, ...],
    ) -> OracleAbstention:
        return OracleAbstention(
            query_id=query.query_id,
            oracle=self.identity,
            reason=reason,
            unsupported_dimensions=unsupported_dimensions,
            applicability=ApplicabilityAssessment(
                status=ApplicabilityStatus.OUT_OF_DOMAIN,
                reasons=("The query is outside the exact Maeda simulator-oracle domain.",),
                unsupported_dimensions=unsupported_dimensions,
                support_evidence_ids=(
                    "maeda-2012-article",
                    "maeda-2012-supporting-information",
                ),
                support_bounds=(_field_support_bound(),),
                model_family_disagreement=(
                    "not assessed because no supported prediction was produced"
                ),
            ),
            note="No numeric prediction was produced.",
        )


def _field_magnitude(field: RequestedMagneticFluxDensity) -> float:
    return sqrt(sum(component**2 for component in field.components_tesla))


def _protocol_is_supported(
    system: MaedaProteinSystem,
    protocol: ExperimentProtocol,
) -> bool:
    expected = maeda_2012_protocol(system, protocol.field.components_tesla)
    return (
        _field_magnitude(protocol.field) <= _MAX_FIELD_TESLA
        and protocol.field.intent is FieldIntent.IDEAL_SIMULATION_TARGET
        and protocol == expected
    )


def _query_has_incomplete_protocol(query: OracleQuery) -> bool:
    system = _SYSTEM_BY_ID.get(query.system.system_id)
    if system is None:
        return False
    protocols = (query.target_protocol, query.reference_protocol)
    for protocol in protocols:
        if protocol is None:
            continue
        expected = maeda_2012_protocol(system, protocol.field.components_tesla)
        required = (
            (protocol.ph.status, expected.ph.status),
            (
                protocol.illumination.wavelength.status,
                expected.illumination.wavelength.status,
            ),
            (
                protocol.illumination.duration.status,
                expected.illumination.duration.status,
            ),
        )
        if any(
            expected_status is ConditionStatus.VALUE and actual_status is not ConditionStatus.VALUE
            for actual_status, expected_status in required
        ):
            return True
    return False


def _field_support_bound() -> SupportBound:
    return SupportBound(
        dimension="magnetic flux-density magnitude",
        lower=0.0,
        upper=_MAX_FIELD_TESLA,
        unit="T",
        definition="Euclidean magnitude of the ideal Cartesian B vector.",
    )


def _execution(result: SecondaryPairYield) -> NumericalExecution:
    return NumericalExecution(
        model_id=result.model_id,
        model_family=_MODEL_FAMILY,
        solver_name=result.solver,
        solver_versions=(
            ("numpy", np.__version__),
            ("scipy", scipy.__version__),
        ),
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
        randomness_note="Deterministic linear-algebra solve; no random sampling is used.",
    )


def _lineage(
    specification: Maeda2012Specification,
    result: SecondaryPairYield,
    execution: NumericalExecution,
) -> tuple[LineageArtifact, ...]:
    hyperfine = specification.hyperfine_parameters[0].equivalent_field
    fitted = specification.back_recombination_rate
    if hyperfine.source_artifact_sha256 is None or fitted.source_artifact_sha256 is None:
        raise AssertionError("published Maeda source artifacts lost their verified hashes")
    execution_payload = {
        "configuration_sha256": execution.configuration_sha256,
        "residual": result.relative_residual,
        "iterations": result.iterations,
    }
    execution_sha256 = sha256(
        json.dumps(execution_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    parameter_lineage = _parameter_lineage(specification)
    return (
        LineageArtifact(
            artifact_id=f"{specification.protein_system.value}-parameter-set",
            sha256=specification.parameter_set_sha256,
            evidence_role="complete reduced-model parameter set",
            evidence_status="calculated, fitted, fixed, and assumed inputs retained separately",
            reuse_status="independent numerical representation of cited facts",
            membership="oracle parameter set; not training data",
            exact_location="Paircraft Maeda2012Specification canonical payload",
            conditions=specification.forward_rate.conditions,
            uncertainty_note=(
                "Fitted-rate covariance is unavailable and common scaling is non-identifiable."
            ),
            source_uri="paircraft:maeda-2012-specification",
            source_artifact_sha256=None,
            license="Internal structured representation; source rights retained separately.",
        ),
        *parameter_lineage,
        LineageArtifact(
            artifact_id="maeda-2012-supporting-information",
            sha256=hyperfine.source_artifact_sha256,
            evidence_role="Hamiltonian, hyperfine, kinetic, and dephasing model source",
            evidence_status="published supporting information",
            reuse_status="reference-only source; numerical facts independently represented",
            membership="source evidence; not training data",
            exact_location="Tables S2-S3 and section 5",
            conditions=("UB3LYP/EPR-III lumiflavin-proxy couplings reduced to five nuclei."),
            uncertainty_note="Calculated hyperfine inputs have no reported uncertainty.",
            source_uri=hyperfine.source_uri,
            source_artifact_sha256=hyperfine.source_artifact_sha256,
            license=hyperfine.license,
        ),
        LineageArtifact(
            artifact_id="maeda-2012-article",
            sha256=fitted.source_artifact_sha256,
            evidence_role="experimental context and fitted-rate source",
            evidence_status="published peer-reviewed article",
            reuse_status="reference-only source; numerical facts independently represented",
            membership="source evidence; not training data",
            exact_location="Figure 6 caption and Results fit paragraph",
            conditions=fitted.conditions,
            uncertainty_note="Best-fit rates are reported without covariance or intervals.",
            source_uri=fitted.source_uri,
            source_artifact_sha256=fitted.source_artifact_sha256,
            license=fitted.license,
        ),
        LineageArtifact(
            artifact_id="maeda-2012-numerical-execution",
            sha256=execution_sha256,
            evidence_role="numerical execution manifest",
            evidence_status="calculated",
            reuse_status="Paircraft internal execution evidence",
            membership="result execution; not training data",
            exact_location="NumericalExecution embedded in OraclePrediction",
            conditions=(
                f"configuration={execution.configuration_sha256}; "
                f"residual={result.relative_residual}; iterations={result.iterations}"
            ),
            uncertainty_note="Residual is a convergence diagnostic, not outcome uncertainty.",
            source_uri="paircraft:maeda-2012-execution",
            source_artifact_sha256=None,
            license="Internal generated metadata.",
        ),
    )


def _parameter_lineage(
    specification: Maeda2012Specification,
) -> tuple[LineageArtifact, ...]:
    artifacts: list[LineageArtifact] = []
    for hyperfine in specification.hyperfine_parameters:
        parameter = hyperfine.equivalent_field
        payload = json.dumps(asdict(hyperfine), sort_keys=True, separators=(",", ":"))
        artifacts.append(
            LineageArtifact(
                artifact_id=parameter.parameter_id,
                sha256=sha256(payload.encode()).hexdigest(),
                evidence_role=(
                    f"signed isotropic hyperfine input for {hyperfine.radical} {hyperfine.nucleus}"
                ),
                evidence_status=parameter.evidence_status.value,
                reuse_status="independent numerical representation of cited fact",
                membership="oracle parameter; not training data",
                exact_location="; ".join(
                    citation.exact_location for citation in parameter.citation_chain
                ),
                conditions=parameter.conditions,
                uncertainty_note=(f"{parameter.uncertainty_status}: {parameter.uncertainty_note}"),
                source_uri=parameter.source_uri,
                source_artifact_sha256=parameter.source_artifact_sha256,
                license=parameter.license,
            )
        )
    scalar_parameters = (
        specification.forward_rate,
        specification.back_recombination_rate,
        specification.st_dephasing_rate,
        specification.electron_g_factor,
        specification.electron_gyromagnetic_ratio,
    )
    for parameter in scalar_parameters:
        payload = json.dumps(asdict(parameter), sort_keys=True, separators=(",", ":"))
        artifacts.append(
            LineageArtifact(
                artifact_id=parameter.parameter_id,
                sha256=sha256(payload.encode()).hexdigest(),
                evidence_role="reduced-model scalar parameter",
                evidence_status=parameter.evidence_status.value,
                reuse_status="independent numerical representation of cited fact",
                membership="oracle parameter; not training data",
                exact_location="; ".join(
                    citation.exact_location for citation in parameter.citation_chain
                ),
                conditions=parameter.conditions,
                uncertainty_note=(f"{parameter.uncertainty_status}: {parameter.uncertainty_note}"),
                source_uri=parameter.source_uri,
                source_artifact_sha256=parameter.source_artifact_sha256,
                license=parameter.license,
            )
        )
    assumption = specification.initial_nuclear_state_assumption
    artifacts.append(
        LineageArtifact(
            artifact_id="initial-nuclear-state-assumption",
            sha256=sha256(assumption.encode()).hexdigest(),
            evidence_role="initial nuclear-state assumption",
            evidence_status="assumed",
            reuse_status="Paircraft internal model declaration",
            membership="oracle parameter; not training data",
            exact_location="Maeda2012Specification.initial_nuclear_state_assumption",
            conditions=assumption,
            uncertainty_note="No nuclear-polarization uncertainty is propagated.",
            source_uri="paircraft:maeda-2012-specification",
            source_artifact_sha256=None,
            license="Internal structured representation.",
        )
    )
    return tuple(artifacts)
