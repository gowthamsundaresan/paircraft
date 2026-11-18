"""One pinned, simulator-only Haberkorn oracle for contract conformance."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import sqrt
from pathlib import Path
from typing import Final

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
    SpinModelFamily,
    SupportBound,
)
from paircraft.spin.haberkorn import (
    _ReducedIsotropicHaberkornSystem,
    _solve_reaction_yields,
)

_QUERY_SCHEMA_VERSION: Final = "paircraft-oracle-query-v1"
_SYSTEM_ID: Final = "reference-fad-anion-trp-cation-one-proton-v1"
_PARAMETER_SET_ID: Final = "reference-haberkorn-one-proton-v1"


@dataclass(frozen=True, slots=True)
class _ScientificParameter:
    parameter_id: str
    value_si: float
    unit: str
    source: str
    exact_location: str
    conditions: str
    evidence_status: str
    uncertainty_status: str
    uncertainty_note: str
    source_uri: str
    source_artifact_sha256: str | None
    license: str


_PARAMETERS: Final = (
    _ScientificParameter(
        "effective-flavin-h8-hyperfine",
        12_330_000.0,
        "Hz",
        "spin-src-kattnig-2017-hfc-si",
        "Supporting Information Table S1, FAD radical anion 3 x H8 row",
        "Reduced isotropic benchmark; not mapped to an exact protein geometry.",
        "inherited",
        "not_reported",
        "The source reports no uncertainty for the inherited calculated scalar.",
        "https://doi.org/10.1021/acs.jpcb.7b07672",
        None,
        "Reuse licence not stated; value used as reference-only inherited evidence.",
    ),
    _ScientificParameter(
        "singlet-recombination-rate",
        1_000_000.0,
        "s^-1",
        "paircraft-reference-haberkorn-v1",
        "Reference benchmark specification, assumed equal reaction rates",
        "Reduced engine-conformance benchmark; no experimental sample instantiated.",
        "assumed",
        "not_applicable",
        "Chosen for a numerical benchmark and not interpreted as a measured rate.",
        "paircraft:reference-haberkorn-v1",
        None,
        "Internal simulator-only benchmark declaration.",
    ),
    _ScientificParameter(
        "triplet-recombination-rate",
        1_000_000.0,
        "s^-1",
        "paircraft-reference-haberkorn-v1",
        "Reference benchmark specification, assumed equal reaction rates",
        "Reduced engine-conformance benchmark; no experimental sample instantiated.",
        "assumed",
        "not_applicable",
        "Chosen for a numerical benchmark and not interpreted as a measured rate.",
        "paircraft:reference-haberkorn-v1",
        None,
        "Internal simulator-only benchmark declaration.",
    ),
    _ScientificParameter(
        "electron-gyromagnetic-ratio",
        176_085_963_023.0,
        "rad s^-1 T^-1",
        "paircraft-reference-haberkorn-v1",
        "Reference benchmark constant declaration",
        "Fixed electron-only Zeeman benchmark; nuclear Zeeman interaction omitted.",
        "assumed",
        "not_modelled",
        "Constant uncertainty is outside this engine-conformance calculation.",
        "paircraft:reference-haberkorn-v1",
        None,
        "Internal simulator-only benchmark declaration.",
    ),
)
_PARAMETER_PAYLOAD: Final = json.dumps(
    [asdict(parameter) for parameter in _PARAMETERS], sort_keys=True, separators=(",", ":")
)
_PARAMETER_SET_SHA256: Final = sha256(_PARAMETER_PAYLOAD.encode()).hexdigest()
_PARAMETER_CATALOG_VERSION: Final = "paircraft-reference-catalog-v1"
_PARAMETER_CATALOG_SHA256: Final = sha256(
    f"{_PARAMETER_CATALOG_VERSION}:{_PARAMETER_SET_SHA256}".encode()
).hexdigest()
_TRAINING_MANIFEST_ID: Final = "none-simulator-only"
_TRAINING_MANIFEST_SHA256: Final = sha256(b"").hexdigest()
_SOURCE_EVIDENCE_DECLARATION: Final = (
    "RadicalPy software-method evidence; DOI 10.1021/acs.jctc.4c00887; "
    "Methods and Haberkorn/MARY examples; source bytes not retained"
)
_SOURCE_EVIDENCE_SHA256: Final = sha256(_SOURCE_EVIDENCE_DECLARATION.encode()).hexdigest()
_PARAMETER_BY_ID: Final = {parameter.parameter_id: parameter for parameter in _PARAMETERS}
_ELECTRON_GYROMAGNETIC_RATIO: Final = _PARAMETER_BY_ID["electron-gyromagnetic-ratio"].value_si
_HYPERFINE_HZ: Final = _PARAMETER_BY_ID["effective-flavin-h8-hyperfine"].value_si
_SINGLET_RATE: Final = _PARAMETER_BY_ID["singlet-recombination-rate"].value_si
_TRIPLET_RATE: Final = _PARAMETER_BY_ID["triplet-recombination-rate"].value_si
_MAX_FIELD_TESLA: Final = 0.1
_REFERENCE_STABILITY_THRESHOLD: Final = 1e-12
_SPIN_SYSTEM: Final = _ReducedIsotropicHaberkornSystem(
    hyperfine_hz=_HYPERFINE_HZ,
    singlet_rate_per_second=_SINGLET_RATE,
    triplet_rate_per_second=_TRIPLET_RATE,
    electron_gyromagnetic_ratio_rad_per_second_tesla=_ELECTRON_GYROMAGNETIC_RATIO,
)


def _artifact_sha256() -> str:
    """Hash the installed oracle implementation, contract, and pinned parameter set."""

    digest = sha256()
    spin_source = Path(__file__).parents[1] / "spin" / "haberkorn.py"
    for source_path in (Path(__file__), Path(__file__).with_name("model.py"), spin_source):
        digest.update(source_path.read_bytes())
    digest.update(_PARAMETER_SET_SHA256.encode())
    return digest.hexdigest()


def _frame() -> CoordinateFrame:
    return CoordinateFrame(
        name="reference-isotropic-lab-frame",
        definition="Origin at the radical pair; axes define the requested Cartesian B vector.",
        axis_convention="Right-handed x, y, z axes; isotropic model has no molecular-frame axis.",
        handedness="right",
    )


def reference_system() -> RadicalPairSystemRef:
    """Return the exact reduced system supported by the reference oracle."""

    return RadicalPairSystemRef(
        system_id=_SYSTEM_ID,
        radical_one=RadicalSpecies(
            identity="FAD radical anion reduced to one effective flavin proton",
            formal_charge=-1,
            redox_state="one-electron-reduced radical anion",
            protonation_state="unprotonated flavin radical anion",
            isotopic_composition="natural abundance with one explicit effective 1H nucleus",
        ),
        radical_two=RadicalSpecies(
            identity="tryptophan radical cation without explicit magnetic nuclei",
            formal_charge=1,
            redox_state="one-electron-oxidized radical cation",
            protonation_state="TrpH radical cation",
            isotopic_composition="magnetic nuclei omitted in the reduced benchmark",
        ),
        initial_spin_state=InitialSpinState.SINGLET,
        parameter_set_id=_PARAMETER_SET_ID,
        parameter_set_sha256=_PARAMETER_SET_SHA256,
        parameter_catalog_version=_PARAMETER_CATALOG_VERSION,
        parameter_catalog_sha256=_PARAMETER_CATALOG_SHA256,
    )


def reference_protocol(z_tesla: float) -> ExperimentProtocol:
    """Return the supported protocol at one ideal field along the declared z axis."""

    not_applicable = DeclaredCondition(
        ConditionStatus.NOT_APPLICABLE,
        None,
        "1",
        "The reduced spin benchmark does not model this condition.",
    )

    def omitted_quantity(unit: str) -> DeclaredQuantity:
        return DeclaredQuantity(
            ConditionStatus.NOT_APPLICABLE,
            None,
            unit,
            "The reduced spin benchmark does not model this optical quantity.",
        )

    return ExperimentProtocol(
        field=RequestedMagneticFluxDensity(
            0.0,
            0.0,
            z_tesla,
            _frame(),
            FieldIntent.IDEAL_SIMULATION_TARGET,
        ),
        temperature_kelvin=298.15,
        solvent="water",
        viscosity=DeclaredQuantity(
            ConditionStatus.VALUE,
            0.00089,
            "Pa s",
            "Pinned ideal reference-benchmark solvent viscosity.",
        ),
        orientation_regime="isotropic effective hyperfine model",
        ph=not_applicable,
        oxygen=not_applicable,
        illumination=IlluminationProtocol(
            wavelength=omitted_quantity("m"),
            irradiance=omitted_quantity("W m^-2"),
            duration=omitted_quantity("s"),
            dose=omitted_quantity("J m^-2"),
            timing="initial state prepared at simulation time zero",
            note="The radical pair is prepared directly in the declared singlet state.",
        ),
        observation_window=ObservationWindow(
            0.0,
            None,
            "Infinite-time integrated reaction yield.",
        ),
        reaction_model_id="haberkorn-equal-singlet-triplet-rates-v1",
        relaxation_model_id="none",
    )


def reference_query(
    target_z_tesla: float,
    observable: ObservableKind = ObservableKind.SINGLET_REACTION_YIELD,
    *,
    reference_z_tesla: float | None = None,
    query_id: str = "reference-query",
) -> OracleQuery:
    """Construct a valid query for the exact supported reference system."""

    reference = None if reference_z_tesla is None else reference_protocol(reference_z_tesla)
    return OracleQuery(
        schema_version=_QUERY_SCHEMA_VERSION,
        query_id=query_id,
        system=reference_system(),
        target_protocol=reference_protocol(target_z_tesla),
        reference_protocol=reference,
        observable=observable,
        model_family=SpinModelFamily.HABERKORN_STATIC_PAIR,
    )


class HaberkornReferenceOracle:
    """Solve one reduced FAD/Trp-like spin system and abstain outside it."""

    @property
    def identity(self) -> OracleIdentity:
        """Return the immutable identity of this simulator-only oracle."""

        return OracleIdentity(
            oracle_id="paircraft-haberkorn-reference-oracle",
            oracle_version="1",
            artifact_sha256=_artifact_sha256(),
            code_version=__version__,
            training_manifest_id=_TRAINING_MANIFEST_ID,
            training_manifest_sha256=_TRAINING_MANIFEST_SHA256,
            parameter_catalog_version=_PARAMETER_CATALOG_VERSION,
            parameter_catalog_sha256=_PARAMETER_CATALOG_SHA256,
            supported_model_families=(SpinModelFamily.HABERKORN_STATIC_PAIR,),
            supported_observables=(
                ObservableKind.SINGLET_REACTION_YIELD,
                ObservableKind.TRIPLET_REACTION_YIELD,
                ObservableKind.SINGLET_REACTION_YIELD_MFE,
                ObservableKind.TRIPLET_REACTION_YIELD_MFE,
            ),
            calibration_tier=CalibrationTier.SIMULATOR_ONLY,
        )

    def predict(self, query: OracleQuery) -> OracleResult:
        """Return the reduced-model yield or a structured out-of-domain abstention."""

        unsupported = self._unsupported_dimensions(query)
        if unsupported:
            if "observable" in unsupported:
                reason = AbstentionReason.UNSUPPORTED_OBSERVABLE
            elif "parameter_set" in unsupported:
                reason = AbstentionReason.UNRESOLVED_PARAMETER_SET
            else:
                reason = AbstentionReason.OUTSIDE_SUPPORTED_DOMAIN
            return self._abstain(query, reason, unsupported)

        try:
            target_yields = _reaction_yields(query.target_protocol.field.components_tesla)
        except (ArithmeticError, ValueError):
            return self._abstain(query, AbstentionReason.NUMERICAL_FAILURE, ("solver",))
        target = _select_yield(query.observable, target_yields)
        if query.observable.is_relative:
            reference_protocol_value = query.reference_protocol
            if reference_protocol_value is None:  # defensive: OracleQuery already rejects this
                raise AssertionError("validated relative query lost its reference protocol")
            try:
                reference_yields = _reaction_yields(reference_protocol_value.field.components_tesla)
            except (ArithmeticError, ValueError):
                return self._abstain(query, AbstentionReason.NUMERICAL_FAILURE, ("solver",))
            reference = _select_yield(query.observable, reference_yields)
            if abs(reference) <= _REFERENCE_STABILITY_THRESHOLD:
                return self._abstain(
                    query,
                    AbstentionReason.UNCALIBRATED_UNCERTAINTY,
                    ("reference_denominator",),
                )
            prediction = (target - reference) / reference
            unit = "1"
        else:
            prediction = target
            unit = "1"

        applicability = ApplicabilityAssessment(
            status=ApplicabilityStatus.IN_DOMAIN,
            reasons=("Exact pinned reference parameter set and protocol are supported.",),
            unsupported_dimensions=(),
            support_evidence_ids=("spin-src-radicalpy-2024",),
            support_bounds=(_field_support_bound(),),
            model_family_disagreement="not assessed; this oracle executes one model family",
        )
        distribution = PredictiveDistribution(
            mean=prediction,
            lower=prediction,
            upper=prediction,
            coverage_probability=None,
            unit=unit,
            method="deterministic infinite-time Haberkorn Liouville solve",
            epistemic_standard_uncertainty=None,
            aleatoric_standard_uncertainty=None,
            simulator_discrepancy_standard_uncertainty=None,
            numerical_standard_uncertainty=None,
            decomposition_status="unavailable for deterministic reference benchmark",
            uncertainty_assumptions=(
                "All scientific parameters are fixed at their pinned benchmark values; unknown "
                "parameter and model discrepancy are not propagated."
            ),
            calibration_status="uncalibrated deterministic benchmark; no predictive coverage",
            note=(
                "Point output verifies the oracle contract and solver only; it is not an "
                "experimentally calibrated predictive distribution."
            ),
        )
        lineage = (
            LineageArtifact(
                artifact_id="spin-src-radicalpy-2024",
                sha256=_SOURCE_EVIDENCE_SHA256,
                evidence_role="model-method source",
                evidence_status="published open-access software paper",
                reuse_status="CC BY 4.0 paper; implementation independently written",
                membership="reference-only; not a training datum",
                exact_location="Methods and Haberkorn/MARY examples",
                conditions="General software method; no exact experimental sample instantiated.",
                uncertainty_note="The software-method source does not calibrate this benchmark.",
                source_uri="https://doi.org/10.1021/acs.jctc.4c00887",
                source_artifact_sha256=None,
                license="CC BY 4.0",
            ),
            *_parameter_lineage(),
        )
        return OraclePrediction(
            query=query,
            resolved_setup=ResolvedExperimentSetup(
                query.system,
                query.target_protocol,
                query.reference_protocol,
            ),
            oracle=self.identity,
            executed_model_family=SpinModelFamily.HABERKORN_STATIC_PAIR,
            distribution=distribution,
            applicability=applicability,
            lineage=lineage,
            claim_boundary="simulator-only engine-conformance output; not wet-lab validated",
        )

    def _unsupported_dimensions(self, query: OracleQuery) -> tuple[str, ...]:
        unsupported: list[str] = []
        expected = reference_system()
        if query.schema_version != _QUERY_SCHEMA_VERSION:
            unsupported.append("schema_version")
        if query.system.parameter_set_id != expected.parameter_set_id or (
            query.system.parameter_set_sha256 != expected.parameter_set_sha256
        ):
            unsupported.append("parameter_set")
        if query.system != expected:
            unsupported.append("system")
        if query.model_family is not SpinModelFamily.HABERKORN_STATIC_PAIR:
            unsupported.append("model_family")
        if query.observable not in self.identity.supported_observables:
            unsupported.append("observable")
        if not _protocol_is_supported(query.target_protocol):
            unsupported.append("target_protocol")
        if query.reference_protocol is not None and not _protocol_is_supported(
            query.reference_protocol
        ):
            unsupported.append("reference_protocol")
        return tuple(dict.fromkeys(unsupported))

    def _abstain(
        self,
        query: OracleQuery,
        reason: AbstentionReason,
        unsupported_dimensions: tuple[str, ...],
    ) -> OracleAbstention:
        applicability = ApplicabilityAssessment(
            status=ApplicabilityStatus.OUT_OF_DOMAIN,
            reasons=("The query is outside the exact pinned reference-oracle domain.",),
            unsupported_dimensions=unsupported_dimensions,
            support_evidence_ids=("spin-src-radicalpy-2024",),
            support_bounds=(_field_support_bound(),),
            model_family_disagreement="not assessed because no supported prediction was produced",
        )
        return OracleAbstention(
            query_id=query.query_id,
            oracle=self.identity,
            reason=reason,
            unsupported_dimensions=unsupported_dimensions,
            applicability=applicability,
            note="No numeric prediction was produced.",
        )


def _protocol_is_supported(protocol: ExperimentProtocol) -> bool:
    expected = reference_protocol(protocol.field.z_tesla)
    magnitude = sqrt(sum(component**2 for component in protocol.field.components_tesla))
    return (
        magnitude <= _MAX_FIELD_TESLA
        and protocol.field.coordinate_frame == expected.field.coordinate_frame
        and protocol.field.intent is FieldIntent.IDEAL_SIMULATION_TARGET
        and protocol.non_field_identity() == expected.non_field_identity()
    )


def _field_support_bound() -> SupportBound:
    return SupportBound(
        dimension="magnetic flux-density magnitude",
        lower=0.0,
        upper=_MAX_FIELD_TESLA,
        unit="T",
        definition="Euclidean magnitude of the ideal Cartesian B vector.",
    )


def _parameter_lineage() -> tuple[LineageArtifact, ...]:
    return tuple(
        LineageArtifact(
            artifact_id=parameter.parameter_id,
            sha256=sha256(
                json.dumps(asdict(parameter), sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            evidence_role="benchmark scientific parameter",
            evidence_status=parameter.evidence_status,
            reuse_status="internal benchmark declaration with cited public source where inherited",
            membership="oracle parameter set",
            exact_location=f"{parameter.source}: {parameter.exact_location}",
            conditions=parameter.conditions,
            uncertainty_note=(f"{parameter.uncertainty_status}: {parameter.uncertainty_note}"),
            source_uri=parameter.source_uri,
            source_artifact_sha256=parameter.source_artifact_sha256,
            license=parameter.license,
        )
        for parameter in _PARAMETERS
    )


def _select_yield(observable: ObservableKind, yields: tuple[float, float]) -> float:
    if observable in {
        ObservableKind.SINGLET_REACTION_YIELD,
        ObservableKind.SINGLET_REACTION_YIELD_MFE,
    }:
        return yields[0]
    return yields[1]


def _reaction_yields(field_tesla: tuple[float, float, float]) -> tuple[float, float]:
    yields = _solve_reaction_yields(_SPIN_SYSTEM, field_tesla)
    return yields.singlet, yields.triplet
