"""Explicitly hypothetical one-at-a-time sensitivity designs for the Maeda model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256
from math import exp, isclose, isfinite, sqrt
from pathlib import Path
from typing import Final

import numpy as np
import scipy

from paircraft import __version__
from paircraft.spin._maeda_kernel import (
    KernelHyperfine,
    MaedaKernelInputs,
    solve_maeda_kernel,
)
from paircraft.spin.maeda2012 import (
    Maeda2012Specification,
    MaedaProteinSystem,
    maeda_2012_specification,
)

_MODEL_ID: Final = "paircraft-maeda-2012-hypothetical-oat-sensitivity-v1"
_DESIGN_VERSION: Final = "paircraft-maeda-oat-log-scale-design-v1"
_MAX_FIELD_TESLA: Final = 0.05
_CONSERVATION_TOLERANCE: Final = 1e-7
_TRANSFORMATION: Final = "proposed_value_si = baseline_value_si * exp(log_scale_delta)"
_EVIDENCE_STATUS: Final = "assumed_hypothetical_design"
_DESIGN_SOURCE_URI: Final = "paircraft:maeda-hypothetical-oat-design-v1"
_DESIGN_EXACT_LOCATION: Final = "MaedaSensitivitySpecification.override"
_OVERRIDE_UNCERTAINTY_STATUS: Final = (
    "not_applicable; enumerated design point, not an uncertainty statement"
)
_RATIONALE: Final = (
    "A deterministic local sensitivity probe, not a reported uncertainty interval, probability "
    "distribution, biological variation, or mutation effect."
)
_UNCERTAINTY_STATUS: Final = (
    "not calculated; the enumerated assumed transform is not a parameter distribution or "
    "uncertainty interval"
)
_CLAIM_BOUNDARY: Final = (
    "simulator-only hypothetical one-at-a-time sensitivity result around an inherited published "
    "point estimate; not experimental calibration, uncertainty propagation, biological "
    "variability, mutation or sequence prediction, or sub-Earth-field validation"
)
_SOLVER: Final = "energy-basis Jacobi-preconditioned GMRES"


class SensitivityDimension(StrEnum):
    """The eight existing scalar model inputs eligible for the v1 local design."""

    FAD_N5_HYPERFINE = "fad-n5-isotropic-hyperfine"
    FAD_PSEUDO_H_HYPERFINE = "fad-pseudo-h-isotropic-hyperfine"
    TRPH_H18_HYPERFINE = "trph-h18-isotropic-hyperfine"
    TRPH_H23_HYPERFINE = "trph-h23-isotropic-hyperfine"
    TRPH_PSEUDO_H_HYPERFINE = "trph-pseudo-h-isotropic-hyperfine"
    FORWARD_RATE = "rp2-forward-rate"
    BACK_RECOMBINATION_RATE = "singlet-back-recombination-rate"
    ST_DEPHASING_RATE = "singlet-triplet-dephasing-rate"


class SensitivityDirection(StrEnum):
    """Fixed local log-scale offsets; these are design points, not random draws."""

    LOWER = "lower-exp-minus-0.05"
    BASELINE = "baseline-exp-0"
    HIGHER = "higher-exp-plus-0.05"

    @property
    def log_scale_delta(self) -> float:
        """Return the exact deterministic log-scale design offset."""

        return {
            SensitivityDirection.LOWER: -0.05,
            SensitivityDirection.BASELINE: 0.0,
            SensitivityDirection.HIGHER: 0.05,
        }[self]


@dataclass(frozen=True, slots=True)
class HypotheticalParameterOverride:
    """One assumed transform whose published baseline evidence remains separate."""

    dimension: SensitivityDimension
    direction: SensitivityDirection
    baseline_parameter_id: str
    baseline_parameter_sha256: str
    baseline_value_si: float
    proposed_value_si: float
    unit: str
    chemical_state: str
    conditions: str
    log_scale_delta: float
    transformation: str = _TRANSFORMATION
    evidence_status: str = _EVIDENCE_STATUS
    design_source_uri: str = _DESIGN_SOURCE_URI
    design_exact_location: str = _DESIGN_EXACT_LOCATION
    uncertainty_status: str = _OVERRIDE_UNCERTAINTY_STATUS
    rationale: str = _RATIONALE

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, SensitivityDimension):
            raise ValueError("sensitivity dimension must be a SensitivityDimension")
        if not isinstance(self.direction, SensitivityDirection):
            raise ValueError("sensitivity direction must be a SensitivityDirection")
        required = (
            self.baseline_parameter_id,
            self.unit,
            self.chemical_state,
            self.conditions,
            self.design_source_uri,
            self.design_exact_location,
            self.uncertainty_status,
        )
        if any(not value.strip() for value in required):
            raise ValueError("hypothetical parameter context and provenance must be explicit")
        if len(self.baseline_parameter_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.baseline_parameter_sha256
        ):
            raise ValueError("baseline parameter sha256 must be a lowercase digest")
        if not isfinite(self.baseline_value_si) or not isfinite(self.proposed_value_si):
            raise ValueError("sensitivity values must be finite")
        if self.baseline_value_si == 0:
            raise ValueError("log-scale sensitivity requires a nonzero baseline")
        expected_delta = self.direction.log_scale_delta
        if self.log_scale_delta != expected_delta:
            raise ValueError("sensitivity log-scale delta contradicts its direction")
        expected_value = self.baseline_value_si * exp(expected_delta)
        if not isclose(self.proposed_value_si, expected_value, rel_tol=1e-15):
            raise ValueError("proposed sensitivity value contradicts the fixed transform")
        if (
            self.transformation != _TRANSFORMATION
            or self.evidence_status != _EVIDENCE_STATUS
            or self.design_source_uri != _DESIGN_SOURCE_URI
            or self.design_exact_location != _DESIGN_EXACT_LOCATION
            or self.uncertainty_status != _OVERRIDE_UNCERTAINTY_STATUS
            or self.rationale != _RATIONALE
        ):
            raise ValueError("hypothetical sensitivity provenance boundary is fixed")


@dataclass(frozen=True, slots=True)
class MaedaSensitivitySpecification:
    """One content-addressed hypothetical design around an exact published baseline."""

    protein_system: MaedaProteinSystem
    baseline: Maeda2012Specification
    override: HypotheticalParameterOverride
    design_version: str = _DESIGN_VERSION
    model_id: str = _MODEL_ID

    def __post_init__(self) -> None:
        if not isinstance(self.protein_system, MaedaProteinSystem):
            raise ValueError("sensitivity protein system must be a MaedaProteinSystem")
        if self.baseline != maeda_2012_specification(self.protein_system):
            raise ValueError("sensitivity baseline must be the exact published specification")
        if not isinstance(self.override, HypotheticalParameterOverride):
            raise ValueError("sensitivity specification requires one typed override")
        if (
            self.override.direction is SensitivityDirection.BASELINE
            and self.override.dimension is not SensitivityDimension.FORWARD_RATE
        ):
            raise ValueError("baseline design is canonical only for the forward-rate dimension")
        expected = _override_for(
            self.baseline,
            self.override.dimension,
            self.override.direction,
        )
        if self.override != expected:
            raise ValueError("sensitivity override does not match its inherited baseline")
        if self.design_version != _DESIGN_VERSION or self.model_id != _MODEL_ID:
            raise ValueError("hypothetical sensitivity design identity is fixed")

    @property
    def baseline_parameter_set_sha256(self) -> str:
        """Return the exact inherited published parameter-set identity."""

        return self.baseline.parameter_set_sha256

    @property
    def parameter_set_sha256(self) -> str:
        """Hash the baseline, assumed transform, and distinct model identity."""

        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class HypotheticalSecondaryPairYield:
    """Audited deterministic result for one explicitly hypothetical design point."""

    specification: MaedaSensitivitySpecification
    field_tesla: tuple[float, float, float]
    secondary_pair: float
    singlet_back_recombination: float
    relative_residual: float
    iterations: int
    requested_relative_tolerance: float
    solver: str
    solver_version: str
    code_version: str
    code_artifact_sha256: str
    restart: int
    maximum_restart_cycles: int
    field_intent: str
    model_id: str
    parameter_set_sha256: str
    baseline_parameter_set_sha256: str
    uncertainty_status: str
    claim_boundary: str

    def __post_init__(self) -> None:
        if len(self.field_tesla) != 3 or not all(isfinite(value) for value in self.field_tesla):
            raise ValueError("result field must contain three finite tesla components")
        yields = (self.secondary_pair, self.singlet_back_recombination)
        if any(not isfinite(value) or not 0 <= value <= 1 for value in yields):
            raise ValueError("reaction yields must be finite and lie in [0, 1]")
        if not isclose(sum(yields), 1.0, rel_tol=0.0, abs_tol=_CONSERVATION_TOLERANCE):
            raise ValueError("hypothetical reaction yields must conserve probability")
        if self.model_id != self.specification.model_id:
            raise ValueError("hypothetical result model identity contradicts its specification")
        if self.parameter_set_sha256 != self.specification.parameter_set_sha256:
            raise ValueError("hypothetical result parameter hash contradicts its specification")
        if self.baseline_parameter_set_sha256 != (self.specification.baseline_parameter_set_sha256):
            raise ValueError("hypothetical result lost its published baseline identity")
        if (
            self.iterations <= 0
            or not isfinite(self.relative_residual)
            or self.relative_residual < 0
            or not 0 < self.requested_relative_tolerance < 1
            or self.relative_residual > self.requested_relative_tolerance
        ):
            raise ValueError("hypothetical solver diagnostics are invalid")
        if self.restart <= 0 or self.maximum_restart_cycles <= 0:
            raise ValueError("hypothetical solver configuration is invalid")
        if self.field_intent != "ideal_simulation_target":
            raise ValueError("hypothetical results require an ideal simulation target field")
        if self.solver != _SOLVER or self.solver_version != _solver_version():
            raise ValueError("hypothetical result solver identity is fixed")
        if self.code_version != __version__:
            raise ValueError("hypothetical result code version is fixed")
        if self.code_artifact_sha256 != _code_artifact_sha256():
            raise ValueError("hypothetical result code identity is invalid")
        if self.uncertainty_status != _UNCERTAINTY_STATUS:
            raise ValueError("hypothetical result cannot claim predictive uncertainty")
        if self.claim_boundary != _CLAIM_BOUNDARY:
            raise ValueError("hypothetical result claim boundary is fixed")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def _baseline_parameter(
    baseline: Maeda2012Specification,
    dimension: SensitivityDimension,
) -> tuple[str, float, str, str, str, str]:
    hyperfine_index = {
        SensitivityDimension.FAD_N5_HYPERFINE: 0,
        SensitivityDimension.FAD_PSEUDO_H_HYPERFINE: 1,
        SensitivityDimension.TRPH_H18_HYPERFINE: 2,
        SensitivityDimension.TRPH_H23_HYPERFINE: 3,
        SensitivityDimension.TRPH_PSEUDO_H_HYPERFINE: 4,
    }.get(dimension)
    if hyperfine_index is not None:
        hyperfine = baseline.hyperfine_parameters[hyperfine_index]
        parameter = hyperfine.equivalent_field
        return (
            parameter.parameter_id,
            parameter.value_si,
            parameter.unit,
            parameter.chemical_state,
            parameter.conditions,
            _canonical_sha256(asdict(hyperfine)),
        )
    parameter = {
        SensitivityDimension.FORWARD_RATE: baseline.forward_rate,
        SensitivityDimension.BACK_RECOMBINATION_RATE: baseline.back_recombination_rate,
        SensitivityDimension.ST_DEPHASING_RATE: baseline.st_dephasing_rate,
    }[dimension]
    return (
        parameter.parameter_id,
        parameter.value_si,
        parameter.unit,
        parameter.chemical_state,
        parameter.conditions,
        _canonical_sha256(asdict(parameter)),
    )


def _override_for(
    baseline: Maeda2012Specification,
    dimension: SensitivityDimension,
    direction: SensitivityDirection,
) -> HypotheticalParameterOverride:
    parameter_id, value, unit, chemical_state, conditions, digest = _baseline_parameter(
        baseline, dimension
    )
    delta = direction.log_scale_delta
    return HypotheticalParameterOverride(
        dimension=dimension,
        direction=direction,
        baseline_parameter_id=parameter_id,
        baseline_parameter_sha256=digest,
        baseline_value_si=value,
        proposed_value_si=value * exp(delta),
        unit=unit,
        chemical_state=chemical_state,
        conditions=conditions,
        log_scale_delta=delta,
    )


def maeda_sensitivity_specification(
    system: MaedaProteinSystem,
    dimension: SensitivityDimension,
    direction: SensitivityDirection,
) -> MaedaSensitivitySpecification:
    """Construct one exact enumerated local sensitivity design point."""

    if not isinstance(system, MaedaProteinSystem):
        raise ValueError("sensitivity system must be a MaedaProteinSystem")
    if not isinstance(dimension, SensitivityDimension):
        raise ValueError("sensitivity dimension must be a SensitivityDimension")
    if not isinstance(direction, SensitivityDirection):
        raise ValueError("sensitivity direction must be a SensitivityDirection")
    if (
        direction is SensitivityDirection.BASELINE
        and dimension is not SensitivityDimension.FORWARD_RATE
    ):
        raise ValueError("baseline design is canonical only for the forward-rate dimension")
    baseline = maeda_2012_specification(system)
    return MaedaSensitivitySpecification(
        protein_system=system,
        baseline=baseline,
        override=_override_for(baseline, dimension, direction),
    )


def _kernel_inputs(specification: MaedaSensitivitySpecification) -> MaedaKernelInputs:
    baseline = specification.baseline
    override = specification.override
    hyperfines: list[KernelHyperfine] = []
    hyperfine_dimensions = tuple(SensitivityDimension)[:5]
    for index, parameter in enumerate(baseline.hyperfine_parameters):
        value = (
            override.proposed_value_si
            if override.dimension is hyperfine_dimensions[index]
            else parameter.equivalent_field.value_si
        )
        hyperfines.append(
            KernelHyperfine(
                radical_index=0 if parameter.radical == "FAD radical anion" else 1,
                spin_quantum_number=parameter.spin_quantum_number,
                equivalent_field_tesla=value,
            )
        )
    return MaedaKernelInputs(
        hyperfine_parameters=tuple(hyperfines),
        electron_gyromagnetic_ratio_rad_per_second_tesla=(
            baseline.electron_gyromagnetic_ratio.value_si
        ),
        forward_rate_per_second=(
            override.proposed_value_si
            if override.dimension is SensitivityDimension.FORWARD_RATE
            else baseline.forward_rate.value_si
        ),
        back_recombination_rate_per_second=(
            override.proposed_value_si
            if override.dimension is SensitivityDimension.BACK_RECOMBINATION_RATE
            else baseline.back_recombination_rate.value_si
        ),
        st_dephasing_rate_per_second=(
            override.proposed_value_si
            if override.dimension is SensitivityDimension.ST_DEPHASING_RATE
            else baseline.st_dephasing_rate.value_si
        ),
    )


def _code_artifact_sha256() -> str:
    digest = sha256()
    for path in (
        Path(__file__),
        Path(__file__).with_name("_maeda_kernel.py"),
        Path(__file__).with_name("maeda2012.py"),
    ):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _solver_version() -> str:
    return f"numpy-{np.__version__}; scipy-{scipy.__version__}"


def solve_maeda_sensitivity_yield(
    specification: MaedaSensitivitySpecification,
    field_tesla: tuple[float, float, float],
    *,
    relative_tolerance: float = 1e-8,
    restart: int = 100,
    maximum_restart_cycles: int = 50,
) -> HypotheticalSecondaryPairYield:
    """Solve one predeclared hypothetical sensitivity design at ideal **B**."""

    if not isinstance(specification, MaedaSensitivitySpecification):
        raise ValueError("hypothetical solve requires a MaedaSensitivitySpecification")
    magnitude = sqrt(sum(component**2 for component in field_tesla))
    if magnitude > _MAX_FIELD_TESLA:
        raise ValueError("magnetic flux-density magnitude exceeds the 0-50 mT design domain")
    numerical = solve_maeda_kernel(
        _kernel_inputs(specification),
        field_tesla,
        relative_tolerance=relative_tolerance,
        restart=restart,
        maximum_restart_cycles=maximum_restart_cycles,
    )
    return HypotheticalSecondaryPairYield(
        specification=specification,
        field_tesla=field_tesla,
        secondary_pair=numerical.secondary_pair,
        singlet_back_recombination=numerical.singlet_back_recombination,
        relative_residual=numerical.relative_residual,
        iterations=numerical.iterations,
        requested_relative_tolerance=relative_tolerance,
        solver=_SOLVER,
        solver_version=_solver_version(),
        code_version=__version__,
        code_artifact_sha256=_code_artifact_sha256(),
        restart=restart,
        maximum_restart_cycles=maximum_restart_cycles,
        field_intent="ideal_simulation_target",
        model_id=specification.model_id,
        parameter_set_sha256=specification.parameter_set_sha256,
        baseline_parameter_set_sha256=specification.baseline_parameter_set_sha256,
        uncertainty_status=_UNCERTAINTY_STATUS,
        claim_boundary=_CLAIM_BOUNDARY,
    )
