"""Published Maeda 2012 five-nucleus reduced radical-pair model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256
from math import isclose, isfinite, sqrt
from pathlib import Path
from typing import Final

import numpy as np
import scipy

from paircraft import __version__
from paircraft.spin._maeda_kernel import (
    LIOUVILLE_DIM,
    KernelHyperfine,
    MaedaKernelInputs,
    solve_maeda_kernel,
    st_dephasing,
)

_MODEL_ID: Final = "maeda-2012-five-nucleus-haberkorn-std-rp2-v1"
_ARTICLE_URI: Final = "https://pmc.ncbi.nlm.nih.gov/articles/PMC3323948/"
_ARTICLE_PDF_URI: Final = "https://pmc.ncbi.nlm.nih.gov/articles/PMC3323948/pdf/pnas.1118959109.pdf"
_ARTICLE_PDF_SHA256: Final = "cc0df2c368ca42357b369b6560212d2264780039b081aca25be50e20b1f5cc8a"
_SI_URI: Final = (
    "https://www.pnas.org/lookup/suppl/doi:10.1073/pnas.1118959109/-/DCSupplemental/Appendix.pdf"
)
_SI_SHA256: Final = "46e2dc81764dbed8ff6be4c7e5780c4963e8763f3bd34069c3008e44977c2560"
_BOHR_MAGNETON_OVER_HBAR_RAD_PER_SECOND_TESLA: Final = 87_941_000_418.53893
_LIOUVILLE_DIM: Final = LIOUVILLE_DIM
_st_dephasing = st_dephasing
_CONSERVATION_TOLERANCE: Final = 1e-7
_MAX_PUBLISHED_FIELD_TESLA: Final = 0.05


class MaedaProteinSystem(StrEnum):
    """Protein labels attached to the two published fitted rate sets."""

    ATCRY1 = "atcry1"
    ECPL_E109A = "ecpl_e109a"


class ParameterEvidence(StrEnum):
    """How a Maeda model parameter entered the specification."""

    CALCULATED = "calculated"
    FITTED = "fitted"
    ASSUMED = "assumed"


class CitationRelation(StrEnum):
    """A source's role in the published parameter chain."""

    DIRECT_MODEL_SOURCE = "direct_model_source"
    ORIGINATING_CALCULATION = "originating_calculation"
    EXPERIMENTAL_CONTEXT = "experimental_context"


@dataclass(frozen=True, slots=True)
class SourceCitation:
    """One exact source location in a Maeda parameter's citation chain."""

    source_uri: str
    exact_location: str
    relation: CitationRelation
    source_artifact_sha256: str | None

    def __post_init__(self) -> None:
        if not self.source_uri.strip() or not self.exact_location.strip():
            raise ValueError("citation URI and exact location must be explicit")
        if self.source_artifact_sha256 is not None and len(self.source_artifact_sha256) != 64:
            raise ValueError("citation artifact sha256 must contain 64 hexadecimal characters")


@dataclass(frozen=True, slots=True)
class PublishedParameter:
    """One provenance-bearing scalar in the published reduced model."""

    parameter_id: str
    value_si: float
    unit: str
    chemical_state: str
    conditions: str
    evidence_status: ParameterEvidence
    uncertainty_status: str
    uncertainty_note: str
    source_uri: str
    exact_source_location: str
    source_artifact_sha256: str | None
    citation_chain: tuple[SourceCitation, ...]
    license: str

    def __post_init__(self) -> None:
        if not isfinite(self.value_si):
            raise ValueError("published parameter value must be finite")
        required = (
            self.parameter_id,
            self.unit,
            self.chemical_state,
            self.conditions,
            self.uncertainty_status,
            self.uncertainty_note,
            self.source_uri,
            self.exact_source_location,
            self.license,
        )
        if any(not value.strip() for value in required):
            raise ValueError("published parameter provenance fields must be explicit")
        if self.source_artifact_sha256 is not None and len(self.source_artifact_sha256) != 64:
            raise ValueError("source artifact sha256 must contain 64 hexadecimal characters")
        if not self.citation_chain:
            raise ValueError("published parameter requires a citation chain")
        direct = self.citation_chain[0]
        if (
            direct.source_uri != self.source_uri
            or direct.exact_location != self.exact_source_location
            or direct.relation is not CitationRelation.DIRECT_MODEL_SOURCE
            or direct.source_artifact_sha256 != self.source_artifact_sha256
        ):
            raise ValueError("citation chain must begin with the declared direct model source")


@dataclass(frozen=True, slots=True)
class PublishedHyperfine:
    """One signed isotropic hyperfine equivalent field and nuclear assignment."""

    radical: str
    nucleus: str
    spin_quantum_number: float
    equivalent_field: PublishedParameter

    def __post_init__(self) -> None:
        if self.radical not in {"FAD radical anion", "TrpH radical cation"}:
            raise ValueError("hyperfine radical assignment is unsupported")
        if not self.nucleus.strip():
            raise ValueError("hyperfine nucleus label must be explicit")
        if self.spin_quantum_number not in {0.5, 1.0}:
            raise ValueError("Maeda reduced model supports only spin-1/2 and spin-1 nuclei")
        if self.equivalent_field.unit != "T":
            raise ValueError("hyperfine equivalent fields must use tesla")


@dataclass(frozen=True, slots=True)
class Maeda2012Specification:
    """Complete, auditable inputs for one published Maeda rate set."""

    protein_system: MaedaProteinSystem
    hyperfine_parameters: tuple[PublishedHyperfine, ...]
    forward_rate: PublishedParameter
    back_recombination_rate: PublishedParameter
    st_dephasing_rate: PublishedParameter
    electron_g_factor: PublishedParameter
    electron_gyromagnetic_ratio: PublishedParameter
    initial_nuclear_state_assumption: str
    model_id: str = _MODEL_ID

    def __post_init__(self) -> None:
        if len(self.hyperfine_parameters) != 5:
            raise ValueError("Maeda specification requires exactly five hyperfine nuclei")
        if self.model_id != _MODEL_ID:
            raise ValueError("Maeda specification model identity cannot be changed")
        for rate in (
            self.forward_rate,
            self.back_recombination_rate,
            self.st_dephasing_rate,
        ):
            if rate.unit != "s^-1" or rate.value_si < 0:
                raise ValueError("Maeda kinetic rates must be nonnegative and use s^-1")
        if self.forward_rate.value_si <= 0:
            raise ValueError("forward RP2 formation rate must be strictly positive")
        if self.electron_g_factor.unit != "1" or self.electron_g_factor.value_si <= 0:
            raise ValueError("electron g factor must be positive and dimensionless")
        if (
            self.electron_gyromagnetic_ratio.unit != "rad s^-1 T^-1"
            or self.electron_gyromagnetic_ratio.value_si <= 0
        ):
            raise ValueError("electron gyromagnetic-ratio magnitude has invalid unit or value")
        expected_gamma = (
            self.electron_g_factor.value_si * _BOHR_MAGNETON_OVER_HBAR_RAD_PER_SECOND_TESLA
        )
        if not isclose(
            self.electron_gyromagnetic_ratio.value_si,
            expected_gamma,
            rel_tol=1e-15,
        ):
            raise ValueError(
                "electron gyromagnetic ratio must be derived from the declared g factor"
            )
        if not self.initial_nuclear_state_assumption.strip():
            raise ValueError("initial nuclear-state assumption must be explicit")

    @property
    def parameter_set_sha256(self) -> str:
        """Hash all resolved values, assignments, assumptions, and provenance."""

        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class SecondaryPairYield:
    """Audited deterministic RP2 result from the published reduced model."""

    specification: Maeda2012Specification
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
    uncertainty_status: str
    claim_boundary: str

    def __post_init__(self) -> None:
        if len(self.field_tesla) != 3 or not all(isfinite(value) for value in self.field_tesla):
            raise ValueError("result field must contain three finite tesla components")
        yields = (self.secondary_pair, self.singlet_back_recombination)
        if any(not isfinite(value) or not 0 <= value <= 1 for value in yields):
            raise ValueError("reaction yields must be finite and lie in [0, 1]")
        if not isclose(self.total, 1.0, rel_tol=0.0, abs_tol=_CONSERVATION_TOLERANCE):
            raise ValueError("secondary and back-recombination yields must conserve probability")
        if self.model_id != self.specification.model_id:
            raise ValueError("result model identity contradicts its specification")
        if self.parameter_set_sha256 != self.specification.parameter_set_sha256:
            raise ValueError("result parameter hash contradicts its specification")
        if (
            self.iterations <= 0
            or not isfinite(self.relative_residual)
            or self.relative_residual < 0
            or not isfinite(self.requested_relative_tolerance)
            or not 0 < self.requested_relative_tolerance < 1
            or self.relative_residual > self.requested_relative_tolerance
        ):
            raise ValueError("solver diagnostics are invalid")
        if self.restart <= 0 or self.maximum_restart_cycles <= 0:
            raise ValueError("result solver configuration is invalid")
        if self.field_intent != "ideal_simulation_target":
            raise ValueError("Maeda results require an ideal simulation target field")
        if len(self.code_artifact_sha256) != 64 or not self.code_version:
            raise ValueError("result code identity is invalid")
        if not self.solver or not self.solver_version or not self.claim_boundary:
            raise ValueError("solver and claim-boundary metadata must be explicit")

    @property
    def total(self) -> float:
        """Return the total probability reaching either chemical channel."""

        return self.secondary_pair + self.singlet_back_recombination


def _parameter(
    parameter_id: str,
    value_si: float,
    unit: str,
    chemical_state: str,
    conditions: str,
    evidence_status: ParameterEvidence,
    location: str,
    uncertainty_note: str,
    *,
    originating_location: str | None = None,
    direct_source_uri: str = _SI_URI,
    direct_source_sha256: str | None = _SI_SHA256,
    uncertainty_status: str = "not_reported",
) -> PublishedParameter:
    citations = [
        SourceCitation(
            direct_source_uri,
            location,
            CitationRelation.DIRECT_MODEL_SOURCE,
            direct_source_sha256,
        )
    ]
    if originating_location is not None:
        citations.append(
            SourceCitation(
                _SI_URI,
                originating_location,
                CitationRelation.ORIGINATING_CALCULATION,
                _SI_SHA256,
            )
        )
    citations.append(
        SourceCitation(
            _ARTICLE_URI,
            "Methods and Figure 6 experimental context",
            CitationRelation.EXPERIMENTAL_CONTEXT,
            None,
        )
    )
    return PublishedParameter(
        parameter_id=parameter_id,
        value_si=value_si,
        unit=unit,
        chemical_state=chemical_state,
        conditions=conditions,
        evidence_status=evidence_status,
        uncertainty_status=uncertainty_status,
        uncertainty_note=uncertainty_note,
        source_uri=direct_source_uri,
        exact_source_location=location,
        source_artifact_sha256=direct_source_sha256,
        citation_chain=tuple(citations),
        license="Reuse licence not identified; numerical fact independently represented.",
    )


def _hyperfine(
    radical: str,
    nucleus: str,
    spin: float,
    equivalent_field_tesla: float,
) -> PublishedHyperfine:
    return PublishedHyperfine(
        radical=radical,
        nucleus=nucleus,
        spin_quantum_number=spin,
        equivalent_field=_parameter(
            f"{radical}-{nucleus}-isotropic-hyperfine",
            equivalent_field_tesla,
            "T",
            f"{radical}; signed isotropic equivalent magnetic flux density",
            "UB3LYP/EPR-III isotropic couplings calculated using lumiflavin as an FAD "
            "proxy, then reduced to the five-nucleus basis shared by AtCry1 and EcPL.",
            ParameterEvidence.CALCULATED,
            "Supporting Information Table S3",
            "The supporting information reports no uncertainty for this calculated input.",
            originating_location=(
                "Supporting Information Table S2 and equations 1-2: UB3LYP/EPR-III "
                "isotropic couplings calculated with lumiflavin as an FAD proxy, then "
                "lumped into the five-nucleus Table S3 basis"
            ),
        ),
    )


_HYPERFINES: Final = (
    _hyperfine("FAD radical anion", "N5", 1.0, +523.3e-6),
    _hyperfine("FAD radical anion", "pseudo-H", 0.5, +1106.5e-6),
    _hyperfine("TrpH radical cation", "H18", 0.5, +1604.6e-6),
    _hyperfine("TrpH radical cation", "H23", 0.5, -598.3e-6),
    _hyperfine("TrpH radical cation", "pseudo-H", 0.5, +915.7e-6),
)


def maeda_2012_specification(system: MaedaProteinSystem) -> Maeda2012Specification:
    """Return one of the two exact published point-estimate parameter sets."""

    back_rate, dephasing_rate = {
        MaedaProteinSystem.ATCRY1: (4.9e5, 1.1e7),
        MaedaProteinSystem.ECPL_E109A: (1.2e5, 2.7e7),
    }[system]
    conditions = (
        "AtCry1 at 270 K in 60% glycerol, pH 7.5 Tris/HCl, 500 mM NaCl and 250 mM "
        "imidazole; 460 nm excitation and 510 nm transient-absorption readout."
        if system is MaedaProteinSystem.ATCRY1
        else "EcPL E109A at 260 K in 50% glycerol, 50 mM HEPES pH 7.0 and 100 mM KCl; "
        "460 nm excitation and 510 nm transient-absorption readout."
    )
    return Maeda2012Specification(
        protein_system=system,
        hyperfine_parameters=_HYPERFINES,
        forward_rate=_parameter(
            "rp2-forward-rate",
            2.5e5,
            "s^-1",
            "spin-independent RP1 to RP2 formation",
            conditions,
            ParameterEvidence.ASSUMED,
            "Supporting Information section 5 and main-text fit description",
            "Fixed to break the common rate-scaling degeneracy; no uncertainty reported.",
        ),
        back_recombination_rate=_parameter(
            "singlet-back-recombination-rate",
            back_rate,
            "s^-1",
            "singlet-selective RP1 electron-hole recombination",
            conditions,
            ParameterEvidence.FITTED,
            "Figure 6 caption and Results fit paragraph",
            "Best-fit point estimate only; no covariance or confidence interval reported.",
            direct_source_uri=_ARTICLE_PDF_URI,
            direct_source_sha256=_ARTICLE_PDF_SHA256,
        ),
        st_dephasing_rate=_parameter(
            "singlet-triplet-dephasing-rate",
            dephasing_rate,
            "s^-1",
            "Shushin singlet-triplet dephasing of RP1",
            conditions,
            ParameterEvidence.FITTED,
            "Figure 6 caption and Results fit paragraph",
            "Best-fit point estimate only; no covariance or confidence interval reported.",
            direct_source_uri=_ARTICLE_PDF_URI,
            direct_source_sha256=_ARTICLE_PDF_SHA256,
        ),
        electron_g_factor=_parameter(
            "electron-g-factor",
            2.0,
            "1",
            "identical isotropic electron g factors on both radicals",
            conditions,
            ParameterEvidence.ASSUMED,
            "Supporting Information section 5, Hamiltonian superoperator",
            "Fixed model assumption; uncertainty not modelled.",
        ),
        electron_gyromagnetic_ratio=_parameter(
            "electron-gyromagnetic-ratio-magnitude",
            2.0 * _BOHR_MAGNETON_OVER_HBAR_RAD_PER_SECOND_TESLA,
            "rad s^-1 T^-1",
            "electron-only Zeeman and equivalent-field hyperfine conversion",
            conditions,
            ParameterEvidence.ASSUMED,
            "Paircraft computational constant declaration",
            "Derived from the declared g=2 and a versioned Paircraft CODATA-2022 numerical "
            "convention; physical-constant uncertainty is not propagated.",
            direct_source_uri="paircraft:codata-2022-bohr-magneton-over-hbar",
            direct_source_sha256=None,
            uncertainty_status="not_propagated",
        ),
        initial_nuclear_state_assumption=(
            "Paircraft assumes a maximally mixed nuclear spin state because the publication "
            "specifies a pure electronic singlet but does not state nuclear polarization."
        ),
    )


def _kernel_inputs(specification: Maeda2012Specification) -> MaedaKernelInputs:
    """Project an audited specification into the provenance-neutral numerical kernel."""

    return MaedaKernelInputs(
        hyperfine_parameters=tuple(
            KernelHyperfine(
                radical_index=0 if parameter.radical == "FAD radical anion" else 1,
                spin_quantum_number=parameter.spin_quantum_number,
                equivalent_field_tesla=parameter.equivalent_field.value_si,
            )
            for parameter in specification.hyperfine_parameters
        ),
        electron_gyromagnetic_ratio_rad_per_second_tesla=(
            specification.electron_gyromagnetic_ratio.value_si
        ),
        forward_rate_per_second=specification.forward_rate.value_si,
        back_recombination_rate_per_second=specification.back_recombination_rate.value_si,
        st_dephasing_rate_per_second=specification.st_dephasing_rate.value_si,
    )


def _code_artifact_sha256() -> str:
    digest = sha256()
    for path in (Path(__file__), Path(__file__).with_name("_maeda_kernel.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def solve_maeda_secondary_pair_yield(
    specification: Maeda2012Specification,
    field_tesla: tuple[float, float, float],
    *,
    relative_tolerance: float = 1e-8,
    restart: int = 100,
    maximum_restart_cycles: int = 50,
) -> SecondaryPairYield:
    """Calculate RP2 yield for one exact published rate set and ideal **B** vector."""

    expected = maeda_2012_specification(specification.protein_system)
    if specification != expected:
        raise ValueError(
            "public Maeda solve accepts only an exact published specification; custom "
            "sensitivity systems require a separate non-published provenance boundary"
        )
    magnitude = sqrt(sum(component**2 for component in field_tesla))
    if magnitude > _MAX_PUBLISHED_FIELD_TESLA:
        raise ValueError("magnetic flux-density magnitude exceeds the published 0-50 mT domain")
    return _solve_maeda_secondary_pair_yield(
        specification,
        field_tesla,
        relative_tolerance=relative_tolerance,
        restart=restart,
        maximum_restart_cycles=maximum_restart_cycles,
    )


def _solve_maeda_secondary_pair_yield(
    specification: Maeda2012Specification,
    field_tesla: tuple[float, float, float],
    *,
    relative_tolerance: float = 1e-8,
    restart: int = 100,
    maximum_restart_cycles: int = 50,
) -> SecondaryPairYield:
    """Internal numerical solve used by exact public calls and analytical tests."""

    numerical = solve_maeda_kernel(
        _kernel_inputs(specification),
        field_tesla,
        relative_tolerance=relative_tolerance,
        restart=restart,
        maximum_restart_cycles=maximum_restart_cycles,
    )
    return SecondaryPairYield(
        specification=specification,
        field_tesla=field_tesla,
        secondary_pair=numerical.secondary_pair,
        singlet_back_recombination=numerical.singlet_back_recombination,
        relative_residual=numerical.relative_residual,
        iterations=numerical.iterations,
        requested_relative_tolerance=relative_tolerance,
        solver="energy-basis Jacobi-preconditioned GMRES",
        solver_version=f"numpy-{np.__version__}; scipy-{scipy.__version__}",
        code_version=__version__,
        code_artifact_sha256=_code_artifact_sha256(),
        restart=restart,
        maximum_restart_cycles=maximum_restart_cycles,
        field_intent="ideal_simulation_target",
        model_id=specification.model_id,
        parameter_set_sha256=specification.parameter_set_sha256,
        uncertainty_status=(
            "not propagated; published fitted rates have no reported covariance and share a "
            "common scaling non-identifiability"
        ),
        claim_boundary=(
            "simulator-only reproduction of a published reduced point-estimate model; not "
            "experimental calibration, sequence prediction, or sub-Earth-field validation"
        ),
    )
