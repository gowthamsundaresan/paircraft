"""Four-state photocycle/readout model with auditable scientific inputs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import pairwise
from math import isclose, isfinite
from typing import Final, Self

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

_PREPRINT_DOI: Final = "https://doi.org/10.64898/2026.07.08.737352"
_NORMALIZATION_ATOL: Final = 1e-12
_NUMERICAL_NEGATIVE_ATOL: Final = 1e-10


class EvidenceStatus(StrEnum):
    """How a scientific input entered the model."""

    MEASURED = "measured"
    FITTED = "fitted"
    CALCULATED = "calculated"
    ASSUMED = "assumed"
    INHERITED = "inherited"


class UncertaintyStatus(StrEnum):
    """Whether a source reports parameter uncertainty."""

    REPORTED = "reported"
    NOT_REPORTED = "not_reported"
    UNKNOWN = "unknown"


class CitationRelation(StrEnum):
    """A source's role in the provenance chain."""

    MODEL_SOURCE = "model_source"
    ORIGINATING_SOURCE = "originating_source"


class SourceLocationStatus(StrEnum):
    """Whether the cited location was directly verified."""

    VERIFIED = "verified"
    UNVERIFIED = "unverified"


@dataclass(frozen=True, slots=True)
class Citation:
    """One structured link in a parameter's citation chain."""

    identifier: str
    location: str
    relation: CitationRelation
    location_status: SourceLocationStatus

    def __post_init__(self) -> None:
        if not self.identifier:
            raise ValueError("citation identifier must be explicit")
        if not self.location:
            raise ValueError("citation location must be explicit")


@dataclass(frozen=True, slots=True)
class Uncertainty:
    """Uncertainty attached to a scientific parameter."""

    status: UncertaintyStatus
    unit: str
    value: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if not self.unit:
            raise ValueError("uncertainty unit must be explicit")
        if self.status is UncertaintyStatus.REPORTED:
            if self.value is None or not isfinite(self.value) or self.value < 0:
                raise ValueError("reported uncertainty requires a finite nonnegative value")
        elif self.value is not None:
            raise ValueError("unreported or unknown uncertainty cannot carry a numeric value")
        if not self.note:
            raise ValueError("uncertainty note must explain its status")


@dataclass(frozen=True, slots=True)
class ParameterMetadata:
    """Units, provenance, conditions, and uncertainty for one input."""

    unit: str
    evidence_status: EvidenceStatus
    source: str
    source_location: str
    citation_chain: tuple[Citation, ...]
    chemical_state: str
    conditions: str
    uncertainty: Uncertainty

    def __post_init__(self) -> None:
        for field_name in ("unit", "source", "source_location", "chemical_state", "conditions"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must be explicit")
        if not self.citation_chain:
            raise ValueError("citation chain must retain at least the direct model source")
        direct = self.citation_chain[0]
        if (
            direct.identifier != self.source
            or direct.location != self.source_location
            or direct.relation is not CitationRelation.MODEL_SOURCE
        ):
            raise ValueError("citation chain must begin with the declared direct model source")
        if self.uncertainty.unit != self.unit:
            raise ValueError("parameter and uncertainty units must match")


@dataclass(frozen=True, slots=True)
class RateConstant:
    """A finite nonnegative first-order rate constant in inverse seconds."""

    value_per_second: float
    metadata: ParameterMetadata

    def __post_init__(self) -> None:
        if self.metadata.unit != "s^-1":
            raise ValueError("rate constants must use s^-1")
        if not isfinite(self.value_per_second) or self.value_per_second < 0:
            raise ValueError("rate constant must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class SingletYield:
    """An imposed dimensionless singlet yield; never inferred from magnetic field here."""

    value: float
    metadata: ParameterMetadata

    def __post_init__(self) -> None:
        if self.metadata.unit != "1":
            raise ValueError("singlet yield must be dimensionless with unit '1'")
        if not isfinite(self.value) or not 0.0 <= self.value <= 1.0:
            raise ValueError("singlet yield must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class FourStateRates:
    """Kinetic inputs to equations 1-4 of the cited preprint."""

    excitation: RateConstant
    fluorescence: RateConstant
    electron_transfer: RateConstant
    singlet_back_electron_transfer: RateConstant
    recovery: RateConstant

    def with_excitation(self, excitation: RateConstant) -> Self:
        """Return the same kinetic system at another imposed excitation rate."""

        return replace(self, excitation=excitation)


@dataclass(frozen=True, slots=True)
class PopulationState:
    """Normalized populations of F, F*, RP_S, and RP_T."""

    ground: float
    excited: float
    singlet_pair: float
    triplet_pair: float
    total: float = 1.0

    def __post_init__(self) -> None:
        values = self.as_tuple()
        if not isfinite(self.total) or self.total <= 0:
            raise ValueError("total population must be finite and positive")
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValueError("all four populations must be finite and nonnegative")
        if not isclose(sum(values), self.total, rel_tol=1e-12, abs_tol=_NORMALIZATION_ATOL):
            raise ValueError("the four populations must sum to the explicit total")

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return populations in the model's fixed state order."""

        return (self.ground, self.excited, self.singlet_pair, self.triplet_pair)


def ground_state(total: float = 1.0) -> PopulationState:
    """Return an initial condition with every flavin in its ground state."""

    return PopulationState(total, 0.0, 0.0, 0.0, total)


@dataclass(frozen=True, slots=True)
class FourStateModel:
    """Linear four-state kinetics for one imposed singlet yield."""

    rates: FourStateRates
    singlet_yield: SingletYield

    def transition_matrix(self) -> NDArray[np.float64]:
        """Return the equation 1-4 generator in state order F, F*, RP_S, RP_T."""

        k_exc = self.rates.excitation.value_per_second
        k_f = self.rates.fluorescence.value_per_second
        k_et = self.rates.electron_transfer.value_per_second
        k_sbet = self.rates.singlet_back_electron_transfer.value_per_second
        k_rec = self.rates.recovery.value_per_second
        phi_s = self.singlet_yield.value
        return np.array(
            [
                [-k_exc, k_f, k_rec + k_sbet, k_rec],
                [k_exc, -(k_f + k_et), 0.0, 0.0],
                [0.0, phi_s * k_et, -(k_rec + k_sbet), 0.0],
                [0.0, (1.0 - phi_s) * k_et, 0.0, -k_rec],
            ],
            dtype=np.float64,
        )

    def evolve(
        self, initial: PopulationState, times_seconds: tuple[float, ...]
    ) -> tuple[PopulationState, ...]:
        """Numerically evolve the linear ODE at sorted, nonnegative times."""

        if not times_seconds:
            raise ValueError("at least one time is required")
        if any(not isfinite(time) or time < 0 for time in times_seconds):
            raise ValueError("times must be finite and nonnegative")
        if any(later <= earlier for earlier, later in pairwise(times_seconds)):
            raise ValueError("times must be strictly increasing")
        if times_seconds[-1] == 0:
            return (initial,)

        initial_array = np.asarray(initial.as_tuple(), dtype=np.float64)
        matrix = self.transition_matrix()

        def derivative(_time: float, populations: NDArray[np.float64]) -> NDArray[np.float64]:
            return matrix @ populations

        solution = solve_ivp(
            derivative,
            (0.0, times_seconds[-1]),
            initial_array,
            method="BDF",
            t_eval=np.asarray(times_seconds),
            jac=matrix,
            rtol=1e-11,
            atol=1e-13,
        )
        if not solution.success:
            raise ArithmeticError(f"numerical evolution failed: {solution.message}")
        return tuple(
            _population_from_numerical(solution.y[:, index], initial.total)
            for index in range(solution.y.shape[1])
        )

    def steady_state(self, total: float = 1.0) -> PopulationState:
        """Return supplementary equations 16-20 for the unique steady state."""

        if not isfinite(total) or total <= 0:
            raise ValueError("total population must be finite and positive")
        k_exc = self.rates.excitation.value_per_second
        k_f = self.rates.fluorescence.value_per_second
        k_et = self.rates.electron_transfer.value_per_second
        k_sbet = self.rates.singlet_back_electron_transfer.value_per_second
        k_rec = self.rates.recovery.value_per_second
        if k_f + k_et <= 0 or k_rec <= 0:
            raise ValueError("steady state is non-unique or singular for these rates")

        phi_s = self.singlet_yield.value
        a = k_exc / (k_f + k_et)
        gamma = k_et * (k_rec + (1.0 - phi_s) * k_sbet) / (k_rec * (k_rec + k_sbet))
        ground = total / (1.0 + a * (1.0 + gamma))
        excited = a * ground
        singlet_pair = phi_s * k_et * excited / (k_rec + k_sbet)
        triplet_pair = (1.0 - phi_s) * k_et * excited / k_rec
        return PopulationState(ground, excited, singlet_pair, triplet_pair, total)

    def fluorescence_emission_rate(self, state: PopulationState) -> float:
        """Return equation 6, k_f[F*], in population units per second."""

        return self.rates.fluorescence.value_per_second * state.excited

    def steady_state_emission_rate(self, total: float = 1.0) -> float:
        """Return the equation 21 steady-state fluorescence emission rate."""

        return self.fluorescence_emission_rate(self.steady_state(total))

    def relative_steady_state_change(self, target_yield: SingletYield) -> float:
        """Return `(I_target - I_reference) / I_reference` for imposed yields.

        This method contains no magnetic-field input and makes no spin-dynamics claim.
        """

        reference = self.steady_state_emission_rate()
        if reference <= 0:
            raise ValueError("relative fluorescence is undefined for zero reference emission")
        target = replace(self, singlet_yield=target_yield).steady_state_emission_rate()
        return (target - reference) / reference

    def singlet_triplet_contrast(self) -> float:
        """Return equation 11 for imposed pure-singlet and pure-triplet yields."""

        pure_singlet = replace(self, singlet_yield=_equation_11_boundary_yield(1.0))
        pure_triplet = replace(self, singlet_yield=_equation_11_boundary_yield(0.0))
        singlet_emission = pure_singlet.steady_state_emission_rate()
        if singlet_emission <= 0:
            raise ValueError("contrast is undefined for zero pure-singlet emission")
        triplet_emission = pure_triplet.steady_state_emission_rate()
        return (singlet_emission - triplet_emission) / singlet_emission


@dataclass(frozen=True, slots=True)
class ReferenceInputs:
    """Reference defaults transcribed from Table I of the cited preprint."""

    rates: FourStateRates
    field_off_yield: SingletYield
    field_on_yield: SingletYield


def ross_2026_reference_inputs() -> ReferenceInputs:
    """Return the paper defaults with their stated evidence limitations.

    The preprint calls its singlet-yield choices somewhat arbitrary. These defaults
    reproduce the readout model only and are not MagLOV2 spin parameters.
    """

    rates_location = "Ross, Lodesani & Aiello (2026), Table I, pp. 5-6; equations 1-4"
    yield_location = "Ross, Lodesani & Aiello (2026), Table I, p. 6"
    return ReferenceInputs(
        rates=FourStateRates(
            excitation=_reference_rate(
                0.6,
                EvidenceStatus.ASSUMED,
                rates_location,
                "Model default corresponding to stated 450 nm illumination of 5 mW cm^-2.",
            ),
            fluorescence=_reference_rate(
                2e8,
                EvidenceStatus.INHERITED,
                rates_location,
                "Model default based on reported flavin-protein fluorescence lifetimes.",
                origin=Citation(
                    "https://doi.org/10.1039/C3PP50414J",
                    "Specific lifetime location not verified in the originating article.",
                    CitationRelation.ORIGINATING_SOURCE,
                    SourceLocationStatus.UNVERIFIED,
                ),
            ),
            electron_transfer=_reference_rate(
                1e10,
                EvidenceStatus.INHERITED,
                rates_location,
                "Model default inherited from the preprint's reference 11.",
                origin=Citation(
                    "https://doi.org/10.1038/nchem.2447",
                    "Specific electron-transfer-rate location not verified in the "
                    "originating article.",
                    CitationRelation.ORIGINATING_SOURCE,
                    SourceLocationStatus.UNVERIFIED,
                ),
            ),
            singlet_back_electron_transfer=_reference_rate(
                2e6,
                EvidenceStatus.INHERITED,
                rates_location,
                "Model default inherited from the preprint's reference 11.",
                origin=Citation(
                    "https://doi.org/10.1038/nchem.2447",
                    "Specific back-electron-transfer-rate location not verified in the "
                    "originating article.",
                    CitationRelation.ORIGINATING_SOURCE,
                    SourceLocationStatus.UNVERIFIED,
                ),
            ),
            recovery=_reference_rate(
                0.01,
                EvidenceStatus.ASSUMED,
                rates_location,
                "Effective slow termination rate motivated by minute-scale MagLOV2 chemistry.",
                origin=Citation(
                    "https://doi.org/10.1038/s41586-025-09971-3",
                    "Specific recovery-rate derivation is not reported or verified in the "
                    "originating article.",
                    CitationRelation.ORIGINATING_SOURCE,
                    SourceLocationStatus.UNVERIFIED,
                ),
            ),
        ),
        field_off_yield=_reference_yield(0.208, yield_location, "External field OFF input."),
        field_on_yield=_reference_yield(0.167, yield_location, "External field ON input."),
    )


def _reference_rate(
    value: float,
    status: EvidenceStatus,
    source_location: str,
    conditions: str,
    origin: Citation | None = None,
) -> RateConstant:
    direct = Citation(
        _PREPRINT_DOI,
        source_location,
        CitationRelation.MODEL_SOURCE,
        SourceLocationStatus.VERIFIED,
    )
    return RateConstant(
        value,
        ParameterMetadata(
            unit="s^-1",
            evidence_status=status,
            source=_PREPRINT_DOI,
            source_location=source_location,
            citation_chain=(direct,) if origin is None else (direct, origin),
            chemical_state="F, F*, RP_S, and RP_T states defined by preprint equations 1-4.",
            conditions=conditions,
            uncertainty=Uncertainty(
                UncertaintyStatus.NOT_REPORTED,
                "s^-1",
                note="The preprint does not report uncertainty for this model default.",
            ),
        ),
    )


def _reference_yield(value: float, source_location: str, conditions: str) -> SingletYield:
    direct = Citation(
        _PREPRINT_DOI,
        source_location,
        CitationRelation.MODEL_SOURCE,
        SourceLocationStatus.VERIFIED,
    )
    origin = Citation(
        "https://arxiv.org/abs/2604.18608",
        "Specific yield expression and its mapping to these defaults not verified.",
        CitationRelation.ORIGINATING_SOURCE,
        SourceLocationStatus.UNVERIFIED,
    )
    return SingletYield(
        value,
        ParameterMetadata(
            unit="1",
            evidence_status=EvidenceStatus.ASSUMED,
            source=_PREPRINT_DOI,
            source_location=source_location,
            citation_chain=(direct, origin),
            chemical_state="Imposed average population of the RP_S state in the four-state model.",
            conditions=(
                f"{conditions} The authors describe the selected flavoprotein yields as "
                "somewhat arbitrary and requiring a quantum model."
            ),
            uncertainty=Uncertainty(
                UncertaintyStatus.NOT_REPORTED,
                "1",
                note="The preprint does not report uncertainty for this imposed yield.",
            ),
        ),
    )


def _equation_11_boundary_yield(value: float) -> SingletYield:
    location = "Ross, Lodesani & Aiello (2026), equation 11, p. 8"
    return SingletYield(
        value,
        ParameterMetadata(
            unit="1",
            evidence_status=EvidenceStatus.ASSUMED,
            source=_PREPRINT_DOI,
            source_location=location,
            citation_chain=(
                Citation(
                    _PREPRINT_DOI,
                    location,
                    CitationRelation.MODEL_SOURCE,
                    SourceLocationStatus.VERIFIED,
                ),
            ),
            chemical_state="Hypothetical pure RP_S or RP_T boundary of the four-state model.",
            conditions=(
                "Equation 11 mathematical boundary used only to calculate singlet-triplet "
                "contrast; not measured and not associated with a magnetic-field condition."
            ),
            uncertainty=Uncertainty(
                UncertaintyStatus.NOT_REPORTED,
                "1",
                note="A hypothetical boundary value has no reported experimental uncertainty.",
            ),
        ),
    )


def _population_from_numerical(values: NDArray[np.float64], total: float) -> PopulationState:
    if not np.all(np.isfinite(values)):
        raise ArithmeticError("numerical evolution produced a non-finite population")
    if np.min(values) < -_NUMERICAL_NEGATIVE_ATOL:
        raise ArithmeticError("numerical evolution produced a negative population")
    cleaned = np.where(np.abs(values) <= _NORMALIZATION_ATOL, 0.0, values)
    conserved_total = float(np.sum(cleaned))
    if not isclose(conserved_total, total, rel_tol=1e-12, abs_tol=_NORMALIZATION_ATOL):
        raise ArithmeticError("numerical evolution did not conserve total population")
    return PopulationState(
        float(cleaned[0]),
        float(cleaned[1]),
        float(cleaned[2]),
        float(cleaned[3]),
        total,
    )
