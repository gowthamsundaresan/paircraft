"""Phenomenological photocycle/readout kinetics.

This package implements the four-state model in Ross, Lodesani, and Aiello
(bioRxiv 2026.07.08.737352, equations 1-6 and supplementary equations 16-21).
It maps an *imposed* singlet yield to fluorescence emission; it does not calculate
singlet yield from a magnetic field or spin Hamiltonian.
"""

from paircraft.readout.model import (
    Citation,
    CitationRelation,
    EvidenceStatus,
    FourStateModel,
    FourStateRates,
    ParameterMetadata,
    PopulationState,
    RateConstant,
    ReferenceInputs,
    SingletYield,
    SourceLocationStatus,
    Uncertainty,
    UncertaintyStatus,
    ground_state,
    ross_2026_reference_inputs,
)

__all__ = [
    "Citation",
    "CitationRelation",
    "EvidenceStatus",
    "FourStateModel",
    "FourStateRates",
    "ParameterMetadata",
    "PopulationState",
    "RateConstant",
    "ReferenceInputs",
    "SingletYield",
    "SourceLocationStatus",
    "Uncertainty",
    "UncertaintyStatus",
    "ground_state",
    "ross_2026_reference_inputs",
]
