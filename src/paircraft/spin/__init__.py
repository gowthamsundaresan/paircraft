"""Auditable spin-dynamics models and internal numerical kernels."""

from paircraft.spin.maeda2012 import (
    CitationRelation,
    Maeda2012Specification,
    MaedaProteinSystem,
    ParameterEvidence,
    PublishedHyperfine,
    PublishedParameter,
    SecondaryPairYield,
    SourceCitation,
    maeda_2012_specification,
    solve_maeda_secondary_pair_yield,
)
from paircraft.spin.maeda_sensitivity import (
    HypotheticalParameterOverride,
    HypotheticalSecondaryPairYield,
    MaedaSensitivitySpecification,
    SensitivityDimension,
    SensitivityDirection,
    maeda_sensitivity_specification,
    solve_maeda_sensitivity_yield,
)

__all__ = [
    "CitationRelation",
    "HypotheticalParameterOverride",
    "HypotheticalSecondaryPairYield",
    "Maeda2012Specification",
    "MaedaProteinSystem",
    "MaedaSensitivitySpecification",
    "ParameterEvidence",
    "PublishedHyperfine",
    "PublishedParameter",
    "SecondaryPairYield",
    "SensitivityDimension",
    "SensitivityDirection",
    "SourceCitation",
    "maeda_2012_specification",
    "maeda_sensitivity_specification",
    "solve_maeda_secondary_pair_yield",
    "solve_maeda_sensitivity_yield",
]
