"""Experimental trace ingestion and auditable magnetic-field-effect analysis."""

from paircraft.experiments.analysis import MfeAnalysisConfig, MfeAnalysisResult, analyze_folded_mfe
from paircraft.experiments.model import ExperimentalTrace, SourceProvenance, TraceClass
from paircraft.experiments.oxford import (
    OxfordMaryProvenance,
    load_oxford_mary_scalar_trace,
    load_oxford_photodiode_csv,
)

__all__ = [
    "ExperimentalTrace",
    "MfeAnalysisConfig",
    "MfeAnalysisResult",
    "OxfordMaryProvenance",
    "SourceProvenance",
    "TraceClass",
    "analyze_folded_mfe",
    "load_oxford_mary_scalar_trace",
    "load_oxford_photodiode_csv",
]
