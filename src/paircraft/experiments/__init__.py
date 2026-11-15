"""Experimental trace ingestion and auditable magnetic-field-effect analysis."""

from paircraft.experiments.analysis import MfeAnalysisConfig, MfeAnalysisResult, analyze_folded_mfe
from paircraft.experiments.model import ExperimentalTrace, SourceProvenance, TraceClass
from paircraft.experiments.oxford import (
    OxfordMaryProvenance,
    load_oxford_mary_scalar_trace,
    load_oxford_photodiode_csv,
)
from paircraft.experiments.oxford_corpus import (
    OXFORD_V3_ACQUISITIONS,
    AcquisitionSpec,
    AcquisitionSummary,
    ExposureState,
    build_oxford_calibration_corpus,
)
from paircraft.experiments.oxford_single_cell import (
    CellCycleObservation,
    CellSummary,
    EmpiricalCellDistribution,
    OxfordFig1cProvenance,
    OxfordProcessedCellDataset,
    OxfordSingleCellAnalysisConfig,
    OxfordSingleCellAnalysisResult,
    PooledReconciliation,
    PublicationDisplayResult,
    analyze_oxford_fig1c,
    load_oxford_fig1c_processed_cells,
)

__all__ = [
    "OXFORD_V3_ACQUISITIONS",
    "AcquisitionSpec",
    "AcquisitionSummary",
    "CellCycleObservation",
    "CellSummary",
    "EmpiricalCellDistribution",
    "ExperimentalTrace",
    "ExposureState",
    "MfeAnalysisConfig",
    "MfeAnalysisResult",
    "OxfordFig1cProvenance",
    "OxfordMaryProvenance",
    "OxfordProcessedCellDataset",
    "OxfordSingleCellAnalysisConfig",
    "OxfordSingleCellAnalysisResult",
    "PooledReconciliation",
    "PublicationDisplayResult",
    "SourceProvenance",
    "TraceClass",
    "analyze_folded_mfe",
    "analyze_oxford_fig1c",
    "build_oxford_calibration_corpus",
    "load_oxford_fig1c_processed_cells",
    "load_oxford_mary_scalar_trace",
    "load_oxford_photodiode_csv",
]
