"""Reproducible file handoff for the exact Oxford Figure 1c processed dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Final

import numpy as np
import scipy

from paircraft.experiments.model import SourceProvenance
from paircraft.experiments.oxford_single_cell import (
    OxfordFig1cProvenance,
    OxfordSingleCellAnalysisResult,
    analyze_oxford_fig1c,
    load_oxford_fig1c_processed_cells,
)

_DEPOSIT_URL: Final = "https://doi.org/10.25446/oxford.30344995.v3"
_DEPOSIT_VERSION: Final = "Oxford data deposit 30344995 version 3"
_LICENSE: Final = "CC BY 4.0"
_PROCESSED_SHA256: Final = "49a58847a9ac1cecdb819d8340f82f7991ad48ffe78e01353d9de0ae5c1eb41b"
_METADATA_SHA256: Final = "a8f579e2335b3cb5400a326f657527082ad73cbc8d2e1f264c628ef1ceabe48e"
_NOTEBOOK_SHA256: Final = "945861335b637891374ec1f1b181224d743388765f92d45775310b8fce6749d4"


def _source(source_id: str, exact_location: str, sha256: str) -> SourceProvenance:
    return SourceProvenance(
        source_id=source_id,
        source_url=_DEPOSIT_URL,
        source_version=_DEPOSIT_VERSION,
        exact_location=exact_location,
        sha256=sha256,
        license=_LICENSE,
    )


def oxford_fig1c_provenance() -> OxfordFig1cProvenance:
    """Return the authenticated public-source contract for final Nature Figure 1c."""

    return OxfordFig1cProvenance(
        processed_traces=_source(
            "figshare-file-59258873",
            "upload/Fig1_PanelC/MFE_single_cell_data_0.npz",
            _PROCESSED_SHA256,
        ),
        metadata=_source(
            "figshare-file-59258273",
            "upload/metadata.json",
            _METADATA_SHA256,
        ),
        publication_notebook=_source(
            "figshare-file-59258753",
            "upload/Fig1_PanelC/Fig1_PanelC.ipynb",
            _NOTEBOOK_SHA256,
        ),
    )


def _implementation_sha256() -> str:
    implementation_path = Path(__file__).with_name("oxford_single_cell.py")
    return hashlib.sha256(implementation_path.read_bytes()).hexdigest()


def _reporting_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _report_document(result: OxfordSingleCellAnalysisResult) -> dict[str, object]:
    dataset = result.dataset
    distribution = result.empirical_distribution
    reconciliation = result.pooled_reconciliation
    return {
        "schema_version": "paircraft.oxford_fig1c_report.v1",
        "algorithm": {
            "identifier": result.algorithm_identifier,
            "paircraft_version": result.algorithm_version,
            "implementation_sha256": _implementation_sha256(),
            "reporting_sha256": _reporting_sha256(),
            "config": asdict(result.config),
            "config_sha256": result.config.config_sha256,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "acquisition": {
            "acquisition_id": dataset.acquisition_id,
            "sample_label": dataset.sample_label,
            "construct_id": dataset.construct_id,
            "construct_mapping_status": dataset.construct_mapping_status,
            "processing_class": dataset.trace_class.value,
            "deposited_trace_shape": [len(dataset.traces), len(dataset.traces[0])],
            "background_camera_count": dataset.background_camera_count,
            "union_mask_shape": list(dataset.union_mask_shape),
            "union_mask_true_pixels": dataset.union_mask_true_pixels,
            "union_mask_sha256": dataset.union_mask_sha256,
            "mask_semantics": dataset.mask_semantics,
            "frames_per_state": dataset.frames_per_state,
            "period_count": dataset.period_count,
            "exposure_seconds": dataset.exposure_seconds,
            "magnet_current_ampere": dataset.magnet_current_ampere,
            "field_protocol_citation": dataset.field_protocol_citation,
            "acquisition_date": dataset.acquisition_date,
            "sources": {
                "processed_traces": asdict(dataset.provenance.processed_traces),
                "metadata": asdict(dataset.provenance.metadata),
                "publication_notebook": asdict(dataset.provenance.publication_notebook),
            },
        },
        "publication_display": asdict(result.publication_display),
        "estimand_definitions": {
            "E2": (
                "median eligible technical-cycle ratio within each deposited row, followed by "
                "a descriptive distribution over eligible rows"
            ),
            "E3": (
                "pooled ratio over eligible cell-cycle numerators and denominators, identical to "
                "the denominator-weighted eligible cell-cycle ratios; E3 does not weight E2 "
                "per-cell medians"
            ),
        },
        "empirical_deposited_cell_distribution": asdict(distribution),
        "pooled_reconciliation": asdict(reconciliation),
        "inclusion_funnel": {
            "deposited_trace_rows": distribution.deposited_cell_count,
            "structurally_eligible_rows": distribution.eligible_cell_count,
            "ineligible_rows": distribution.ineligible_cell_count,
            "eligible_cycle_observations": sum(
                observation.eligible for observation in result.cycle_observations
            ),
            "ineligible_cycle_observations": sum(
                not observation.eligible for observation in result.cycle_observations
            ),
            "ineligible_deposited_rows": [
                summary.deposited_row for summary in result.cell_summaries if not summary.eligible
            ],
            "reason_counts": {
                "invalid_off_only_spline_baseline": sum(
                    "invalid_off_only_spline_baseline" in summary.qc_codes
                    for summary in result.cell_summaries
                )
            },
        },
        "cell_summaries": [asdict(summary) for summary in result.cell_summaries],
        "cycle_observations": [asdict(observation) for observation in result.cycle_observations],
        "claim_boundaries": {
            "raw_segmentation": result.raw_segmentation_status,
            "biological_generalization": result.biological_generalization_status,
            "terminal_baseline": result.terminal_baseline_status,
            "field_strength": "10 mT inherited from cited paper; not inferred from 0.22 A",
            "separate_scalar_archive": (
                "the earlier -7.66 percent scalar result is contextual only; acquisition "
                "equivalence is not established"
            ),
        },
    }


def _markdown_report(result: OxfordSingleCellAnalysisResult) -> str:
    display = result.publication_display
    distribution = result.empirical_distribution
    pooled = result.pooled_reconciliation
    excluded = [
        str(summary.deposited_row) for summary in result.cell_summaries if not summary.eligible
    ]
    return f"""# Oxford final Figure 1c processed-trace reproduction

## Result

- Selected publication row: {display.selected_deposited_row}
- Paircraft audit mean late-ON display value: {display.mean_late_on_percent:.6f}%
- Paircraft audit mean late-OFF display value: {display.mean_late_off_percent:.6f}%
- Deposited trace rows: {distribution.deposited_cell_count}
- Structurally eligible rows: {distribution.eligible_cell_count}
- Ineligible rows: {distribution.ineligible_cell_count} ({", ".join(excluded)})
- Eligible-row median cycle-median MFE: {100 * distribution.median_relative_effect:.6f}%
- Eligible-row mean cycle-median MFE: {100 * distribution.mean_relative_effect:.6f}%
- Same-acquisition pooled MFE: {100 * pooled.pooled_relative_effect:.6f}%
- Pooled identity residual: {pooled.absolute_identity_residual:.3e}

The approximately -50% value reconstructs one illustratively selected deposited row. The empirical
distribution describes eligible processed trace rows in this one deposited acquisition; it is not
a biological-population estimate. {distribution.ineligible_cell_count} rows fail the response-blind
requirement that the OFF-only spline baseline remain finite and positive across all retained frames.
E2 first takes the median of nine technical-cycle ratios for each eligible cell row. E3 instead
pools every eligible cell-cycle numerator and denominator and equals their denominator-weighted
cell-cycle ratios; E3 is not calculated by weighting the E2 per-cell medians.

## Claim boundary

Raw images and individual masks for the final acquisition are not publicly deposited, so this is
not a segmentation reproduction or a biological-population estimate. The final ON window is cubic
extrapolation beyond the last OFF anchor; the predeclared robust cell summary is therefore the
median across nine technical cycles. The cited paper supplies 10 mT; Paircraft does not convert the
deposited 0.22 A control value to
magnetic flux density. The separate -7.66% scalar result is from another acquisition and analysis
and is not a reconciliation target.
"""


def _write_observations_csv(path: Path, result: OxfordSingleCellAnalysisResult) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "observation_id",
                "cell_id",
                "deposited_row",
                "cycle_index",
                "source_frame_start",
                "source_frame_stop",
                "eligible",
                "qc_codes",
                "off_baseline_denominator",
                "relative_effect",
            )
        )
        for observation in result.cycle_observations:
            writer.writerow(
                (
                    observation.observation_id,
                    observation.cell_id,
                    observation.deposited_row,
                    observation.cycle_index,
                    observation.source_frame_start,
                    observation.source_frame_stop,
                    observation.eligible,
                    ";".join(observation.qc_codes),
                    observation.off_baseline_denominator,
                    observation.relative_effect,
                )
            )


def write_oxford_fig1c_report(
    result: OxfordSingleCellAnalysisResult, output_directory: Path
) -> tuple[Path, Path, Path]:
    """Atomically reveal one complete report directory and refuse replacement."""

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    if output_directory.exists():
        raise FileExistsError(f"report output directory already exists: {output_directory}")
    final_paths = (
        output_directory / "oxford-fig1c-report.json",
        output_directory / "oxford-fig1c-observations.csv",
        output_directory / "oxford-fig1c-report.md",
    )
    temporary_directory = Path(
        tempfile.mkdtemp(prefix=f".{output_directory.name}-", dir=output_directory.parent)
    )
    temporary_paths = tuple(temporary_directory / path.name for path in final_paths)
    try:
        temporary_paths[0].write_text(
            json.dumps(_report_document(result), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_observations_csv(temporary_paths[1], result)
        temporary_paths[2].write_text(_markdown_report(result), encoding="utf-8")
        if output_directory.exists():
            raise FileExistsError(
                f"report output directory appeared during staging: {output_directory}"
            )
        os.rename(temporary_directory, output_directory)
    except BaseException:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise
    return final_paths


def reproduce_oxford_fig1c(
    processed_trace_path: Path,
    metadata_path: Path,
    publication_notebook_path: Path,
    output_directory: Path,
) -> tuple[Path, Path, Path]:
    """Run the authenticated final Figure 1c analysis and publish its outputs."""

    dataset = load_oxford_fig1c_processed_cells(
        processed_trace_path,
        metadata_path,
        publication_notebook_path,
        provenance=oxford_fig1c_provenance(),
        construct_id="construct-maglov2-domain-aa",
    )
    return write_oxford_fig1c_report(analyze_oxford_fig1c(dataset), output_directory)
