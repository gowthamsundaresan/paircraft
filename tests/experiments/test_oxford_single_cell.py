import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from paircraft.experiments import (
    OxfordFig1cProvenance,
    OxfordProcessedCellDataset,
    OxfordSingleCellAnalysisConfig,
    SourceProvenance,
    TraceClass,
    analyze_oxford_fig1c,
    load_oxford_fig1c_processed_cells,
    oxford_fig1c_report,
)
from paircraft.experiments.oxford_fig1c_report import write_oxford_fig1c_report


def _source(path: Path, source_id: str) -> SourceProvenance:
    return SourceProvenance(
        source_id=source_id,
        source_url=f"https://example.test/{path.name}",
        source_version="fixed fixture v1",
        exact_location=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        license="test fixture",
    )


def _write_source_fixture(tmp_path: Path) -> tuple[Path, Path, Path, OxfordFig1cProvenance]:
    processed = tmp_path / "MFE_single_cell_data_0.npz"
    metadata = tmp_path / "metadata.json"
    notebook = tmp_path / "Fig1_PanelC.ipynb"
    signal = np.tile(np.concatenate((np.full(45, 100.0), np.full(45, 80.0))), 10).astype(np.float32)
    np.savez_compressed(
        processed,
        time_traces=np.tile(signal, (153, 1)),
        mask=np.ones((3200, 3200), dtype=np.bool_),
        background=np.asarray(5.0, dtype=np.float32),
    )
    metadata.write_text(
        json.dumps(
            {
                "Fig1_PanelC": {
                    "Sample": "ML 2",
                    "ImagesPerStep": 45,
                    "NumSteps": 10,
                    "ImageExposure_ms": 200,
                    "MagnetCurrentA": 0.22,
                    "date": "23-07-2025",
                }
            }
        ),
        encoding="utf-8",
    )
    notebook.write_text('{"cells": []}\n', encoding="utf-8")
    provenance = OxfordFig1cProvenance(
        processed_traces=_source(processed, "processed"),
        metadata=_source(metadata, "metadata"),
        publication_notebook=_source(notebook, "notebook"),
    )
    return processed, metadata, notebook, provenance


def _dataset(*, invalid_row: int | None = None) -> OxfordProcessedCellDataset:
    one_period = (100.0,) * 45 + (80.0,) * 45
    rows = [one_period * 10 for _ in range(153)]
    if invalid_row is not None:
        rows[invalid_row] = (0.0,) * 900
    source = SourceProvenance(
        source_id="synthetic-cells",
        source_url="https://example.test/cells",
        source_version="v1",
        exact_location="generated in test",
        sha256="a" * 64,
        license="test fixture",
    )
    provenance = OxfordFig1cProvenance(source, source, source)
    return OxfordProcessedCellDataset(
        acquisition_id="synthetic-acquisition",
        sample_label="ML 2",
        construct_id="construct-maglov2-domain-aa",
        construct_mapping_status="test fixture",
        trace_class=TraceClass.ROI_DERIVED_TRACE,
        traces=tuple(rows),
        background_camera_count=5.0,
        union_mask_shape=(3200, 3200),
        union_mask_true_pixels=153,
        union_mask_sha256="b" * 64,
        mask_semantics="synthetic union mask with no row mapping",
        frames_per_state=45,
        period_count=10,
        exposure_seconds=0.2,
        magnet_current_ampere=0.22,
        field_protocol_citation="test fixture; no field conversion",
        acquisition_date="23-07-2025",
        provenance=provenance,
    )


def test_processed_reader_preserves_roi_derivation_and_union_mask_limit(tmp_path: Path) -> None:
    processed, metadata, notebook, provenance = _write_source_fixture(tmp_path)

    dataset = load_oxford_fig1c_processed_cells(
        processed,
        metadata,
        notebook,
        provenance=provenance,
        construct_id="construct-maglov2-domain-aa",
    )

    assert dataset.trace_class is TraceClass.ROI_DERIVED_TRACE
    assert len(dataset.traces) == 153
    assert len(dataset.traces[0]) == 900
    assert dataset.union_mask_true_pixels == 3200 * 3200
    assert "no deposited row-to-individual-mask mapping" in dataset.mask_semantics
    assert dataset.magnet_current_ampere == pytest.approx(0.22)
    assert "not inferred from current" in dataset.field_protocol_citation


def test_processed_reader_refuses_object_arrays_without_deserializing(tmp_path: Path) -> None:
    processed, metadata, notebook, provenance = _write_source_fixture(tmp_path)
    np.savez_compressed(
        processed,
        time_traces=np.full((153, 900), 1, dtype=object),
        mask=np.ones((3200, 3200), dtype=np.bool_),
        background=np.asarray(5.0),
    )
    provenance = replace(provenance, processed_traces=_source(processed, "unsafe"))

    with pytest.raises(ValueError, match="unsafe or invalid"):
        load_oxford_fig1c_processed_cells(
            processed,
            metadata,
            notebook,
            provenance=provenance,
            construct_id="construct-maglov2-domain-aa",
        )


def test_processed_reader_refuses_complex_traces(tmp_path: Path) -> None:
    processed, metadata, notebook, provenance = _write_source_fixture(tmp_path)
    np.savez_compressed(
        processed,
        time_traces=np.ones((153, 900), dtype=np.complex64),
        mask=np.ones((3200, 3200), dtype=np.bool_),
        background=np.asarray(5.0),
    )
    provenance = replace(provenance, processed_traces=_source(processed, "complex"))

    with pytest.raises(ValueError, match="finite real numeric"):
        load_oxford_fig1c_processed_cells(
            processed,
            metadata,
            notebook,
            provenance=provenance,
            construct_id="construct-maglov2-domain-aa",
        )


@pytest.mark.parametrize(("field", "value"), [("ImagesPerStep", 45.5), ("NumSteps", True)])
def test_processed_reader_requires_integral_schedule_metadata(
    tmp_path: Path, field: str, value: object
) -> None:
    processed, metadata, notebook, provenance = _write_source_fixture(tmp_path)
    document = json.loads(metadata.read_text(encoding="utf-8"))
    document["Fig1_PanelC"][field] = value
    metadata.write_text(json.dumps(document), encoding="utf-8")
    provenance = replace(provenance, metadata=_source(metadata, "non-integral-metadata"))

    with pytest.raises(ValueError, match=f"{field} must be a finite integral number"):
        load_oxford_fig1c_processed_cells(
            processed,
            metadata,
            notebook,
            provenance=provenance,
            construct_id="construct-maglov2-domain-aa",
        )


def test_single_cell_analysis_recovers_sign_and_pooled_identity() -> None:
    result = analyze_oxford_fig1c(_dataset())

    assert result.publication_display.mean_late_off_percent == pytest.approx(0.0, abs=1e-12)
    assert result.publication_display.mean_late_on_percent == pytest.approx(-20.0, abs=1e-12)
    assert result.empirical_distribution.eligible_cell_count == 153
    assert result.empirical_distribution.median_relative_effect == pytest.approx(-0.2)
    assert len(result.cycle_observations) == 153 * 9
    assert result.pooled_reconciliation.pooled_relative_effect == pytest.approx(-0.2)
    assert result.pooled_reconciliation.denominator_weighted_relative_effect == pytest.approx(-0.2)
    assert result.pooled_reconciliation.absolute_identity_residual <= 1e-12
    assert result.pooled_reconciliation.identity_passed
    assert "extrapolation" in result.terminal_baseline_status


def test_invalid_off_baseline_preserves_row_and_all_cycle_reasons() -> None:
    result = analyze_oxford_fig1c(_dataset(invalid_row=0))

    summary = result.cell_summaries[0]
    assert not summary.eligible
    assert summary.qc_codes == ("invalid_off_only_spline_baseline",)
    assert summary.median_relative_effect is None
    assert result.empirical_distribution.eligible_cell_count == 152
    row_observations = [
        observation for observation in result.cycle_observations if observation.deposited_row == 0
    ]
    assert len(row_observations) == 9
    assert all(not observation.eligible for observation in row_observations)
    assert all(observation.qc_codes == summary.qc_codes for observation in row_observations)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frames_per_state": 44}, "45 frames"),
        ({"discard_periods": 0}, "first-period discard"),
        ({"late_dwell_frames": 46}, "late dwell"),
        ({"spline_anchor_index_in_period": 42}, "notebook index 43"),
        ({"spline_degree": 2}, "cubic spline"),
        ({"publication_cell_index": 39}, "deposited row 40"),
        ({"pooled_identity_tolerance": 0.0}, "positive and finite"),
    ],
)
def test_analysis_config_refuses_estimand_drift(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        OxfordSingleCellAnalysisConfig(**kwargs)


def test_report_writer_publishes_auditable_outputs_atomically(tmp_path: Path) -> None:
    result = analyze_oxford_fig1c(_dataset(invalid_row=0))
    output_directory = tmp_path / "complete-report"

    report_path, observations_path, markdown_path = write_oxford_fig1c_report(
        result, output_directory
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "paircraft.oxford_fig1c_report.v1"
    assert report["inclusion_funnel"]["ineligible_deposited_rows"] == [0]
    assert report["claim_boundaries"]["raw_segmentation"].startswith("not_reproducible")
    assert "does not weight E2" in report["estimand_definitions"]["E3"]
    assert observations_path.read_text(encoding="utf-8").count("\n") == 153 * 9 + 1
    markdown = markdown_path.read_text(encoding="utf-8").replace("\n", " ")
    assert "not a biological-population estimate" in markdown
    assert not list(tmp_path.glob(".complete-report-*"))


def test_report_failure_cannot_expose_a_partial_output_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = analyze_oxford_fig1c(_dataset())
    output_directory = tmp_path / "must-remain-absent"

    def fail_csv(path: Path, analysis: object) -> None:
        del path, analysis
        raise RuntimeError("injected mid-staging failure")

    monkeypatch.setattr(oxford_fig1c_report, "_write_observations_csv", fail_csv)
    with pytest.raises(RuntimeError, match="injected mid-staging failure"):
        write_oxford_fig1c_report(result, output_directory)

    assert not output_directory.exists()
    assert not list(tmp_path.glob(".must-remain-absent-*"))


def test_report_writer_refuses_to_replace_an_existing_result(tmp_path: Path) -> None:
    result = analyze_oxford_fig1c(_dataset())
    output_directory = tmp_path / "existing-report"
    output_directory.mkdir()
    sentinel = output_directory / "sentinel.txt"
    sentinel.write_text("preserve me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        write_oxford_fig1c_report(result, output_directory)

    assert sentinel.read_text(encoding="utf-8") == "preserve me"


def test_authenticated_reproduction_runs_from_source_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed, metadata, notebook, _ = _write_source_fixture(tmp_path)
    monkeypatch.setattr(oxford_fig1c_report, "_PROCESSED_SHA256", _source(processed, "p").sha256)
    monkeypatch.setattr(oxford_fig1c_report, "_METADATA_SHA256", _source(metadata, "m").sha256)
    monkeypatch.setattr(oxford_fig1c_report, "_NOTEBOOK_SHA256", _source(notebook, "n").sha256)

    paths = oxford_fig1c_report.reproduce_oxford_fig1c(
        processed, metadata, notebook, tmp_path / "output"
    )

    assert all(path.is_file() for path in paths)
