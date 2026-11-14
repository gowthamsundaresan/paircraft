"""Oxford Figure 1c processed single-cell ingestion and descriptive analysis."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import isclose, isfinite
from pathlib import Path
from typing import Final

import numpy as np
from scipy.interpolate import make_interp_spline

from paircraft import __version__
from paircraft.experiments.model import SourceProvenance, TraceClass

_EXPECTED_TRACE_SHAPE: Final = (153, 900)
_EXPECTED_MASK_SHAPE: Final = (3200, 3200)
_EXPECTED_KEYS: Final = frozenset({"time_traces", "mask", "background"})
_ALGORITHM_IDENTIFIER: Final = "paircraft.experiments.analyze_oxford_fig1c"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_payload(path: Path, provenance: SourceProvenance) -> None:
    if _sha256(path) != provenance.sha256.lower():
        raise ValueError(f"payload sha256 does not match provenance: {path.name}")


def _stable_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_integral_metadata(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not isfinite(float(value))
        or float(value) != int(value)
    ):
        raise ValueError(f"{name} must be a finite integral number")
    return int(value)


@dataclass(frozen=True, slots=True)
class OxfordFig1cProvenance:
    """Exact processed data, metadata, and publication notebook sources."""

    processed_traces: SourceProvenance
    metadata: SourceProvenance
    publication_notebook: SourceProvenance


@dataclass(frozen=True, slots=True)
class OxfordProcessedCellDataset:
    """Deposited ROI-derived traces whose raw segmentation is unavailable."""

    acquisition_id: str
    sample_label: str
    construct_id: str
    construct_mapping_status: str
    trace_class: TraceClass
    traces: tuple[tuple[float, ...], ...]
    background_camera_count: float
    union_mask_shape: tuple[int, int]
    union_mask_true_pixels: int
    union_mask_sha256: str
    mask_semantics: str
    frames_per_state: int
    period_count: int
    exposure_seconds: float
    magnet_current_ampere: float
    field_protocol_citation: str
    acquisition_date: str
    provenance: OxfordFig1cProvenance

    def __post_init__(self) -> None:
        if self.trace_class is not TraceClass.ROI_DERIVED_TRACE:
            raise ValueError("Oxford processed cells must remain classified as ROI-derived")
        if len(self.traces) != _EXPECTED_TRACE_SHAPE[0] or any(
            len(trace) != _EXPECTED_TRACE_SHAPE[1] for trace in self.traces
        ):
            raise ValueError("Oxford Figure 1c traces must retain the deposited 153x900 shape")
        if self.union_mask_shape != _EXPECTED_MASK_SHAPE:
            raise ValueError("Oxford Figure 1c union mask must retain its deposited shape")
        if self.union_mask_true_pixels <= 0:
            raise ValueError("union mask must contain at least one selected pixel")
        if not isfinite(self.background_camera_count):
            raise ValueError("deposited background must be finite")
        if self.frames_per_state != 45 or self.period_count != 10:
            raise ValueError("Oxford Figure 1c schedule must contain ten 45+45-frame periods")


@dataclass(frozen=True, slots=True)
class OxfordSingleCellAnalysisConfig:
    """Frozen processed-trace analysis declared before viewing cell effects."""

    frames_per_state: int = 45
    discard_periods: int = 1
    late_dwell_frames: int = 10
    spline_anchor_index_in_period: int = 43
    spline_degree: int = 3
    publication_cell_index: int = 40
    publication_period_count: int = 4
    pooled_identity_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.frames_per_state != 45:
            raise ValueError("Figure 1c analysis requires 45 frames per field state")
        if self.discard_periods != 1:
            raise ValueError("Figure 1c analysis requires the declared first-period discard")
        if not 1 <= self.late_dwell_frames <= self.frames_per_state:
            raise ValueError("late dwell must fit within one field state")
        if self.spline_anchor_index_in_period != self.frames_per_state - 2:
            raise ValueError("Figure 1c spline anchor must reproduce deposited notebook index 43")
        if self.spline_degree != 3:
            raise ValueError("Figure 1c publication reconstruction requires a cubic spline")
        if self.publication_cell_index != 40 or self.publication_period_count != 4:
            raise ValueError("publication display requires deposited row 40 and four periods")
        if self.pooled_identity_tolerance <= 0 or not isfinite(self.pooled_identity_tolerance):
            raise ValueError("pooled identity tolerance must be positive and finite")

    @property
    def config_sha256(self) -> str:
        """Return a stable identity for the complete analysis configuration."""

        return _stable_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class PublicationDisplayResult:
    """Exact deposited display plus an explicitly Paircraft-defined audit summary."""

    selected_deposited_row: int
    source_frame_start: int
    source_frame_stop: int
    constructed_time_seconds: tuple[float, ...]
    relative_signal_percent: tuple[float, ...]
    late_off_percent_by_period: tuple[float, ...]
    late_on_percent_by_period: tuple[float, ...]
    mean_late_off_percent: float
    mean_late_on_percent: float
    selection_status: str = "illustrative_selection_reason_not_deposited"
    scalar_summary_status: str = "paircraft_audit_summary_not_defined_by_publication"


@dataclass(frozen=True, slots=True)
class CellCycleObservation:
    """One technical-cycle observation retained even when it is ineligible."""

    observation_id: str
    cell_id: str
    deposited_row: int
    cycle_index: int
    source_frame_start: int
    source_frame_stop: int
    eligible: bool
    qc_codes: tuple[str, ...]
    off_baseline_denominator: float | None
    relative_effect: float | None


@dataclass(frozen=True, slots=True)
class CellSummary:
    """Descriptive result for one deposited trace row, not a biological replicate."""

    cell_id: str
    deposited_row: int
    identity_basis: str
    eligible: bool
    qc_codes: tuple[str, ...]
    eligible_cycle_count: int
    median_relative_effect: float | None
    within_cell_cycle_iqr: float | None


@dataclass(frozen=True, slots=True)
class EmpiricalCellDistribution:
    """Descriptive spread conditional on this one deposited acquisition."""

    deposited_cell_count: int
    eligible_cell_count: int
    ineligible_cell_count: int
    median_relative_effect: float
    mean_relative_effect: float
    standard_deviation: float
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    count_at_or_below_minus_10_percent: int
    count_at_or_below_minus_25_percent: int
    count_at_or_below_minus_40_percent: int
    estimand_definition: str = (
        "E2: median cycle ratio within each cell, then distribution over cells"
    )
    uncertainty_status: str = "descriptive_only_no_independent_biological_replicates"


@dataclass(frozen=True, slots=True)
class PooledReconciliation:
    """Algebraic identity over the same eligible cell-cycle observations."""

    pooled_relative_effect: float
    denominator_weighted_relative_effect: float
    absolute_identity_residual: float
    tolerance: float
    identity_passed: bool
    estimand_definition: str = (
        "E3: denominator-weighted eligible cell-cycle ratios; not E2 per-cell medians"
    )
    residual_fluorescence_status: str = "not_available_from_deposited_roi_traces"


@dataclass(frozen=True, slots=True)
class OxfordSingleCellAnalysisResult:
    """All three processed-data estimands with explicit claim boundaries."""

    dataset: OxfordProcessedCellDataset
    config: OxfordSingleCellAnalysisConfig
    publication_display: PublicationDisplayResult
    cycle_observations: tuple[CellCycleObservation, ...]
    cell_summaries: tuple[CellSummary, ...]
    empirical_distribution: EmpiricalCellDistribution
    pooled_reconciliation: PooledReconciliation
    algorithm_identifier: str = _ALGORITHM_IDENTIFIER
    algorithm_version: str = __version__
    raw_segmentation_status: str = "not_reproducible_from_final_public_deposit"
    biological_generalization_status: str = "not_estimable_from_one_deposited_acquisition"
    terminal_baseline_status: str = (
        "final ON window uses cubic extrapolation beyond the final OFF anchor; retained from the "
        "predeclared deposited method and summarized by a cycle median"
    )


def load_oxford_fig1c_processed_cells(
    processed_trace_path: Path,
    metadata_path: Path,
    publication_notebook_path: Path,
    *,
    provenance: OxfordFig1cProvenance,
    construct_id: str,
) -> OxfordProcessedCellDataset:
    """Load the exact final Oxford Figure 1c processed artifact without executing code."""

    _verify_payload(processed_trace_path, provenance.processed_traces)
    _verify_payload(metadata_path, provenance.metadata)
    _verify_payload(publication_notebook_path, provenance.publication_notebook)

    try:
        with np.load(processed_trace_path, allow_pickle=False) as payload:
            if frozenset(payload.files) != _EXPECTED_KEYS:
                raise ValueError(
                    "processed archive must contain exactly time_traces, mask, background"
                )
            traces = np.asarray(payload["time_traces"])
            mask = np.asarray(payload["mask"])
            background = np.asarray(payload["background"])
    except ValueError as error:
        raise ValueError(f"unsafe or invalid Oxford processed-cell archive: {error}") from error

    if traces.shape != _EXPECTED_TRACE_SHAPE or traces.dtype.hasobject:
        raise ValueError("time_traces must be a numeric non-object array with shape 153x900")
    if (
        not np.issubdtype(traces.dtype, np.number)
        or np.issubdtype(traces.dtype, np.complexfloating)
        or not np.isfinite(traces).all()
    ):
        raise ValueError("time_traces must contain only finite real numeric values")
    if mask.shape != _EXPECTED_MASK_SHAPE or mask.dtype != np.bool_:
        raise ValueError("mask must be the deposited 3200x3200 boolean union mask")
    if (
        background.shape != ()
        or background.dtype.hasobject
        or not np.issubdtype(background.dtype, np.number)
        or np.issubdtype(background.dtype, np.complexfloating)
    ):
        raise ValueError("background must be one real numeric scalar")
    background_value = float(background)
    if not isfinite(background_value):
        raise ValueError("background must be finite")

    try:
        metadata_document = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata = metadata_document["Fig1_PanelC"]
        if not isinstance(metadata, dict):
            raise TypeError
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("metadata must contain the exact Fig1_PanelC acquisition block") from error
    try:
        sample_label = str(metadata["Sample"])
        exposure_seconds = float(metadata["ImageExposure_ms"]) / 1000.0
        magnet_current = float(metadata["MagnetCurrentA"])
        acquisition_date = str(metadata["date"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("metadata must contain the exact Fig1_PanelC acquisition block") from error
    frames_per_state = _require_integral_metadata(metadata.get("ImagesPerStep"), "ImagesPerStep")
    period_count = _require_integral_metadata(metadata.get("NumSteps"), "NumSteps")

    if (
        sample_label != "ML 2"
        or frames_per_state != 45
        or period_count != 10
        or not isclose(exposure_seconds, 0.2)
        or not isclose(magnet_current, 0.22)
        or acquisition_date != "23-07-2025"
    ):
        raise ValueError("Fig1_PanelC metadata contradicts the authenticated acquisition contract")

    mask_bytes = np.ascontiguousarray(mask).view(np.uint8).tobytes()
    acquisition_id = _stable_hash(
        {
            "source_id": provenance.processed_traces.source_id,
            "source_version": provenance.processed_traces.source_version,
            "exact_location": provenance.processed_traces.exact_location,
            "sha256": provenance.processed_traces.sha256.lower(),
        }
    )
    return OxfordProcessedCellDataset(
        acquisition_id=acquisition_id,
        sample_label=sample_label,
        construct_id=construct_id,
        construct_mapping_status="publication_level_not_independent_plasmid_verification",
        trace_class=TraceClass.ROI_DERIVED_TRACE,
        traces=tuple(tuple(float(value) for value in row) for row in traces),
        background_camera_count=background_value,
        union_mask_shape=(int(mask.shape[0]), int(mask.shape[1])),
        union_mask_true_pixels=int(np.count_nonzero(mask)),
        union_mask_sha256=hashlib.sha256(mask_bytes).hexdigest(),
        mask_semantics="union mask only; no deposited row-to-individual-mask mapping",
        frames_per_state=frames_per_state,
        period_count=period_count,
        exposure_seconds=exposure_seconds,
        magnet_current_ampere=magnet_current,
        field_protocol_citation=(
            "Nature 649, 1172-1179 (2026), DOI 10.1038/s41586-025-09971-3; "
            "10 mT is inherited from the paper, not inferred from current"
        ),
        acquisition_date=acquisition_date,
        provenance=provenance,
    )


def _detrend_periods(
    signal: np.ndarray,
    *,
    config: OxfordSingleCellAnalysisConfig,
    period_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    period = config.frames_per_state * 2
    expected = period_count * period
    if signal.shape != (expected,):
        raise ValueError("trace length does not match the declared complete periods")
    indices = np.arange(expected, dtype=np.float64)
    anchors = np.arange(config.spline_anchor_index_in_period, expected, period)
    if anchors.size < config.spline_degree + 1:
        raise ValueError("not enough OFF-state anchors for the cubic baseline")
    baseline = make_interp_spline(anchors, signal[anchors], k=config.spline_degree)(indices)
    if not np.isfinite(baseline).all() or np.any(baseline <= 0.0):
        raise ValueError("OFF-only spline baseline must remain finite and positive")
    return signal / baseline - 1.0, baseline


def _publication_display(
    dataset: OxfordProcessedCellDataset, config: OxfordSingleCellAnalysisConfig
) -> PublicationDisplayResult:
    period = config.frames_per_state * 2
    start = config.discard_periods * period
    stop = start + config.publication_period_count * period
    signal = np.asarray(dataset.traces[config.publication_cell_index][start:stop], dtype=np.float64)
    relative, _ = _detrend_periods(
        signal, config=config, period_count=config.publication_period_count
    )
    folded = relative.reshape(config.publication_period_count, period)
    off = folded[
        :, config.frames_per_state - config.late_dwell_frames : config.frames_per_state
    ].mean(axis=1)
    on = folded[:, period - config.late_dwell_frames : period].mean(axis=1)
    constructed_time = np.linspace(0.0, 20.0 * config.publication_period_count, signal.size)
    return PublicationDisplayResult(
        selected_deposited_row=config.publication_cell_index,
        source_frame_start=start,
        source_frame_stop=stop,
        constructed_time_seconds=tuple(float(value) for value in constructed_time),
        relative_signal_percent=tuple(float(100.0 * value) for value in relative),
        late_off_percent_by_period=tuple(float(100.0 * value) for value in off),
        late_on_percent_by_period=tuple(float(100.0 * value) for value in on),
        mean_late_off_percent=float(100.0 * off.mean()),
        mean_late_on_percent=float(100.0 * on.mean()),
    )


def _distribution(values: np.ndarray, deposited_count: int) -> EmpiricalCellDistribution:
    if values.size == 0:
        raise ValueError("processed acquisition has no structurally eligible cell traces")
    percentiles = np.percentile(values, (5, 25, 75, 95))
    return EmpiricalCellDistribution(
        deposited_cell_count=deposited_count,
        eligible_cell_count=int(values.size),
        ineligible_cell_count=deposited_count - int(values.size),
        median_relative_effect=float(np.median(values)),
        mean_relative_effect=float(values.mean()),
        standard_deviation=float(values.std(ddof=1)) if values.size > 1 else 0.0,
        percentile_5=float(percentiles[0]),
        percentile_25=float(percentiles[1]),
        percentile_75=float(percentiles[2]),
        percentile_95=float(percentiles[3]),
        count_at_or_below_minus_10_percent=int(np.count_nonzero(values <= -0.10)),
        count_at_or_below_minus_25_percent=int(np.count_nonzero(values <= -0.25)),
        count_at_or_below_minus_40_percent=int(np.count_nonzero(values <= -0.40)),
    )


def analyze_oxford_fig1c(
    dataset: OxfordProcessedCellDataset,
    config: OxfordSingleCellAnalysisConfig | None = None,
) -> OxfordSingleCellAnalysisResult:
    """Reconstruct the selected display and describe all deposited processed traces."""

    executed_config = config or OxfordSingleCellAnalysisConfig()
    if dataset.trace_class is not TraceClass.ROI_DERIVED_TRACE:
        raise ValueError("Figure 1c analysis requires deposited ROI-derived traces")
    if dataset.frames_per_state != executed_config.frames_per_state:
        raise ValueError("analysis configuration contradicts the deposited acquisition schedule")

    publication = _publication_display(dataset, executed_config)
    period = executed_config.frames_per_state * 2
    start = executed_config.discard_periods * period
    retained_periods = dataset.period_count - executed_config.discard_periods
    config_hash = executed_config.config_sha256
    observations: list[CellCycleObservation] = []
    summaries: list[CellSummary] = []
    pooled_numerator = 0.0
    pooled_denominator = 0.0
    weighted_numerator = 0.0

    for row_index, deposited_trace in enumerate(dataset.traces):
        cell_id = _stable_hash(
            {
                "acquisition_id": dataset.acquisition_id,
                "identity_basis": "deposited_trace_row_no_individual_mask_mapping",
                "row": row_index,
            }
        )
        signal = np.asarray(deposited_trace[start:], dtype=np.float64)
        cell_qc: tuple[str, ...] = ()
        try:
            _, baseline = _detrend_periods(
                signal, config=executed_config, period_count=retained_periods
            )
        except ValueError:
            baseline = np.empty(0, dtype=np.float64)
            cell_qc = ("invalid_off_only_spline_baseline",)

        cycle_effects: list[float] = []
        for cycle_index in range(retained_periods):
            cycle_start = cycle_index * period
            on_start = cycle_start + period - executed_config.late_dwell_frames
            cycle_stop = cycle_start + period
            observation_id = _stable_hash(
                {
                    "cell_id": cell_id,
                    "cycle_index": cycle_index + executed_config.discard_periods,
                    "config_sha256": config_hash,
                }
            )
            if cell_qc:
                observations.append(
                    CellCycleObservation(
                        observation_id=observation_id,
                        cell_id=cell_id,
                        deposited_row=row_index,
                        cycle_index=cycle_index + executed_config.discard_periods,
                        source_frame_start=start + cycle_start,
                        source_frame_stop=start + cycle_stop,
                        eligible=False,
                        qc_codes=cell_qc,
                        off_baseline_denominator=None,
                        relative_effect=None,
                    )
                )
                continue
            denominator = float(baseline[on_start:cycle_stop].sum())
            numerator = float((signal[on_start:cycle_stop] - baseline[on_start:cycle_stop]).sum())
            if denominator <= 0.0 or not isfinite(numerator):
                raise ValueError("validated baseline produced an invalid late-ON denominator")
            effect = numerator / denominator
            cycle_effects.append(effect)
            pooled_numerator += numerator
            pooled_denominator += denominator
            weighted_numerator += denominator * effect
            observations.append(
                CellCycleObservation(
                    observation_id=observation_id,
                    cell_id=cell_id,
                    deposited_row=row_index,
                    cycle_index=cycle_index + executed_config.discard_periods,
                    source_frame_start=start + cycle_start,
                    source_frame_stop=start + cycle_stop,
                    eligible=True,
                    qc_codes=(),
                    off_baseline_denominator=denominator,
                    relative_effect=effect,
                )
            )

        effects = np.asarray(cycle_effects, dtype=np.float64)
        summaries.append(
            CellSummary(
                cell_id=cell_id,
                deposited_row=row_index,
                identity_basis="deposited trace row; individual mask mapping not deposited",
                eligible=not cell_qc,
                qc_codes=cell_qc,
                eligible_cycle_count=int(effects.size),
                median_relative_effect=float(np.median(effects)) if effects.size else None,
                within_cell_cycle_iqr=(
                    float(np.percentile(effects, 75) - np.percentile(effects, 25))
                    if effects.size
                    else None
                ),
            )
        )

    cell_values = np.asarray(
        [
            summary.median_relative_effect
            for summary in summaries
            if summary.median_relative_effect is not None
        ],
        dtype=np.float64,
    )
    distribution = _distribution(cell_values, len(summaries))
    if pooled_denominator <= 0.0:
        raise ValueError("no eligible positive denominator remains for pooled reconciliation")
    pooled = pooled_numerator / pooled_denominator
    weighted = weighted_numerator / pooled_denominator
    residual = abs(pooled - weighted)
    reconciliation = PooledReconciliation(
        pooled_relative_effect=pooled,
        denominator_weighted_relative_effect=weighted,
        absolute_identity_residual=residual,
        tolerance=executed_config.pooled_identity_tolerance,
        identity_passed=residual <= executed_config.pooled_identity_tolerance,
    )
    if not reconciliation.identity_passed:
        raise ValueError("same-acquisition pooled identity failed")

    return OxfordSingleCellAnalysisResult(
        dataset=dataset,
        config=executed_config,
        publication_display=publication,
        cycle_observations=tuple(observations),
        cell_summaries=tuple(summaries),
        empirical_distribution=distribution,
        pooled_reconciliation=reconciliation,
    )
