"""Audited primitives for the final Oxford v3 Tier-D calibration corpus."""

from __future__ import annotations

import csv
import fcntl
import gzip
import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from pathlib import Path
from typing import BinaryIO, Final, cast

import numpy as np
from scipy.interpolate import make_interp_spline

TRANSFORM_ID: Final = "oxford-v3-nine-off-anchor-late-on-v1"
EXPECTED_MEMBERS: Final = frozenset({"time_traces.npy", "mask.npy", "background.npy"})
MINIMUM_FREE_BYTES: Final = 10 * 1024**3
MAX_TRANSIENT_BYTES: Final = 128 * 1024**2
MAX_SCRATCH_BYTES: Final = 512 * 1024**2
MAX_OUTPUT_BYTES: Final = 128 * 1024**2
PRODUCTION_METADATA_SHA256: Final = (
    "a8f579e2335bcb5400a326f657527082ad73cbc8d2e1f264c628ef1ceabe48e"
)


class ExposureState(StrEnum):
    """Monotone outcome-exposure state, ordered least to most restrictive."""

    METADATA_ONLY = "metadata_only"
    OUTCOME_SEALED = "outcome_sealed"
    PUBLISHED_SUMMARY_KNOWN = "published_summary_known"
    EXPOSURE_UNKNOWN = "exposure_unknown"
    DERIVATIVE_OF_EXPOSED = "derivative_of_exposed"
    OUTCOME_INSPECTED = "outcome_inspected"
    FIXED_ANSWER_FIXTURE = "fixed_answer_fixture"


class ResultStatus(StrEnum):
    """Whether a requested result exists."""

    PRODUCED = "produced"
    EXCLUDED = "excluded"
    REFUSED = "refused"
    NOT_RUN = "not_run"


_EXPOSURE_ORDER: Final = {state: index for index, state in enumerate(ExposureState)}


def join_exposure(left: ExposureState, right: ExposureState) -> ExposureState:
    """Return the more restrictive state without permitting downgrade."""

    return max((left, right), key=_EXPOSURE_ORDER.__getitem__)


@dataclass(frozen=True, slots=True)
class AcquisitionSpec:
    """Hash-bound source and acquisition identity."""

    acquisition_id: str
    provider_file_id: int
    provider_path: str
    local_name: str
    sha256: str
    provider_md5: str
    row_count: int
    metadata_path: tuple[str, ...]
    construct_id: str
    initial_exposure: ExposureState

    @property
    def source_id(self) -> str:
        return f"figshare:30344995:v3:file:{self.provider_file_id}"

    @property
    def byte_id(self) -> str:
        return f"sha256:{self.sha256}"

    @property
    def connected_group_id(self) -> str:
        return f"cg:{self.acquisition_id}"


@dataclass(frozen=True, slots=True)
class NpyHeader:
    """Allocation-free description of one NPY member."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    fortran_order: bool
    header_bytes: int
    numeric_bytes: int


@dataclass(frozen=True, slots=True)
class PreflightRecord:
    """Authenticated archive structure established without numeric access."""

    acquisition_id: str
    archive_bytes: int
    sha256: str
    provider_md5: str
    headers: tuple[NpyHeader, ...]


@dataclass(frozen=True, slots=True)
class ExposureWindow:
    """One nominal exposure interval; never a measured timestamp."""

    timing_realization_id: str
    source_state_block_index: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float = 0.2
    timestamp_source: str = "nominal_metadata"


@dataclass(frozen=True, slots=True)
class CycleObservation:
    """One retained technical-cycle estimate."""

    cycle_id: str
    trace_id: str
    cycle_index: int
    transform_id: str
    eligible: bool
    reason_codes: tuple[str, ...]
    numerator: float | None
    denominator: float | None
    relative_magnitude: float | None


@dataclass(frozen=True, slots=True)
class RowSummary:
    """One anonymous deposited trace-row summary."""

    trace_id: str
    acquisition_id: str
    transform_id: str
    result_status: ResultStatus
    deposited_cycle_count: int
    eligible_cycle_count: int
    excluded_cycle_count: int
    median_cycle_ratio: float | None
    ratio_unit: str
    reason_codes: tuple[str, ...]
    uncertainty_label: str = "descriptive_only_no_independent_biological_replicates"


@dataclass(frozen=True, slots=True)
class FrameObservation:
    """One retained frame with explicit inherited phase and timing."""

    frame_id: str
    trace_id: str
    cycle_id: str
    transform_id: str
    source_frame_index: int
    retained_frame_index: int
    source_period_index: int
    nominal_state: str
    phase_frame_index: int
    deposited_value: float | None
    baseline: float | None
    normalized_residual: float | None
    numeric_status: str
    eligible: bool
    reason_codes: tuple[str, ...]
    signal_unit: str
    residual_unit: str
    exposure_windows: tuple[ExposureWindow, ...]


@dataclass(frozen=True, slots=True)
class TraceAnalysis:
    """Complete transform result for one deposited trace row."""

    row: RowSummary
    cycles: tuple[CycleObservation, ...]
    frames: tuple[FrameObservation, ...]


@dataclass(frozen=True, slots=True)
class AcquisitionSummary:
    """Within-acquisition descriptive result."""

    acquisition_id: str
    construct_id: str
    result_status: ResultStatus
    deposited_row_count: int
    eligible_row_count: int
    eligible_cycle_count: int
    median_row_median: float | None
    row_median_distribution: tuple[float, ...] | None
    denominator_weighted_reconciliation: float | None
    exclusion_reason_counts: tuple[tuple[str, int], ...]


OXFORD_V3_ACQUISITIONS: Final = (
    AcquisitionSpec(
        "oxv3-mfe-fig1c-20250723-e5-r0",
        59258873,
        "upload/Fig1_PanelC/MFE_single_cell_data_0.npz",
        "59258873_MFE_single_cell_data_0.npz",
        "49a58847a9ac1cecdb819d8340f82f7991ad48ffe78e01353d9de0ae5c1eb41b",
        "e278f432f3acb9b261cc810ddcd53dbc",
        153,
        ("Fig1_PanelC",),
        "construct:maglov2-r10",
        ExposureState.FIXED_ANSWER_FIXTURE,
    ),
    AcquisitionSpec(
        "oxv3-mfe-fig2-maglov-20250730-e10-r0",
        59258822,
        "upload/Fig2/MFE/MagLOV/MFE_single_cell_data_0.npz",
        "59258822_MFE_single_cell_data_0.npz",
        "c80864843b1557ddaf0c3609a9dea1d4c69f4293c2516098a02718e0232350fd",
        "54fd63aad881b034f33c6771f4982c83",
        120,
        ("Fig2", "MFE", "MagLOV"),
        "construct:maglov-r6",
        ExposureState.PUBLISHED_SUMMARY_KNOWN,
    ),
    AcquisitionSpec(
        "oxv3-mfe-fig2-maglov2-20250730-e14-r0",
        59258855,
        "upload/Fig2/MFE/MagLOV 2/MFE_single_cell_data_0.npz",
        "59258855_MFE_single_cell_data_0.npz",
        "054a26eb1fd92a3b0312c1dc98c361c9e34be770c218b7359801251b798e4bb2",
        "ad7d93a2a0640c7efda814fa2612f7ed",
        57,
        ("Fig2", "MFE", "MagLOV 2"),
        "construct:maglov2-r10",
        ExposureState.PUBLISHED_SUMMARY_KNOWN,
    ),
    AcquisitionSpec(
        "oxv3-mfe-fig2-aslov2r2-20250731-e0-r0",
        59258870,
        "upload/Fig2/MFE/AsLOV2 R2/MFE_single_cell_data_0.npz",
        "59258870_MFE_single_cell_data_0.npz",
        "144bbceea4924e2f374e738eb937e9fd9caf6480abcc2fb50e12466b7e329ae6",
        "8532390075e6f27ea806fb9746dd2f40",
        143,
        ("Fig2", "MFE", "AsLOV2 R2"),
        "construct:aslov2-r2",
        ExposureState.PUBLISHED_SUMMARY_KNOWN,
    ),
    AcquisitionSpec(
        "oxv3-mfe-fig2-maglov2fast-20250731-e4-r0",
        59258882,
        "upload/Fig2/MFE/MagLOV 2 fast/MFE_single_cell_data_0.npz",
        "59258882_MFE_single_cell_data_0.npz",
        "aaca5ae5b4467652c28ab7953c85d00947a00ecc7e9105a748f11372b9860e2f",
        "74574006ecb570a46b3cc5ed578a9ffa",
        171,
        ("Fig2", "MFE", "MagLOV 2 fast"),
        "construct:maglov2-fast-r11f",
        ExposureState.PUBLISHED_SUMMARY_KNOWN,
    ),
)


def _hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _read_npy_header(handle: BinaryIO, name: str, member_size: int) -> NpyHeader:
    version = np.lib.format.read_magic(handle)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
    else:
        raise ValueError("unsupported_npy_version")
    if dtype.hasobject or dtype.fields is not None:
        raise ValueError("unsafe_npy_dtype")
    element_count = 1
    for dimension in shape:
        if dimension < 0:
            raise ValueError("negative_npy_dimension")
        element_count *= dimension
        if element_count > 10_240_000:
            raise ValueError("npy_element_limit_exceeded")
    numeric_bytes = element_count * dtype.itemsize
    header_bytes = handle.tell()
    if header_bytes + numeric_bytes != member_size:
        raise ValueError("npy_size_or_trailing_payload_mismatch")
    return NpyHeader(
        name,
        dtype.str,
        tuple(shape),
        bool(fortran_order),
        header_bytes,
        numeric_bytes,
    )


def preflight_npz(path: Path, spec: AcquisitionSpec) -> PreflightRecord:
    """Authenticate an archive and inspect ZIP/NPY headers without numeric access."""

    if _hash_file(path) != spec.sha256:
        raise ValueError("source_sha256_mismatch")
    if _hash_file(path, "md5") != spec.provider_md5:
        raise ValueError("provider_md5_mismatch")
    headers: list[NpyHeader] = []
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if {info.filename for info in infos} != EXPECTED_MEMBERS or len(infos) != 3:
            raise ValueError("unexpected_zip_members")
        if sum(info.file_size for info in infos) > 16 * 1024**2:
            raise ValueError("expanded_archive_limit_exceeded")
        for info in infos:
            member_path = Path(info.filename)
            if info.flag_bits & 1 or member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError("unsafe_zip_member")
            with archive.open(info) as handle:
                headers.append(
                    _read_npy_header(cast(BinaryIO, handle), info.filename, info.file_size)
                )
    by_name = {header.name: header for header in headers}
    expected = {
        "time_traces.npy": ("<f4", (spec.row_count, 900)),
        "mask.npy": ("|b1", (3200, 3200)),
        "background.npy": ("<f4", ()),
    }
    for name, (dtype, shape) in expected.items():
        header = by_name[name]
        if header.dtype != dtype or header.shape != shape or header.fortran_order:
            raise ValueError(f"invalid_npy_contract:{name}")
    return PreflightRecord(
        spec.acquisition_id,
        path.stat().st_size,
        spec.sha256,
        spec.provider_md5,
        tuple(sorted(headers, key=lambda header: header.name)),
    )


def require_resource_budget(
    path: Path,
    *,
    persistent_bytes: int,
    scratch_bytes: int,
    transient_bytes: int,
    free_bytes: int | None = None,
) -> None:
    """Refuse planned resource use outside the reviewed fixed budget."""

    if transient_bytes < 0 or transient_bytes > MAX_TRANSIENT_BYTES:
        raise ValueError("transient_allocation_budget_exceeded")
    if scratch_bytes < 0 or scratch_bytes > MAX_SCRATCH_BYTES:
        raise ValueError("scratch_budget_exceeded")
    if persistent_bytes < 0 or persistent_bytes > MAX_TRANSIENT_BYTES:
        raise ValueError("persistent_output_budget_exceeded")
    available = shutil.disk_usage(path).free if free_bytes is None else free_bytes
    if available < MINIMUM_FREE_BYTES + persistent_bytes + scratch_bytes:
        raise ValueError("free_space_reserve_breached")


def validate_partitions(
    development: Iterable[str], training: Iterable[str], holdout: Iterable[str]
) -> None:
    """Refuse a connected leakage component appearing in multiple partitions."""

    partitions = (set(development), set(training), set(holdout))
    if any(
        partitions[left] & partitions[right] for left in range(3) for right in range(left + 1, 3)
    ):
        raise ValueError("connected_group_partition_overlap")


def validate_component_partitions(
    component_by_item: Mapping[str, str], partitions: Mapping[str, Iterable[str]]
) -> None:
    """Refuse byte/derivative/acquisition components crossing any split."""

    component_partitions: dict[str, set[str]] = {}
    for partition, items in partitions.items():
        for item in items:
            if item not in component_by_item:
                raise ValueError("unresolved_leakage_component")
            component_partitions.setdefault(component_by_item[item], set()).add(partition)
    if any(len(names) > 1 for names in component_partitions.values()):
        raise ValueError("connected_component_partition_overlap")


def _timing_windows(source_period: int, state_index: int, phase: int) -> tuple[ExposureWindow, ...]:
    state_block = source_period * 2 + state_index
    state_start = float(state_block * 10)
    relative_starts = (phase * (10.0 / 45.0), phase * 0.2, 1.0 + phase * 0.2)
    identifiers = ("uniform_slots", "contiguous_then_gap", "gap_then_contiguous")
    return tuple(
        ExposureWindow(identifier, state_block, state_start + start, state_start + start + 0.2)
        for identifier, start in zip(identifiers, relative_starts, strict=True)
    )


def analyze_trace_row(
    signal_input: np.ndarray, spec: AcquisitionSpec, row_index: int
) -> TraceAnalysis:
    """Apply the frozen nine-anchor transform to one anonymous deposited row."""

    if signal_input.shape != (900,) or np.iscomplexobj(signal_input):
        raise ValueError("trace_shape_or_type_mismatch")
    signal = np.asarray(signal_input[90:], dtype=np.float64)
    trace_id = f"{spec.acquisition_id}:trace:{row_index}"
    reasons: list[str] = []
    baseline: np.ndarray | None = None
    if not np.isfinite(signal).all():
        reasons.append("nonfinite_retained_trace")
    else:
        anchors = np.arange(43, 810, 90, dtype=np.int64)
        try:
            spline = make_interp_spline(anchors, signal[anchors], k=3, bc_type=None)
            baseline = np.asarray(
                spline(np.arange(810, dtype=np.float64), extrapolate=True),
                dtype=np.float64,
            )
        except (ValueError, np.linalg.LinAlgError):
            reasons.append("spline_fit_failed")
        if baseline is not None:
            if not np.isfinite(baseline).all():
                reasons.append("nonfinite_baseline_retained_frame")
            if np.any(baseline <= 0.0):
                reasons.append("nonpositive_baseline_retained_frame")
    cycles: list[CycleObservation] = []
    for retained_cycle in range(9):
        source_period = retained_cycle + 1
        cycle_id = f"{trace_id}:cycle:{source_period}"
        cycle_reasons = list(reasons)
        numerator: float | None = None
        denominator: float | None = None
        magnitude: float | None = None
        if not cycle_reasons and baseline is not None:
            start = retained_cycle * 90 + 80
            stop = start + 10
            numerator = float(np.sum(signal[start:stop] - baseline[start:stop], dtype=np.float64))
            denominator = float(np.sum(baseline[start:stop], dtype=np.float64))
            if not isfinite(numerator) or not isfinite(denominator):
                cycle_reasons.append("nonfinite_cycle_estimate")
            elif denominator <= 0.0:
                cycle_reasons.append("nonpositive_cycle_denominator")
            else:
                magnitude = numerator / denominator
        eligible = not cycle_reasons
        cycles.append(
            CycleObservation(
                cycle_id,
                trace_id,
                retained_cycle,
                TRANSFORM_ID,
                eligible,
                tuple(cycle_reasons),
                numerator if eligible else None,
                denominator if eligible else None,
                magnitude if eligible else None,
            )
        )
    magnitudes = [cast(float, cycle.relative_magnitude) for cycle in cycles if cycle.eligible]
    row_reasons = tuple(sorted({reason for cycle in cycles for reason in cycle.reason_codes}))
    row = RowSummary(
        trace_id,
        spec.acquisition_id,
        TRANSFORM_ID,
        ResultStatus.PRODUCED if magnitudes else ResultStatus.EXCLUDED,
        9,
        len(magnitudes),
        9 - len(magnitudes),
        float(np.median(magnitudes)) if magnitudes else None,
        "dimensionless_relative_signal",
        row_reasons,
    )
    frames: list[FrameObservation] = []
    for retained_index, value in enumerate(signal):
        retained_cycle, local_index = divmod(retained_index, 90)
        source_period = retained_cycle + 1
        state_index = int(local_index >= 45)
        phase = local_index % 45
        cycle = cycles[retained_cycle]
        finite_value = isfinite(float(value))
        baseline_value = float(baseline[retained_index]) if baseline is not None else None
        eligible = cycle.eligible and finite_value and baseline_value is not None
        frames.append(
            FrameObservation(
                f"{trace_id}:frame:{retained_index}",
                trace_id,
                cycle.cycle_id,
                TRANSFORM_ID,
                retained_index + 90,
                retained_index,
                source_period,
                "OFF" if state_index == 0 else "ON",
                phase,
                float(value) if finite_value else None,
                baseline_value if eligible else None,
                float(value / baseline_value - 1.0) if eligible else None,
                (
                    "finite"
                    if finite_value
                    else (
                        "nan"
                        if np.isnan(value)
                        else "positive_infinity"
                        if value > 0
                        else "negative_infinity"
                    )
                ),
                eligible,
                () if eligible else row_reasons,
                "deposited_roi_fluorescence_unit_not_documented",
                "dimensionless_relative_signal",
                _timing_windows(source_period, state_index, phase),
            )
        )
    return TraceAnalysis(row, tuple(cycles), tuple(frames))


def summarize_acquisition(
    spec: AcquisitionSpec, analyses: Sequence[TraceAnalysis]
) -> AcquisitionSummary:
    """Summarize rows without treating them as biological replicates."""

    row_values = [
        analysis.row.median_cycle_ratio
        for analysis in analyses
        if analysis.row.median_cycle_ratio is not None
    ]
    cycles = [cycle for analysis in analyses for cycle in analysis.cycles if cycle.eligible]
    reason_counts = Counter(reason for analysis in analyses for reason in analysis.row.reason_codes)
    numerator = sum(cast(float, cycle.numerator) for cycle in cycles)
    denominator = sum(cast(float, cycle.denominator) for cycle in cycles)
    return AcquisitionSummary(
        spec.acquisition_id,
        spec.construct_id,
        ResultStatus.PRODUCED if row_values else ResultStatus.EXCLUDED,
        len(analyses),
        len(row_values),
        len(cycles),
        float(np.median(row_values)) if row_values else None,
        tuple(sorted(row_values)) if row_values else None,
        numerator / denominator if cycles and denominator > 0.0 else None,
        tuple(sorted(reason_counts.items())),
    )


def freeze_document(
    preflights: Sequence[PreflightRecord], exposures: Mapping[str, ExposureState]
) -> dict[str, object]:
    """Create the canonical outcome-blind Tier-D freeze document."""

    development = sorted(f"cg:{record.acquisition_id}" for record in preflights)
    validate_partitions(development, (), ())
    return {
        "schema_version": "paircraft.oxford_calibration.pre_read_freeze.v1",
        "feasibility_tier": "D_descriptive",
        "development_group_ids": development,
        "training_group_ids": [],
        "holdout_group_ids": [],
        "preflights": [
            asdict(record) for record in sorted(preflights, key=lambda item: item.acquisition_id)
        ],
        "exposures": {key: exposures[key].value for key in sorted(exposures)},
        "transform": {
            "transform_id": TRANSFORM_ID,
            "discard_source_periods": 1,
            "retained_anchor_indices": [43 + 90 * index for index in range(9)],
            "spline_degree": 3,
            "boundary_condition": "not-a-knot",
            "extrapolate": True,
            "late_on_indices": list(range(80, 90)),
        },
        "timing_realizations": [
            "uniform_slots",
            "contiguous_then_gap",
            "gap_then_contiguous",
        ],
        "claim_boundary": "descriptive_only_no_model_fitting_or_variant_ranking",
    }


def persist_freeze(output_root: Path, document: Mapping[str, object]) -> tuple[str, Path]:
    """Atomically write and durably fsync a content-addressed freeze."""

    encoded = _canonical_bytes(document) + b"\n"
    digest = hashlib.sha256(encoded).hexdigest()
    directory = output_root / "freezes"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{digest}.json"
    if target.exists():
        if target.read_bytes() != encoded:
            raise ValueError("freeze_content_address_collision")
        return digest, target
    descriptor, temporary_name = tempfile.mkstemp(prefix=".freeze-", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
        directory_descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return digest, target


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_access_journal(path: Path) -> tuple[dict[str, object], ...]:
    """Validate and return a complete hash-chained journal."""

    if not path.exists():
        return ()
    events: list[dict[str, object]] = []
    previous: str | None = None
    with path.open(encoding="utf-8") as handle:
        for sequence, line in enumerate(handle):
            try:
                event = cast(dict[str, object], json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError("invalid_outcome_access_journal") from error
            claimed_hash = event.pop("event_sha256", None)
            if event.get("sequence") != sequence or event.get("previous_event_sha256") != previous:
                raise ValueError("invalid_outcome_access_journal_chain")
            actual_hash = _stable_hash(event)
            if claimed_hash != actual_hash:
                raise ValueError("invalid_outcome_access_journal_hash")
            event["event_sha256"] = actual_hash
            events.append(event)
            previous = actual_hash
    return tuple(events)


def append_access_event(
    output_root: Path,
    spec: AcquisitionSpec,
    freeze_sha256: str,
    *,
    run_id: str,
) -> dict[str, object]:
    """Append and fsync an outcome-read intent before numeric member access."""

    freeze_path = output_root / "freezes" / f"{freeze_sha256}.json"
    if (
        not freeze_path.is_file()
        or hashlib.sha256(freeze_path.read_bytes()).hexdigest() != freeze_sha256
    ):
        raise ValueError("durable_pre_read_freeze_missing")
    access_directory = output_root / "access"
    access_directory.mkdir(parents=True, exist_ok=True)
    journal = access_directory / "outcome-access.jsonl"
    with _exclusive_lock(access_directory / "outcome-access.lock"):
        events = read_access_journal(journal)
        previous = cast(str, events[-1]["event_sha256"]) if events else None
        event: dict[str, object] = {
            "sequence": len(events),
            "event_id": f"{run_id}:{spec.provider_file_id}:{len(events)}",
            "previous_event_sha256": previous,
            "source_id": spec.source_id,
            "byte_id": spec.byte_id,
            "connected_group_id": spec.connected_group_id,
            "actor_process": "paircraft-cli",
            "utc_timestamp": datetime.now(UTC).isoformat(),
            "access_purpose": "tier_d_descriptive",
            "accessed_member": "time_traces.npy",
            "event_phase": "before_open",
            "pre_read_freeze_sha256": freeze_sha256,
        }
        event["event_sha256"] = _stable_hash(event)
        new_file = not journal.exists()
        with journal.open("ab") as handle:
            handle.write(_canonical_bytes(event) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        if new_file:
            descriptor = os.open(access_directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return event


def _read_numeric_members(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    with zipfile.ZipFile(path) as archive:
        with archive.open("time_traces.npy") as handle:
            traces = np.lib.format.read_array(handle, allow_pickle=False)
        with archive.open("mask.npy") as handle:
            mask = np.lib.format.read_array(handle, allow_pickle=False)
        with archive.open("background.npy") as handle:
            background_array = np.lib.format.read_array(handle, allow_pickle=False)
    background = float(background_array)
    if not isfinite(background):
        raise ValueError("nonfinite_background")
    return traces, mask, background


def read_admitted_arrays(
    path: Path,
    spec: AcquisitionSpec,
    preflight: PreflightRecord,
    output_root: Path,
    freeze_sha256: str,
    *,
    run_id: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Reauthenticate, journal, then and only then open numeric members."""

    output_root.mkdir(parents=True, exist_ok=True)
    if preflight != preflight_npz(path, spec):
        raise ValueError("preflight_state_changed")
    require_resource_budget(
        output_root,
        persistent_bytes=0,
        scratch_bytes=0,
        transient_bytes=sum(header.numeric_bytes for header in preflight.headers),
    )
    append_access_event(output_root, spec, freeze_sha256, run_id=run_id)
    return _read_numeric_members(path)


def _protocol_record(metadata: Mapping[str, object], spec: AcquisitionSpec) -> dict[str, object]:
    node: object = metadata
    for part in spec.metadata_path:
        if not isinstance(node, Mapping) or part not in node:
            raise ValueError("metadata_path_missing")
        node = node[part]
    if not isinstance(node, Mapping):
        raise ValueError("metadata_record_invalid")
    expected = {
        "ImageExposure_ms": 200,
        "ImagesPerStep": 45,
        "NumSteps": 10,
        "MagnetOnTime": 10000,
        "MagnetCurrentA": 0.22,
        "NSweeps": 1,
        "LED": 5,
        "LEDIntensity": 0.05,
        "PhotobleachTime_s": 0,
        "repeat": 0,
    }
    for key, expected_value in expected.items():
        value = node.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float) or value != expected_value:
            raise ValueError(f"protocol_metadata_mismatch:{key}")
    return dict(node)


def _csv_value(value: object) -> object:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, tuple):
        return ";".join(str(item) for item in value)
    return value


def _write_csv_record(
    handle: io.TextIOBase,
    writer: csv.DictWriter[str] | None,
    record: Mapping[str, object],
) -> csv.DictWriter[str]:
    if writer is None:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        writer.writeheader()
    writer.writerow({key: _csv_value(value) for key, value in record.items()})
    return writer


@contextmanager
def _gzip_text(path: Path) -> Iterator[io.TextIOWrapper]:
    with (
        path.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text,
    ):
        yield text


def _frame_record(frame: FrameObservation) -> dict[str, object]:
    windows = {window.timing_realization_id: window for window in frame.exposure_windows}
    record = asdict(frame)
    del record["exposure_windows"]
    for identifier in ("uniform_slots", "contiguous_then_gap", "gap_then_contiguous"):
        record[f"{identifier}_start_s"] = windows[identifier].start_seconds
        record[f"{identifier}_end_s"] = windows[identifier].end_seconds
    return record


def _trajectory_records(
    spec: AcquisitionSpec, values: Mapping[tuple[str, int], list[float]]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for state_index, state in enumerate(("OFF", "ON")):
        for phase in range(45):
            observed = values[(state, phase)]
            records.append(
                {
                    "acquisition_id": spec.acquisition_id,
                    "transform_id": TRANSFORM_ID,
                    "nominal_state": state,
                    "phase_frame_index": phase,
                    "result_status": (
                        ResultStatus.PRODUCED.value if observed else ResultStatus.EXCLUDED.value
                    ),
                    "eligible_row_cycle_count": len(observed),
                    "median_normalized_residual": (
                        float(np.median(observed)) if observed else None
                    ),
                    "residual_unit": "dimensionless_relative_signal",
                    "nominal_exposure_windows": [
                        asdict(window) for window in _timing_windows(1, state_index, phase)
                    ],
                }
            )
    return records


def _fsync_tree_files(directory: Path) -> int:
    total = 0
    for path in directory.iterdir():
        total += path.stat().st_size
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    return total


def build_oxford_calibration_corpus(
    input_root: Path,
    metadata_path: Path,
    output_root: Path,
    result_name: str,
    *,
    specs: Sequence[AcquisitionSpec] = OXFORD_V3_ACQUISITIONS,
    metadata_sha256: str = PRODUCTION_METADATA_SHA256,
) -> Path:
    """Build one atomic descriptive generation after freeze and access barriers."""

    if not result_name or result_name in {".", ".."} or Path(result_name).name != result_name:
        raise ValueError("invalid_result_name")
    output_root.mkdir(parents=True, exist_ok=True)
    require_resource_budget(
        output_root,
        persistent_bytes=MAX_OUTPUT_BYTES,
        scratch_bytes=MAX_SCRATCH_BYTES,
        transient_bytes=MAX_TRANSIENT_BYTES,
    )
    paths = {spec.acquisition_id: input_root / spec.local_name for spec in specs}
    preflights = [preflight_npz(paths[spec.acquisition_id], spec) for spec in specs]
    if sum(record.archive_bytes for record in preflights) > 64 * 1024**2:
        raise ValueError("aggregate_archive_limit_exceeded")
    if _hash_file(metadata_path) != metadata_sha256:
        raise ValueError("metadata_sha256_mismatch")
    metadata = cast(dict[str, object], json.loads(metadata_path.read_text(encoding="utf-8")))
    protocols = {spec.acquisition_id: _protocol_record(metadata, spec) for spec in specs}
    journal = output_root / "access" / "outcome-access.jsonl"
    previous_events = read_access_journal(journal)
    exposures = {spec.connected_group_id: spec.initial_exposure for spec in specs}
    for event in previous_events:
        group = cast(str, event["connected_group_id"])
        if group in exposures:
            exposures[group] = join_exposure(exposures[group], ExposureState.OUTCOME_INSPECTED)
    freeze_sha256, freeze_path = persist_freeze(output_root, freeze_document(preflights, exposures))
    run_id = _stable_hash({"freeze": freeze_sha256, "result": result_name})[:16]
    results_root = output_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    destination = results_root / result_name
    if destination.exists():
        raise FileExistsError(f"result output already exists: {destination}")
    staging = Path(tempfile.mkdtemp(prefix=f".staging-{run_id}-", dir=results_root))
    summaries: list[AcquisitionSummary] = []
    artifacts: list[dict[str, object]] = []
    try:
        with (
            (staging / "rows.csv").open("w", encoding="utf-8", newline="") as row_handle,
            (staging / "cycles.csv").open("w", encoding="utf-8", newline="") as cycle_handle,
            (staging / "trajectories.jsonl").open("w", encoding="utf-8") as trajectory_handle,
            _gzip_text(staging / "frames.csv.gz") as frame_handle,
        ):
            row_writer: csv.DictWriter[str] | None = None
            cycle_writer: csv.DictWriter[str] | None = None
            frame_writer: csv.DictWriter[str] | None = None
            for spec, preflight in zip(specs, preflights, strict=True):
                require_resource_budget(
                    output_root,
                    persistent_bytes=MAX_OUTPUT_BYTES,
                    scratch_bytes=MAX_SCRATCH_BYTES,
                    transient_bytes=sum(header.numeric_bytes for header in preflight.headers),
                )
                traces, mask, background = read_admitted_arrays(
                    paths[spec.acquisition_id],
                    spec,
                    preflight,
                    output_root,
                    freeze_sha256,
                    run_id=run_id,
                )
                analyses: list[TraceAnalysis] = []
                trajectory_values: dict[tuple[str, int], list[float]] = {
                    (state, phase): [] for state in ("OFF", "ON") for phase in range(45)
                }
                for row_index, trace in enumerate(traces):
                    analysis = analyze_trace_row(trace, spec, row_index)
                    analyses.append(TraceAnalysis(analysis.row, analysis.cycles, ()))
                    row_writer = _write_csv_record(row_handle, row_writer, asdict(analysis.row))
                    for cycle in analysis.cycles:
                        cycle_writer = _write_csv_record(cycle_handle, cycle_writer, asdict(cycle))
                    for frame in analysis.frames:
                        frame_writer = _write_csv_record(
                            frame_handle, frame_writer, _frame_record(frame)
                        )
                        if frame.eligible and frame.normalized_residual is not None:
                            trajectory_values[
                                (frame.nominal_state, frame.phase_frame_index)
                            ].append(frame.normalized_residual)
                summary = summarize_acquisition(spec, analyses)
                summaries.append(summary)
                for record in _trajectory_records(spec, trajectory_values):
                    trajectory_handle.write(_canonical_bytes(record).decode() + "\n")
                events = read_access_journal(journal)
                artifacts.append(
                    {
                        "acquisition_id": spec.acquisition_id,
                        "source_id": spec.source_id,
                        "byte_id": spec.byte_id,
                        "provider_path": spec.provider_path,
                        "access_event_id": events[-1]["event_id"],
                        "background": background,
                        "background_unit": "deposited_camera_count",
                        "union_mask_shape": list(mask.shape),
                        "union_mask_true_pixels": int(np.count_nonzero(mask)),
                        "row_to_mask_identity": None,
                        "instrument_current_a": protocols[spec.acquisition_id]["MagnetCurrentA"],
                        "magnetic_flux_density_t": None,
                        "field_reason": "missing_versioned_current_to_B_calibration",
                    }
                )
                del analyses, traces, mask
        corpus: dict[str, object] = {
            "schema_version": "paircraft.oxford_calibration_corpus.v1",
            "feasibility_tier": "D_descriptive",
            "pre_read_freeze_sha256": freeze_sha256,
            "pre_read_freeze_path": str(freeze_path),
            "transform_id": TRANSFORM_ID,
            "deposited_row_count": sum(spec.row_count for spec in specs),
            "acquisition_summaries": [asdict(summary) for summary in summaries],
            "source_artifacts": artifacts,
            "claim_boundaries": {
                "model_fitting": "not_run",
                "variant_ranking": "not_run",
                "preparation_generalization": "not_estimable",
                "raw_segmentation": "not_reproducible_from_roi_derived_traces",
                "measured_time": "absent",
            },
        }
        (staging / "corpus.json").write_bytes(_canonical_bytes(corpus) + b"\n")
        lines = [
            "# Oxford v3 descriptive calibration corpus",
            "",
            "Tier D only: within-acquisition description, not model fitting or prediction.",
            "",
        ]
        for summary in summaries:
            value = summary.median_row_median
            rendered = "excluded" if value is None else f"{100 * value:.6f}%"
            lines.append(
                f"- `{summary.acquisition_id}`: {rendered} "
                f"({summary.eligible_row_count}/{summary.deposited_row_count} eligible rows)"
            )
        (staging / "AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if _fsync_tree_files(staging) > MAX_OUTPUT_BYTES:
            raise ValueError("persistent_output_budget_exceeded")
        os.replace(staging, destination)
        directory_descriptor = os.open(results_root, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return destination
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
