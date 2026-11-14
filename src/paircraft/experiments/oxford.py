"""Strict readers for the public Oxford experimental trace formats."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from paircraft.experiments.model import ExperimentalTrace, SourceProvenance, TraceClass


@dataclass(frozen=True, slots=True)
class OxfordMaryProvenance:
    """Exact source records for an Oxford MARY trace and its two schedule files."""

    raw_trace: SourceProvenance
    configuration: SourceProvenance
    schedule: SourceProvenance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_payload(path: Path, provenance: SourceProvenance) -> None:
    if _sha256(path) != provenance.sha256.lower():
        raise ValueError(f"payload sha256 does not match provenance: {path.name}")


def _json_attributes(configuration: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (key, json.dumps(value, sort_keys=True, separators=(",", ":")))
        for key, value in sorted(configuration.items())
    )


def _require_positive_integer(configuration: dict[str, Any], key: str) -> int:
    value = configuration.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"configuration {key} must be a positive integer")
    return value


def _load_numeric_schedule(path: Path) -> tuple[np.ndarray[Any, Any], ...]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            required = ("timings", "camera_triggers", "magnet_current")
            if set(payload.files) != set(required):
                raise ValueError(
                    "schedule must contain exactly timings, camera_triggers, and magnet_current"
                )
            arrays = tuple(np.asarray(payload[key]) for key in required)
    except ValueError as error:
        raise ValueError(f"unsafe or invalid Oxford schedule: {error}") from error
    for array in arrays:
        if array.ndim != 1 or array.dtype.hasobject or not np.issubdtype(array.dtype, np.number):
            raise ValueError("schedule arrays must be one-dimensional non-object numeric arrays")
        if not np.isfinite(array).all():
            raise ValueError("schedule arrays must contain only finite values")
    return arrays


def load_oxford_mary_scalar_trace(
    raw_trace_path: Path,
    configuration_path: Path,
    schedule_path: Path,
    *,
    provenance: OxfordMaryProvenance,
    construct_id: str,
    trace_id: str,
) -> ExperimentalTrace:
    """Load one raw scalar MARY OME-TIFF trace with its exact schedule.

    The Oxford format contains one terminal schedule event after the final camera frame. This
    reader accepts that offset only when the extra event disables the camera trigger and returns
    magnet current to exactly zero.
    """

    _verify_payload(raw_trace_path, provenance.raw_trace)
    _verify_payload(configuration_path, provenance.configuration)
    _verify_payload(schedule_path, provenance.schedule)

    configuration_value = json.loads(configuration_path.read_text(encoding="utf-8"))
    if not isinstance(configuration_value, dict):
        raise ValueError("Oxford configuration must be a JSON object")
    configuration: dict[str, Any] = configuration_value
    sample_label = configuration.get("Sample")
    if not isinstance(sample_label, str) or not sample_label.strip():
        raise ValueError("Oxford configuration must declare Sample")
    periods = _require_positive_integer(configuration, "NumPeriods")
    frames_per_state = _require_positive_integer(configuration, "ImagesPerOnPeriod")
    configured_currents = configuration.get("MagnetCurrent_A")
    if (
        not isinstance(configured_currents, list)
        or len(configured_currents) != periods
        or any(
            isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value)
            for value in configured_currents
        )
    ):
        raise ValueError("MagnetCurrent_A must contain one finite numeric value per period")

    expected_frames = periods * frames_per_state * 2
    with tifffile.TiffFile(raw_trace_path) as tif:
        if len(tif.pages) != expected_frames:
            raise ValueError("raw frame count contradicts NumPeriods and ImagesPerOnPeriod")
        signal_values: list[float] = []
        for page in tif.pages:
            if int(np.prod(page.shape)) != 1:
                raise ValueError("MARY scalar reader refuses non-scalar TIFF frames")
            page_array = page.asarray()
            if page_array.size != 1 or not np.issubdtype(page_array.dtype, np.number):
                raise ValueError("MARY scalar reader refuses non-scalar TIFF frames")
            signal_values.append(float(page_array.item()))
    signal = np.asarray(signal_values, dtype=np.float64)
    if not np.isfinite(signal).all():
        raise ValueError("raw fluorescence trace must contain only finite values")
    timings, camera_triggers, magnet_current = _load_numeric_schedule(schedule_path)
    if not (len(timings) == len(camera_triggers) == len(magnet_current) == signal.size + 1):
        raise ValueError("MARY schedule must contain exactly one terminal event after all frames")
    if camera_triggers[-1] != -1 or magnet_current[-1] != 0:
        raise ValueError("terminal schedule event must disable the camera and zero magnet current")
    frame_timings = timings[:-1].astype(np.float64)
    frame_triggers = camera_triggers[:-1]
    frame_current = magnet_current[:-1].astype(np.float64)
    if np.any(frame_timings <= 0) or np.any(frame_triggers <= 0):
        raise ValueError("every acquired frame must have positive timing and camera trigger")

    for period_index, configured_current in enumerate(configured_currents):
        start = period_index * frames_per_state * 2
        middle = start + frames_per_state
        stop = middle + frames_per_state
        if not np.all(frame_current[start:middle] == 0.0):
            raise ValueError("each MARY period must begin with a zero-current OFF block")
        if not np.allclose(
            frame_current[middle:stop], float(configured_current), rtol=1e-12, atol=0.0
        ):
            raise ValueError("schedule ON block contradicts configured magnet current")

    elapsed_seconds = np.concatenate(
        (np.array([0.0]), np.cumsum(frame_timings[:-1], dtype=np.float64) / 1000.0)
    )
    return ExperimentalTrace(
        trace_id=trace_id,
        sample_label=sample_label,
        construct_id=construct_id,
        trace_class=TraceClass.INSTRUMENT_EXPORT_RAW_TRACE,
        observable="scalar fluorescence intensity",
        signal_unit="camera_count",
        time_seconds=tuple(float(value) for value in elapsed_seconds),
        signal=tuple(float(value) for value in signal),
        magnet_current_ampere=tuple(float(value) for value in frame_current),
        attributes=_json_attributes(configuration),
        conditions=(
            "Oxford MARY acquisition; raw scalar OME-TIFF frames aligned to the deposited "
            "camera-trigger and magnet-current schedule."
        ),
        provenance=(provenance.raw_trace, provenance.configuration, provenance.schedule),
    )


def load_oxford_photodiode_csv(
    path: Path,
    *,
    provenance: SourceProvenance,
    construct_id: str,
    trace_id: str,
) -> ExperimentalTrace:
    """Load the Oxford key/value-header photodiode CSV format without executing notebooks."""

    _verify_payload(path, provenance)
    rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8-sig"))))
    if not rows or rows[0] != ["Parameter", "Value"]:
        raise ValueError("Oxford CSV must begin with the Parameter,Value header")
    try:
        separator = rows.index([])
    except ValueError as error:
        raise ValueError(
            "Oxford CSV must separate metadata from observations with a blank row"
        ) from error
    metadata_rows = rows[1:separator]
    metadata: dict[str, str] = {}
    for row in metadata_rows:
        if len(row) != 2 or not row[0] or row[0] in metadata:
            raise ValueError("Oxford CSV metadata rows require unique key/value pairs")
        metadata[row[0]] = row[1]
    if rows[separator + 1] != ["Time (s)", "Measurement (W)"]:
        raise ValueError("Oxford CSV observation header must declare seconds and watts")
    observation_rows = rows[separator + 2 :]
    if len(observation_rows) < 2:
        raise ValueError("Oxford CSV requires at least two observations")
    try:
        source_time = np.asarray([float(row[0]) for row in observation_rows], dtype=np.float64)
        signal = np.asarray([float(row[1]) for row in observation_rows], dtype=np.float64)
        current = float(metadata["Current Reading (A)"])
        sample_label = metadata["Sample Description"]
    except (KeyError, ValueError, IndexError) as error:
        raise ValueError("Oxford CSV contains invalid metadata or observations") from error
    if any(len(row) != 2 for row in observation_rows) or not np.isfinite(source_time).all():
        raise ValueError("Oxford CSV observations must be finite two-column values")
    elapsed = source_time - source_time[0]
    attributes = tuple(sorted((*metadata.items(), ("Source time origin (s)", str(source_time[0])))))
    return ExperimentalTrace(
        trace_id=trace_id,
        sample_label=sample_label,
        construct_id=construct_id,
        trace_class=TraceClass.INSTRUMENT_EXPORT_RAW_TRACE,
        observable="optical power",
        signal_unit="W",
        time_seconds=tuple(float(value) for value in elapsed),
        signal=tuple(float(value) for value in signal),
        magnet_current_ampere=(current,) * len(signal),
        attributes=attributes,
        conditions="Oxford photodiode acquisition metadata preserved as immutable attributes.",
        provenance=(provenance,),
    )
