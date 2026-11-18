"""Authenticated reader for the deposited York MagLOV Figure 6 trace."""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
import tifffile

from paircraft.experiments.measured import (
    DeclaredMagnetSchedule,
    MagnetControlState,
    MeasuredDatum,
    PublicationStatus,
    ReportedFieldCondition,
    SourceBundle,
    SourceMember,
    SourceMemberRole,
    SourceReportedCondition,
    SourceReportStatus,
    declared_schedule_states,
)

_PREFIX = "AndrewGYork-gfp_magnetofluorescence-500afd6"
_DATA_PREFIX = f"{_PREFIX}/images/MagLOV_figure/script_and_data"

YORK_MAGLOV_SOURCE = SourceBundle(
    source_id="york-gfp-magnetofluorescence-maglov-figure-6",
    source_url="https://doi.org/10.5281/zenodo.11406498",
    source_version="v0.0.1; archive root 500afd6",
    archive_sha256="d5ac6c2c4e566c1323ae80e1b0155f00035ec29a1741d387b42a2509b1795cd3",
    members=(
        SourceMember(
            SourceMemberRole.SIGNAL_CSV,
            f"{_DATA_PREFIX}/9.csv",
            "5cfd9d90bf1c92f9c7cc44d655d7831c83f16bb326b6a1b2ced8534feb8b70d5",
        ),
        SourceMember(
            SourceMemberRole.TIMESTAMP_TIFF,
            f"{_DATA_PREFIX}/timestamps.tif",
            "9e45fe1d827798bc895c7f3a0ff8aed2864c54b9f6cdf6fc9e2d05cc7c947d61",
        ),
        SourceMember(
            SourceMemberRole.SCHEDULE_SCRIPT,
            f"{_DATA_PREFIX}/MagLOV_figure.py",
            "847c667a8baa3e7d00b55ca3c765154ff72087e886049e86c7fc2daebe61a3d0",
        ),
        SourceMember(
            SourceMemberRole.CONTEXT_HTML,
            f"{_PREFIX}/index.html",
            "892c01000076e2a3cb38ffa59337006e9211d2d7f9d62c55028e39da3aa82ef7",
        ),
        SourceMember(
            SourceMemberRole.APPENDIX_HTML,
            f"{_PREFIX}/appendix.html",
            "74959e9e6b5c799e56c52b639206f9a5b28d95533039b1b30f1c7bbfa5467624",
        ),
        SourceMember(
            SourceMemberRole.LICENSE,
            f"{_PREFIX}/LICENSE",
            "79e752c55bf489feaf8808414bded63b35079c06981283d4b26b6b7ce744d22b",
        ),
    ),
    license="Creative Commons Attribution 4.0 International (CC BY 4.0)",
    attribution="Andrew G. York et al., GFP-family and LOV-domain magnetofluorescence",
    publication_status=PublicationStatus.PREPRINT_NON_PEER_REVIEWED,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_verified_members(archive: Path, source: SourceBundle) -> dict[SourceMemberRole, bytes]:
    if _file_sha256(archive) != source.archive_sha256.lower():
        raise ValueError("York archive sha256 does not match provenance")
    try:
        with ZipFile(archive) as payload:
            names = payload.namelist()
            result: dict[SourceMemberRole, bytes] = {}
            for member in source.members:
                if names.count(member.exact_location) != 1:
                    raise ValueError(
                        f"York archive requires one exact member: {member.exact_location}"
                    )
                content = payload.read(member.exact_location)
                if _bytes_sha256(content) != member.sha256.lower():
                    raise ValueError(
                        f"York archive member sha256 does not match provenance: {member.role}"
                    )
                result[member.role] = content
    except BadZipFile as error:
        raise ValueError("York payload is not a valid ZIP archive") from error
    return result


def _parse_signal(payload: bytes) -> tuple[tuple[int, ...], tuple[float, ...]]:
    try:
        rows = list(csv.reader(io.StringIO(payload.decode("utf-8-sig"))))
    except UnicodeDecodeError as error:
        raise ValueError("York signal CSV must be UTF-8 text") from error
    if not rows or rows[0] != ["Slice", "Mean"]:
        raise ValueError("York signal CSV must declare Slice,Mean")
    if len(rows) < 3:
        raise ValueError("York signal CSV requires at least two observations")
    try:
        frame_index = tuple(int(row[0]) for row in rows[1:])
        signal = tuple(float(row[1]) for row in rows[1:])
    except (IndexError, ValueError) as error:
        raise ValueError("York signal CSV contains invalid observations") from error
    if any(len(row) != 2 for row in rows[1:]):
        raise ValueError("York signal CSV observations must have exactly two columns")
    if frame_index != tuple(range(1, len(frame_index) + 1)):
        raise ValueError("York Slice values must be sequential and one-based")
    if any(not np.isfinite(value) or value < 0 for value in signal):
        raise ValueError("York Mean values must be finite and nonnegative")
    return frame_index, signal


def _decode_timestamps(payload: bytes, expected_frames: int) -> tuple[float, ...]:
    try:
        encoded = np.asarray(tifffile.imread(io.BytesIO(payload)))
    except (tifffile.TiffFileError, ValueError) as error:
        raise ValueError("York timestamp payload is not a valid scalar TIFF stack") from error
    if (
        encoded.ndim != 3
        or encoded.shape[0] != expected_frames
        or encoded.shape[1] < 1
        or encoded.shape[2] < 14
        or encoded.dtype.hasobject
        or not np.issubdtype(encoded.dtype, np.unsignedinteger)
    ):
        raise ValueError("York timestamp TIFF has an unsupported shape or dtype")
    digits = encoded[:, 0, :14]
    low = digits & 0x0F
    high = digits >> 4
    if np.any(low > 9) or np.any(high > 9):
        raise ValueError("York timestamp TIFF contains invalid BCD digits")
    decimal = low + high * 10
    weights = np.asarray((3_600_000_000, 60_000_000, 1_000_000, 10_000, 100, 1), dtype=np.uint64)
    microseconds = np.sum(decimal[:, 8:14] * weights, axis=1, dtype=np.uint64)
    if np.any(microseconds[1:] <= microseconds[:-1]):
        raise ValueError("York embedded frame timestamps must be strictly increasing")
    elapsed = (microseconds - microseconds[0]).astype(np.float64) * 1e-6
    return tuple(float(value) for value in elapsed)


def load_york_maglov_9(
    archive: Path,
) -> MeasuredDatum:
    """Load the authenticated deposited MagLOV Figure 6 trace without executing source code."""

    return _load_york_maglov_9(archive, YORK_MAGLOV_SOURCE)


def _load_york_maglov_9(archive: Path, source: SourceBundle) -> MeasuredDatum:
    """Internal fixture seam; public callers cannot rebind the pinned York claims."""

    members = _read_verified_members(archive, source)
    frame_index, signal = _parse_signal(members[SourceMemberRole.SIGNAL_CSV])
    time_seconds = _decode_timestamps(members[SourceMemberRole.TIMESTAMP_TIFF], len(frame_index))

    # These constants are interpreted only after the exact source script has been authenticated.
    field_off_transitions = tuple(float(value) for value in np.arange(3.66, 76.0, 9.98))
    field_on_transitions = tuple(float(value) for value in np.arange(8.69, 76.0, 9.98))
    states = declared_schedule_states(time_seconds, field_off_transitions, field_on_transitions)
    schedule = DeclaredMagnetSchedule(
        initial_state=MagnetControlState.FIELD_ON,
        field_off_transition_seconds=field_off_transitions,
        field_on_transition_seconds=field_on_transitions,
        state_by_observation=states,
        source_member_sha256=source.member(SourceMemberRole.SCHEDULE_SCRIPT).sha256,
    )
    context_location = source.member(SourceMemberRole.CONTEXT_HTML).exact_location
    appendix_location = source.member(SourceMemberRole.APPENDIX_HTML).exact_location
    return MeasuredDatum(
        trace_id="york-maglov-figure-6-trace-9",
        sample_label="Round 6 winner (MagLOV)",
        construct_id="AsLOV2-C450P-D540M-Q513K-L496V-G528K",
        frame_index=frame_index,
        time_seconds=time_seconds,
        signal=signal,
        signal_unit="count_per_pixel",
        schedule=schedule,
        reported_fields=(
            ReportedFieldCondition(
                control_state=MagnetControlState.FIELD_ON,
                reported_text="approximately 10 mT magnetic field on",
                approximate_magnitude_tesla=0.010,
                exact_source_location=context_location,
            ),
            ReportedFieldCondition(
                control_state=MagnetControlState.FIELD_OFF,
                reported_text="magnetic field off; residual or ambient field not reported",
                approximate_magnitude_tesla=None,
                exact_source_location=context_location,
            ),
        ),
        conditions=(
            SourceReportedCondition(
                "measurement_system",
                "fluorescence photography system",
                "description",
                SourceReportStatus.SOURCE_REPORTED,
                context_location,
            ),
            SourceReportedCondition(
                "field_actuator",
                "KK-P80/10 Kaka Electric electromagnet below sample",
                "description",
                SourceReportStatus.SOURCE_REPORTED,
                appendix_location,
            ),
            SourceReportedCondition(
                "illumination_center_wavelength",
                4.70e-7,
                "m",
                SourceReportStatus.APPROXIMATE_SOURCE_REPORT,
                context_location,
            ),
            SourceReportedCondition(
                "illumination_half_bandwidth",
                2.0e-8,
                "m",
                SourceReportStatus.APPROXIMATE_SOURCE_REPORT,
                context_location,
            ),
            SourceReportedCondition(
                "illumination_irradiance",
                500.0,
                "W/m^2",
                SourceReportStatus.APPROXIMATE_SOURCE_REPORT,
                context_location,
            ),
            SourceReportedCondition(
                "illumination_temporal_mode",
                "continuous",
                "description",
                SourceReportStatus.APPROXIMATE_SOURCE_REPORT,
                context_location,
            ),
        ),
        provenance=source,
        limitations=(
            "Deposited figure-input Mean trace; underlying pixels and aggregation geometry "
            "are not deposited here.",
            "Field transitions are unsynchronized script annotations, not measured field readback.",
            "The approximately 10 mT ON field is uncalibrated; the OFF-state field is "
            "unquantified.",
            "The deposit/report provides no independent biological-replicate identity or "
            "pointwise uncertainty.",
            "Fluorescence readout is not a radical-pair yield or an oracle calibration target "
            "by itself.",
        ),
    )
