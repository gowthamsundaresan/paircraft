from __future__ import annotations

import hashlib
import io
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest
import tifffile

import paircraft.experiments.york_maglov as york_module
from paircraft.experiments import (
    MagnetControlState,
    MeasurementCorpusMembership,
    MeasurementEvidenceOrigin,
    MeasurementUncertaintyStatus,
    ObservationLayer,
    PublicationStatus,
    ReplicationStatus,
    ReportedFieldStatus,
    ScheduleBasis,
    SourceBundle,
    SourceMember,
    SourceMemberRole,
    SourceReportedCondition,
    SourceReportStatus,
    TraceClass,
    load_york_maglov_9,
    measured_datum_json,
)
from paircraft.experiments.york_maglov import (
    _decode_timestamps,
    _load_york_maglov_9,
    _parse_signal,
    _read_verified_members,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _bcd(value: int) -> int:
    return (value // 10 << 4) | value % 10


def _timestamp_tiff(times_microseconds: tuple[int, ...]) -> bytes:
    stack = np.zeros((len(times_microseconds), 1, 14), dtype=np.uint16)
    for index, total in enumerate(times_microseconds):
        hours, remainder = divmod(total, 3_600_000_000)
        minutes, remainder = divmod(remainder, 60_000_000)
        seconds, remainder = divmod(remainder, 1_000_000)
        ten_ms, remainder = divmod(remainder, 10_000)
        hundred_us, microseconds = divmod(remainder, 100)
        values = (hours, minutes, seconds, ten_ms, hundred_us, microseconds)
        stack[index, 0, 8:14] = tuple(_bcd(value) for value in values)
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, stack, photometric="minisblack")
    return buffer.getvalue()


def _fixture_archive(tmp_path: Path) -> tuple[Path, SourceBundle, dict[SourceMemberRole, bytes]]:
    prefix = "source"
    payloads = {
        SourceMemberRole.SIGNAL_CSV: b"Slice,Mean\n1,10.5\n2,11.5\n3,12.5\n4,13.5\n",
        SourceMemberRole.TIMESTAMP_TIFF: _timestamp_tiff(
            (1_000_000, 5_000_000, 10_000_000, 15_000_000)
        ),
        SourceMemberRole.SCHEDULE_SCRIPT: b"pinned schedule declaration",
        SourceMemberRole.CONTEXT_HTML: b"pinned Figure 6 context",
        SourceMemberRole.APPENDIX_HTML: b"pinned appendix apparatus context",
        SourceMemberRole.LICENSE: b"CC BY 4.0",
    }
    names = {
        SourceMemberRole.SIGNAL_CSV: f"{prefix}/9.csv",
        SourceMemberRole.TIMESTAMP_TIFF: f"{prefix}/timestamps.tif",
        SourceMemberRole.SCHEDULE_SCRIPT: f"{prefix}/figure.py",
        SourceMemberRole.CONTEXT_HTML: f"{prefix}/index.html",
        SourceMemberRole.APPENDIX_HTML: f"{prefix}/appendix.html",
        SourceMemberRole.LICENSE: f"{prefix}/LICENSE",
    }
    archive = tmp_path / "york.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as output:
        for role, payload in payloads.items():
            output.writestr(names[role], payload)
    source = SourceBundle(
        source_id="test-york",
        source_url="https://example.test/york",
        source_version="test-version",
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        members=tuple(
            SourceMember(role, names[role], _sha256(payloads[role])) for role in SourceMemberRole
        ),
        license="CC BY 4.0",
        attribution="Test fixture",
        publication_status=PublicationStatus.PREPRINT_NON_PEER_REVIEWED,
    )
    return archive, source, payloads


def test_loads_signal_decoded_timestamps_and_declared_schedule(tmp_path: Path) -> None:
    archive, source, _ = _fixture_archive(tmp_path)

    datum = _load_york_maglov_9(archive, source)

    assert datum.frame_index == (1, 2, 3, 4)
    assert datum.time_seconds == (0.0, 4.0, 9.0, 14.0)
    assert datum.signal == (10.5, 11.5, 12.5, 13.5)
    assert datum.schedule.state_by_observation == (
        MagnetControlState.FIELD_ON,
        MagnetControlState.FIELD_OFF,
        MagnetControlState.FIELD_ON,
        MagnetControlState.FIELD_OFF,
    )
    field_on, field_off = datum.reported_fields
    assert field_on.approximate_magnitude_tesla == 0.010
    assert field_off.approximate_magnitude_tesla is None
    assert datum.conditions[0].value == "fluorescence photography system"
    assert datum.conditions[1].value == "KK-P80/10 Kaka Electric electromagnet below sample"
    assert datum.conditions[2].value == 4.70e-7
    assert datum.conditions[4].value == 500.0
    assert "no independent biological-replicate identity" in datum.limitations[3]
    assert datum.publication_status is PublicationStatus.PREPRINT_NON_PEER_REVIEWED


def test_public_loader_uses_only_the_pinned_module_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, source, _ = _fixture_archive(tmp_path)
    monkeypatch.setattr(york_module, "YORK_MAGLOV_SOURCE", source)

    assert load_york_maglov_9(archive).provenance is source


def test_refuses_archive_and_member_hash_mismatches(tmp_path: Path) -> None:
    archive, source, _ = _fixture_archive(tmp_path)
    wrong_archive = replace(source, archive_sha256="0" * 64)
    with pytest.raises(ValueError, match="archive sha256"):
        _load_york_maglov_9(archive, wrong_archive)

    signal_member = source.member(SourceMemberRole.SIGNAL_CSV)
    wrong_signal = replace(signal_member, sha256="1" * 64)
    wrong_members = tuple(
        wrong_signal if member.role is SourceMemberRole.SIGNAL_CSV else member
        for member in source.members
    )
    with pytest.raises(ValueError, match="member sha256"):
        _load_york_maglov_9(archive, replace(source, members=wrong_members))


def test_refuses_invalid_csv_and_bcd_payloads(tmp_path: Path) -> None:
    archive, source, payloads = _fixture_archive(tmp_path)
    del archive
    signal_name = source.member(SourceMemberRole.SIGNAL_CSV).exact_location
    timestamp_name = source.member(SourceMemberRole.TIMESTAMP_TIFF).exact_location

    invalid_csv = b"Frame,Value\n1,2\n2,3\n"
    csv_payloads = {**payloads, SourceMemberRole.SIGNAL_CSV: invalid_csv}
    csv_archive, csv_source = _write_payload_variant(tmp_path / "bad-csv.zip", source, csv_payloads)
    assert csv_source.member(SourceMemberRole.SIGNAL_CSV).exact_location == signal_name
    with pytest.raises(ValueError, match="Slice,Mean"):
        _load_york_maglov_9(csv_archive, csv_source)

    negative_payloads = {
        **payloads,
        SourceMemberRole.SIGNAL_CSV: b"Slice,Mean\n1,2\n2,-1\n",
    }
    negative_archive, negative_source = _write_payload_variant(
        tmp_path / "negative.csv.zip", source, negative_payloads
    )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _load_york_maglov_9(negative_archive, negative_source)

    invalid_bcd = bytearray(payloads[SourceMemberRole.TIMESTAMP_TIFF])
    decoded = tifffile.imread(io.BytesIO(invalid_bcd))
    decoded[0, 0, 8] = 0xFA
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, decoded, photometric="minisblack")
    bcd_payloads = {**payloads, SourceMemberRole.TIMESTAMP_TIFF: buffer.getvalue()}
    bcd_archive, bcd_source = _write_payload_variant(tmp_path / "bad-bcd.zip", source, bcd_payloads)
    assert bcd_source.member(SourceMemberRole.TIMESTAMP_TIFF).exact_location == timestamp_name
    with pytest.raises(ValueError, match="invalid BCD"):
        _load_york_maglov_9(bcd_archive, bcd_source)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"\xff\xfe", "UTF-8"),
        (b"Slice,Mean\n1,2\n", "at least two observations"),
        (b"Slice,Mean\n1,2\ninvalid,3\n", "invalid observations"),
        (b"Slice,Mean\n1,2,extra\n2,3\n", "exactly two columns"),
        (b"Slice,Mean\n1,2\n3,3\n", "sequential and one-based"),
        (b"Slice,Mean\n1,nan\n2,3\n", "finite and nonnegative"),
    ],
)
def test_signal_parser_rejects_distinct_malformed_deposit_classes(
    payload: bytes,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _parse_signal(payload)


def test_timestamp_decoder_rejects_invalid_stack_shape_dtype_and_order() -> None:
    with pytest.raises(ValueError, match="valid scalar TIFF stack"):
        _decode_timestamps(b"not-a-tiff", 2)

    wrong_shape = io.BytesIO()
    tifffile.imwrite(wrong_shape, np.zeros((2, 1, 13), dtype=np.uint16))
    with pytest.raises(ValueError, match="unsupported shape or dtype"):
        _decode_timestamps(wrong_shape.getvalue(), 2)

    signed = io.BytesIO()
    tifffile.imwrite(signed, np.zeros((2, 1, 14), dtype=np.int16))
    with pytest.raises(ValueError, match="unsupported shape or dtype"):
        _decode_timestamps(signed.getvalue(), 2)

    with pytest.raises(ValueError, match="strictly increasing"):
        _decode_timestamps(_timestamp_tiff((1_000_000, 1_000_000)), 2)


def test_verified_member_reader_rejects_invalid_zip_and_missing_member(tmp_path: Path) -> None:
    _, source, payloads = _fixture_archive(tmp_path)
    invalid = tmp_path / "invalid.zip"
    invalid.write_bytes(b"not a zip")
    invalid_source = replace(source, archive_sha256=_sha256(invalid.read_bytes()))
    with pytest.raises(ValueError, match="valid ZIP"):
        _read_verified_members(invalid, invalid_source)

    missing = tmp_path / "missing.zip"
    omitted = SourceMemberRole.LICENSE
    with ZipFile(missing, "w", ZIP_DEFLATED) as output:
        for member in source.members:
            if member.role is not omitted:
                output.writestr(member.exact_location, payloads[member.role])
    missing_source = replace(source, archive_sha256=_sha256(missing.read_bytes()))
    with pytest.raises(ValueError, match="requires one exact member"):
        _read_verified_members(missing, missing_source)


def _write_payload_variant(
    archive: Path,
    source: SourceBundle,
    payloads: dict[SourceMemberRole, bytes],
) -> tuple[Path, SourceBundle]:
    with ZipFile(archive, "w", ZIP_DEFLATED) as output:
        for member in source.members:
            output.writestr(member.exact_location, payloads[member.role])
    members = tuple(
        replace(member, sha256=_sha256(payloads[member.role])) for member in source.members
    )
    return archive, replace(
        source,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        members=members,
    )


def test_canonical_document_is_deterministic_and_reference_only(tmp_path: Path) -> None:
    archive, source, _ = _fixture_archive(tmp_path)
    datum = _load_york_maglov_9(archive, source)

    first = measured_datum_json(datum)
    assert first == measured_datum_json(datum)
    assert b'"evidence_origin":"measurement"' in first
    assert b'"membership":"reference_only_not_training"' in first
    assert b'"datum_sha256"' in first
    assert replace(datum, signal=(10.5, 11.5, 12.5, 14.0)).datum_sha256 != datum.datum_sha256
    with pytest.raises(ValueError, match="reference-only"):
        replace(datum, membership="training")  # type: ignore[arg-type]

    assert datum.membership is MeasurementCorpusMembership.REFERENCE_ONLY_NOT_TRAINING


def test_source_and_field_contracts_reject_false_provenance(tmp_path: Path) -> None:
    archive, source, _ = _fixture_archive(tmp_path)
    datum = _load_york_maglov_9(archive, source)
    member = source.members[0]
    field_on = datum.reported_fields[0]

    invalid_members: tuple[Callable[[], object], ...] = (
        lambda: replace(member, role="signal_csv"),  # type: ignore[arg-type]
        lambda: replace(member, exact_location=" "),
        lambda: replace(member, sha256="bad"),
        lambda: replace(source, members=list(source.members)),  # type: ignore[arg-type]
        lambda: replace(source, members=source.members[:-1]),
        lambda: replace(field_on, control_state="field_on"),  # type: ignore[arg-type]
        lambda: replace(field_on, status="calibrated"),  # type: ignore[arg-type]
        lambda: replace(field_on, approximate_magnitude_tesla=-1.0),
    )
    for invalid in invalid_members:
        with pytest.raises(ValueError):
            invalid()


def test_schedule_contract_rejects_invalid_annotations(tmp_path: Path) -> None:
    archive, source, _ = _fixture_archive(tmp_path)
    schedule = _load_york_maglov_9(archive, source).schedule
    invalid_schedules: tuple[Callable[[], object], ...] = (
        lambda: replace(schedule, initial_state=MagnetControlState.FIELD_OFF),
        lambda: replace(schedule, basis="measured"),  # type: ignore[arg-type]
        lambda: replace(schedule, source_member_sha256="bad"),
        lambda: replace(schedule, field_off_transition_seconds=[3.66]),  # type: ignore[arg-type]
        lambda: replace(schedule, field_off_transition_seconds=(-1.0,)),
        lambda: replace(schedule, field_off_transition_seconds=(4.0, 3.0)),
        lambda: replace(schedule, field_off_transition_seconds=()),
        lambda: replace(schedule, field_on_transition_seconds=(2.0,)),
        lambda: replace(schedule, state_by_observation=()),
        lambda: replace(schedule, state_by_observation=("field_on",)),  # type: ignore[arg-type]
    )
    for invalid in invalid_schedules:
        with pytest.raises(ValueError):
            invalid()


def test_measured_datum_rejects_epistemic_and_shape_mutations(tmp_path: Path) -> None:
    archive, source, _ = _fixture_archive(tmp_path)
    datum = _load_york_maglov_9(archive, source)
    wrong_hash_schedule = replace(
        datum.schedule,
        source_member_sha256="a" * 64,
    )
    wrong_states_schedule = replace(
        datum.schedule,
        state_by_observation=(MagnetControlState.FIELD_ON,) * 4,
    )
    invalid_data: tuple[Callable[[], object], ...] = (
        lambda: replace(datum, schema_version="v2"),
        lambda: replace(datum, trace_class=TraceClass.ROI_DERIVED_TRACE),
        lambda: replace(datum, observation_layer="spin_yield"),  # type: ignore[arg-type]
        lambda: replace(datum, evidence_origin="simulation"),  # type: ignore[arg-type]
        lambda: replace(datum, membership="training"),  # type: ignore[arg-type]
        lambda: replace(datum, replication_status="replicated"),  # type: ignore[arg-type]
        lambda: replace(datum, uncertainty_status="known"),  # type: ignore[arg-type]
        lambda: replace(datum, signal=datum.signal[:-1]),
        lambda: replace(datum, time_seconds=list(datum.time_seconds)),  # type: ignore[arg-type]
        lambda: replace(datum, signal=list(datum.signal)),  # type: ignore[arg-type]
        lambda: replace(datum, reported_fields=list(datum.reported_fields)),  # type: ignore[arg-type]
        lambda: replace(datum, conditions=list(datum.conditions)),  # type: ignore[arg-type]
        lambda: replace(datum, limitations=list(datum.limitations)),  # type: ignore[arg-type]
        lambda: replace(datum, frame_index=(0, 1, 2, 3)),
        lambda: replace(datum, signal=(10.5, float("nan"), 12.5, 13.5)),
        lambda: replace(datum, time_seconds=(0.0, 4.0, 4.0, 14.0)),
        lambda: replace(datum, time_seconds=(1.0, 4.0, 9.0, 14.0)),
        lambda: replace(datum, timestamp_basis="nominal"),  # type: ignore[arg-type]
        lambda: replace(datum, schedule=wrong_hash_schedule),
        lambda: replace(datum, schedule=wrong_states_schedule),
        lambda: replace(datum, reported_fields=datum.reported_fields[:1]),
        lambda: replace(
            datum,
            reported_fields=(datum.reported_fields[0], datum.reported_fields[0]),
        ),
        lambda: replace(
            datum,
            reported_fields=(datum.reported_fields[0], "field_off"),  # type: ignore[arg-type]
        ),
        lambda: replace(
            datum,
            reported_fields=(
                datum.reported_fields[0],
                replace(datum.reported_fields[1], approximate_magnitude_tesla=0.0),
            ),
        ),
        lambda: replace(datum, conditions=()),
        lambda: replace(datum, conditions=(datum.conditions[0], datum.conditions[0])),
        lambda: replace(datum, limitations=()),
    )
    for invalid in invalid_data:
        with pytest.raises(ValueError):
            invalid()

    assert datum.observation_layer is ObservationLayer.FLUORESCENCE_READOUT
    assert datum.evidence_origin is MeasurementEvidenceOrigin.MEASUREMENT
    assert datum.replication_status is ReplicationStatus.NO_DEPOSITED_INDEPENDENT_REPLICATE_EVIDENCE
    assert (
        datum.uncertainty_status is MeasurementUncertaintyStatus.NO_REPORTED_POINTWISE_UNCERTAINTY
    )
    assert datum.schedule.basis is ScheduleBasis.DECLARED_UNSYNCHRONIZED_SCRIPT_SCHEDULE
    assert datum.reported_fields[0].status is ReportedFieldStatus.UNCALIBRATED_APPROXIMATE_REPORT


def test_source_reported_condition_requires_typed_finite_values() -> None:
    condition = SourceReportedCondition(
        "irradiance",
        500.0,
        "W/m^2",
        SourceReportStatus.APPROXIMATE_SOURCE_REPORT,
        "index.html",
    )
    assert condition.value == 500.0
    invalid_conditions: tuple[Callable[[], object], ...] = (
        lambda: replace(condition, status="approximate"),  # type: ignore[arg-type]
        lambda: replace(condition, value=float("nan")),
        lambda: replace(condition, value=" "),
        lambda: replace(condition, value=object()),  # type: ignore[arg-type]
    )
    for invalid in invalid_conditions:
        with pytest.raises(ValueError):
            invalid()
