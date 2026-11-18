"""Canonical, non-training records for processed experimental observations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256
from itertools import pairwise
from math import isfinite
from string import hexdigits
from typing import Any, cast

from paircraft.experiments.model import TraceClass

_SCHEMA_VERSION = "paircraft-measured-datum-v1"


def _require_text(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be explicit")


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in hexdigits for character in value):
        raise ValueError(f"{name} must be a 64-character hexadecimal digest")


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return sha256(encoded).hexdigest()


class ObservationLayer(StrEnum):
    """Physical layer represented by the observations."""

    FLUORESCENCE_READOUT = "fluorescence_readout"


class MagnetControlState(StrEnum):
    """Source-declared electromagnet state, not a measured field value."""

    FIELD_ON = "field_on"
    FIELD_OFF = "field_off"


class TimestampBasis(StrEnum):
    """Origin of observation times."""

    EMBEDDED_FRAME_TIMESTAMP_REBASED_TO_FIRST_FRAME = (
        "embedded_frame_timestamp_rebased_to_first_frame"
    )


class ScheduleBasis(StrEnum):
    """Origin of magnetic-control annotations."""

    DECLARED_UNSYNCHRONIZED_SCRIPT_SCHEDULE = "declared_unsynchronized_script_schedule"


class ReplicationStatus(StrEnum):
    """Independent-replication evidence available for a datum."""

    NO_DEPOSITED_INDEPENDENT_REPLICATE_EVIDENCE = "no_deposited_independent_replicate_evidence"


class PublicationStatus(StrEnum):
    """Review status of the publication supporting a source bundle."""

    PREPRINT_NON_PEER_REVIEWED = "preprint_non_peer_reviewed"


class SourceReportStatus(StrEnum):
    """Epistemic status of a condition transcribed from a source."""

    SOURCE_REPORTED = "source_reported"
    APPROXIMATE_SOURCE_REPORT = "approximate_source_report"


class MeasurementUncertaintyStatus(StrEnum):
    """Pointwise measurement uncertainty available for a datum."""

    NO_REPORTED_POINTWISE_UNCERTAINTY = "no_reported_pointwise_uncertainty"


class ReportedFieldStatus(StrEnum):
    """Epistemic status of a source-reported field scalar."""

    UNCALIBRATED_APPROXIMATE_REPORT = "uncalibrated_approximate_report"


class MeasurementEvidenceOrigin(StrEnum):
    """Origin of evidence carried by a measured datum."""

    MEASUREMENT = "measurement"


class MeasurementCorpusMembership(StrEnum):
    """Current corpus role; training assignment requires a separate manifest."""

    REFERENCE_ONLY_NOT_TRAINING = "reference_only_not_training"


class SourceMemberRole(StrEnum):
    """Required role of an exact source-archive member."""

    SIGNAL_CSV = "signal_csv"
    TIMESTAMP_TIFF = "timestamp_tiff"
    SCHEDULE_SCRIPT = "schedule_script"
    CONTEXT_HTML = "context_html"
    APPENDIX_HTML = "appendix_html"
    LICENSE = "license"


@dataclass(frozen=True, slots=True)
class SourceMember:
    """Identity and hash of one exact archive member."""

    role: SourceMemberRole
    exact_location: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, SourceMemberRole):
            raise ValueError("source member role must be a SourceMemberRole")
        _require_text(self.exact_location, "source member location")
        _require_sha256(self.sha256, "source member sha256")


@dataclass(frozen=True, slots=True)
class SourceBundle:
    """Pinned, attributable source archive and its relevant members."""

    source_id: str
    source_url: str
    source_version: str
    archive_sha256: str
    members: tuple[SourceMember, ...]
    license: str
    attribution: str
    publication_status: PublicationStatus

    def __post_init__(self) -> None:
        for value, name in (
            (self.source_id, "source ID"),
            (self.source_url, "source URL"),
            (self.source_version, "source version"),
            (self.license, "source license"),
            (self.attribution, "source attribution"),
        ):
            _require_text(value, name)
        _require_sha256(self.archive_sha256, "archive sha256")
        if self.publication_status is not PublicationStatus.PREPRINT_NON_PEER_REVIEWED:
            raise ValueError("publication status must preserve the non-peer-reviewed preprint")
        if not isinstance(self.members, tuple):
            raise ValueError("source members must be an immutable tuple")
        roles = [member.role for member in self.members]
        if len(roles) != len(set(roles)) or set(roles) != set(SourceMemberRole):
            raise ValueError("source bundle requires exactly one member for every required role")

    def member(self, role: SourceMemberRole) -> SourceMember:
        """Return the unique member having ``role``."""

        return next(member for member in self.members if member.role is role)


@dataclass(frozen=True, slots=True)
class ReportedFieldCondition:
    """Approximate source report, deliberately not a calibrated or measured B vector."""

    control_state: MagnetControlState
    reported_text: str
    approximate_magnitude_tesla: float | None
    exact_source_location: str
    status: ReportedFieldStatus = ReportedFieldStatus.UNCALIBRATED_APPROXIMATE_REPORT

    def __post_init__(self) -> None:
        if not isinstance(self.control_state, MagnetControlState):
            raise ValueError("reported field control state must be a MagnetControlState")
        if self.status is not ReportedFieldStatus.UNCALIBRATED_APPROXIMATE_REPORT:
            raise ValueError("reported field must remain an uncalibrated approximate report")
        _require_text(self.reported_text, "reported field text")
        _require_text(self.exact_source_location, "reported field source location")
        value = self.approximate_magnitude_tesla
        if value is not None and (not isfinite(value) or value < 0):
            raise ValueError("approximate field magnitude must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class SourceReportedCondition:
    """One individually structured condition transcribed from an exact source location."""

    name: str
    value: float | str
    unit: str
    status: SourceReportStatus
    exact_source_location: str

    def __post_init__(self) -> None:
        _require_text(self.name, "source-reported condition name")
        _require_text(self.unit, "source-reported condition unit")
        _require_text(self.exact_source_location, "source-reported condition location")
        if not isinstance(self.status, SourceReportStatus):
            raise ValueError("source-reported condition status must be explicit")
        if isinstance(self.value, float):
            if not isfinite(self.value):
                raise ValueError("source-reported numeric condition must be finite")
        elif isinstance(self.value, str):
            _require_text(self.value, "source-reported condition value")
        else:
            raise ValueError("source-reported condition value must be numeric or text")


@dataclass(frozen=True, slots=True)
class DeclaredMagnetSchedule:
    """Script-declared field-control annotations aligned to frame timestamps."""

    initial_state: MagnetControlState
    field_off_transition_seconds: tuple[float, ...]
    field_on_transition_seconds: tuple[float, ...]
    state_by_observation: tuple[MagnetControlState, ...]
    source_member_sha256: str
    basis: ScheduleBasis = ScheduleBasis.DECLARED_UNSYNCHRONIZED_SCRIPT_SCHEDULE

    def __post_init__(self) -> None:
        if self.initial_state is not MagnetControlState.FIELD_ON:
            raise ValueError("this schedule contract requires the field initially on")
        if self.basis is not ScheduleBasis.DECLARED_UNSYNCHRONIZED_SCRIPT_SCHEDULE:
            raise ValueError("schedule basis must remain the declared unsynchronized script")
        _require_sha256(self.source_member_sha256, "schedule source member sha256")
        for transitions, name in (
            (self.field_off_transition_seconds, "field-off transitions"),
            (self.field_on_transition_seconds, "field-on transitions"),
        ):
            if not isinstance(transitions, tuple):
                raise ValueError(f"{name} must be an immutable tuple")
            if any(not isfinite(value) or value < 0 for value in transitions):
                raise ValueError(f"{name} must be finite and nonnegative")
            if any(right <= left for left, right in pairwise(transitions)):
                raise ValueError(f"{name} must be strictly increasing")
        if not self.field_off_transition_seconds or not self.field_on_transition_seconds:
            raise ValueError("schedule requires field-on and field-off transitions")
        combined = sorted(
            [(value, MagnetControlState.FIELD_OFF) for value in self.field_off_transition_seconds]
            + [(value, MagnetControlState.FIELD_ON) for value in self.field_on_transition_seconds]
        )
        expected = MagnetControlState.FIELD_OFF
        for _, state in combined:
            if state is not expected:
                raise ValueError("field-on and field-off transitions must alternate")
            expected = (
                MagnetControlState.FIELD_ON
                if expected is MagnetControlState.FIELD_OFF
                else MagnetControlState.FIELD_OFF
            )
        if not isinstance(self.state_by_observation, tuple) or not self.state_by_observation:
            raise ValueError("schedule states must be a nonempty immutable tuple")
        if any(not isinstance(state, MagnetControlState) for state in self.state_by_observation):
            raise ValueError("schedule states must be MagnetControlState values")


@dataclass(frozen=True, slots=True)
class MeasuredDatum:
    """One processed fluorescence trace with explicit evidentiary limits."""

    trace_id: str
    sample_label: str
    construct_id: str
    frame_index: tuple[int, ...]
    time_seconds: tuple[float, ...]
    signal: tuple[float, ...]
    signal_unit: str
    schedule: DeclaredMagnetSchedule
    reported_fields: tuple[ReportedFieldCondition, ...]
    conditions: tuple[SourceReportedCondition, ...]
    provenance: SourceBundle
    limitations: tuple[str, ...]
    observable: str = "mean_fluorescence_intensity"
    trace_class: TraceClass = TraceClass.DEPOSITED_FIGURE_INPUT_MEAN_INTENSITY_TRACE
    observation_layer: ObservationLayer = ObservationLayer.FLUORESCENCE_READOUT
    timestamp_basis: TimestampBasis = TimestampBasis.EMBEDDED_FRAME_TIMESTAMP_REBASED_TO_FIRST_FRAME
    replication_status: ReplicationStatus = (
        ReplicationStatus.NO_DEPOSITED_INDEPENDENT_REPLICATE_EVIDENCE
    )
    uncertainty_status: MeasurementUncertaintyStatus = (
        MeasurementUncertaintyStatus.NO_REPORTED_POINTWISE_UNCERTAINTY
    )
    schema_version: str = _SCHEMA_VERSION
    evidence_origin: MeasurementEvidenceOrigin = MeasurementEvidenceOrigin.MEASUREMENT
    membership: MeasurementCorpusMembership = (
        MeasurementCorpusMembership.REFERENCE_ONLY_NOT_TRAINING
    )
    publication_status: PublicationStatus = PublicationStatus.PREPRINT_NON_PEER_REVIEWED

    def __post_init__(self) -> None:
        for value, name in (
            (self.trace_id, "trace ID"),
            (self.sample_label, "sample label"),
            (self.construct_id, "construct ID"),
            (self.signal_unit, "signal unit"),
            (self.observable, "observable"),
        ):
            _require_text(value, name)
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("measured datum schema version is unsupported")
        if self.trace_class is not TraceClass.DEPOSITED_FIGURE_INPUT_MEAN_INTENSITY_TRACE:
            raise ValueError("measured datum must remain a deposited figure-input mean trace")
        if self.observation_layer is not ObservationLayer.FLUORESCENCE_READOUT:
            raise ValueError("measured datum must remain a fluorescence readout")
        if self.evidence_origin is not MeasurementEvidenceOrigin.MEASUREMENT:
            raise ValueError("measured datum evidence origin must be measurement")
        if self.membership is not MeasurementCorpusMembership.REFERENCE_ONLY_NOT_TRAINING:
            raise ValueError("measured datum membership must remain reference-only")
        if (
            self.replication_status
            is not ReplicationStatus.NO_DEPOSITED_INDEPENDENT_REPLICATE_EVIDENCE
        ):
            raise ValueError("this datum has no deposited independent replicate evidence")
        if self.publication_status is not self.provenance.publication_status:
            raise ValueError("datum publication status must match source provenance")
        if (
            self.uncertainty_status
            is not MeasurementUncertaintyStatus.NO_REPORTED_POINTWISE_UNCERTAINTY
        ):
            raise ValueError("this datum has no reported pointwise uncertainty")
        for tuple_value, name in (
            (self.frame_index, "frame indices"),
            (self.time_seconds, "times"),
            (self.signal, "signals"),
            (self.reported_fields, "reported fields"),
            (self.conditions, "conditions"),
            (self.limitations, "limitations"),
        ):
            if not isinstance(tuple_value, tuple):
                raise ValueError(f"{name} must be an immutable tuple")
        lengths = {
            len(self.frame_index),
            len(self.time_seconds),
            len(self.signal),
            len(self.schedule.state_by_observation),
        }
        if len(lengths) != 1 or next(iter(lengths)) < 2:
            raise ValueError("frame, time, signal, and schedule arrays must have equal length")
        if self.frame_index != tuple(range(1, len(self.frame_index) + 1)):
            raise ValueError("frame indices must be sequential and one-based")
        if any(not isfinite(value) for value in self.time_seconds):
            raise ValueError("times must be finite")
        if any(not isfinite(value) or value < 0 for value in self.signal):
            raise ValueError("signals must be finite and nonnegative")
        if any(right <= left for left, right in pairwise(self.time_seconds)):
            raise ValueError("times must be strictly increasing")
        if self.time_seconds[0] != 0.0:
            raise ValueError("rebased frame timestamps must begin at zero")
        if (
            self.timestamp_basis
            is not TimestampBasis.EMBEDDED_FRAME_TIMESTAMP_REBASED_TO_FIRST_FRAME
        ):
            raise ValueError("timestamp basis must preserve embedded rebased frame timestamps")
        if (
            self.schedule.source_member_sha256
            != self.provenance.member(SourceMemberRole.SCHEDULE_SCRIPT).sha256
        ):
            raise ValueError("schedule provenance must match the source script member")
        expected_states = _states_at_times(
            self.time_seconds,
            self.schedule.field_off_transition_seconds,
            self.schedule.field_on_transition_seconds,
        )
        if self.schedule.state_by_observation != expected_states:
            raise ValueError("schedule states contradict declared transitions")
        if len(self.reported_fields) != 2 or any(
            not isinstance(field, ReportedFieldCondition) for field in self.reported_fields
        ):
            raise ValueError("reported fields must be an immutable ON/OFF pair")
        field_states = [field.control_state for field in self.reported_fields]
        if len(set(field_states)) != 2 or set(field_states) != set(MagnetControlState):
            raise ValueError("reported fields require exactly one condition for each control state")
        off_field = next(
            field
            for field in self.reported_fields
            if field.control_state is MagnetControlState.FIELD_OFF
        )
        if off_field.approximate_magnitude_tesla is not None:
            raise ValueError("the York OFF-state field magnitude is unreported")
        if not self.conditions or any(
            not isinstance(condition, SourceReportedCondition) for condition in self.conditions
        ):
            raise ValueError("measurement conditions must be individually structured")
        condition_names = [condition.name for condition in self.conditions]
        if len(condition_names) != len(set(condition_names)):
            raise ValueError("measurement condition names must be unique")
        if not self.limitations or any(not value.strip() for value in self.limitations):
            raise ValueError("measurement limitations must be explicit")

    @property
    def observation_sha256(self) -> str:
        """Hash the source signal and its observation semantics."""

        return _canonical_sha256(
            {
                "frame_index": self.frame_index,
                "time_seconds": self.time_seconds,
                "signal": self.signal,
                "signal_unit": self.signal_unit,
                "observable": self.observable,
                "trace_class": self.trace_class,
                "observation_layer": self.observation_layer,
            }
        )

    @property
    def source_group_sha256(self) -> str:
        """Hash the exact source version and archive members."""

        return _canonical_sha256(asdict(self.provenance))

    @property
    def dependency_group_sha256(self) -> str:
        """Group derived views of this construct and acquisition."""

        return _canonical_sha256(
            {
                "construct_id": self.construct_id,
                "source_id": self.provenance.source_id,
                "source_version": self.provenance.source_version,
                "archive_sha256": self.provenance.archive_sha256,
                "members": [asdict(member) for member in self.provenance.members],
            }
        )

    @property
    def datum_sha256(self) -> str:
        """Hash the complete measured-datum claim."""

        payload = asdict(self)
        payload["observation_sha256"] = self.observation_sha256
        payload["source_group_sha256"] = self.source_group_sha256
        payload["dependency_group_sha256"] = self.dependency_group_sha256
        return _canonical_sha256(payload)


def _states_at_times(
    time_seconds: tuple[float, ...],
    field_off_transitions: tuple[float, ...],
    field_on_transitions: tuple[float, ...],
) -> tuple[MagnetControlState, ...]:
    transitions = sorted(
        [(value, MagnetControlState.FIELD_OFF) for value in field_off_transitions]
        + [(value, MagnetControlState.FIELD_ON) for value in field_on_transitions]
    )
    states: list[MagnetControlState] = []
    index = 0
    state = MagnetControlState.FIELD_ON
    for time in time_seconds:
        while index < len(transitions) and transitions[index][0] <= time:
            state = transitions[index][1]
            index += 1
        states.append(state)
    return tuple(states)


def declared_schedule_states(
    time_seconds: tuple[float, ...],
    field_off_transition_seconds: tuple[float, ...],
    field_on_transition_seconds: tuple[float, ...],
) -> tuple[MagnetControlState, ...]:
    """Align a pinned declared schedule to measured frame timestamps."""

    return _states_at_times(time_seconds, field_off_transition_seconds, field_on_transition_seconds)


def measured_datum_document(datum: MeasuredDatum) -> dict[str, object]:
    """Return a complete JSON-compatible document with derived hashes."""

    document: dict[str, Any] = asdict(datum)
    document["hashes"] = {
        "observation_sha256": datum.observation_sha256,
        "source_group_sha256": datum.source_group_sha256,
        "dependency_group_sha256": datum.dependency_group_sha256,
        "datum_sha256": datum.datum_sha256,
    }
    normalized = json.loads(
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    if not isinstance(normalized, dict):
        raise AssertionError("measured datum normalization did not produce an object")
    return cast(dict[str, object], normalized)


def measured_datum_json(datum: MeasuredDatum) -> bytes:
    """Return canonical compact JSON bytes; no persistence policy is implied."""

    return json.dumps(
        measured_datum_document(datum),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
