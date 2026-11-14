"""Immutable experimental traces with explicit provenance and units."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from string import hexdigits


def _require_text(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be explicit")


class TraceClass(StrEnum):
    """The processing level of an experimental trace."""

    INSTRUMENT_EXPORT_RAW_TRACE = "instrument_export_raw_trace"
    ROI_DERIVED_TRACE = "roi_derived_trace"
    BACKGROUND_CORRECTED_TRACE = "background_corrected_trace"
    NORMALIZED_FITTED_RESULT = "normalized_fitted_result"


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    """Identity and content hash of one exact source payload."""

    source_id: str
    source_url: str
    source_version: str
    exact_location: str
    sha256: str
    license: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.source_id, "source ID"),
            (self.source_url, "source URL"),
            (self.source_version, "source version"),
            (self.exact_location, "exact source location"),
            (self.license, "source license"),
        ):
            _require_text(value, name)
        digest = self.sha256.lower()
        if len(digest) != 64 or any(character not in hexdigits for character in digest):
            raise ValueError("source sha256 must be a 64-character hexadecimal digest")


@dataclass(frozen=True, slots=True)
class ExperimentalTrace:
    """One scalar observation trace and its magnetic-control schedule.

    ``magnet_current_ampere`` is an instrument control/readback quantity. It is deliberately not
    represented as magnetic flux density B and must not be converted to tesla without a separately
    sourced calibration.
    """

    trace_id: str
    sample_label: str
    construct_id: str
    trace_class: TraceClass
    observable: str
    signal_unit: str
    time_seconds: tuple[float, ...]
    signal: tuple[float, ...]
    magnet_current_ampere: tuple[float, ...]
    attributes: tuple[tuple[str, str], ...]
    conditions: str
    provenance: tuple[SourceProvenance, ...]

    def __post_init__(self) -> None:
        for value, name in (
            (self.trace_id, "trace ID"),
            (self.sample_label, "sample label"),
            (self.construct_id, "construct ID"),
            (self.observable, "observable"),
            (self.signal_unit, "signal unit"),
            (self.conditions, "conditions"),
        ):
            _require_text(value, name)
        tuple_fields: tuple[tuple[object, str], ...] = (
            (self.time_seconds, "time"),
            (self.signal, "signal"),
            (self.magnet_current_ampere, "magnet current"),
            (self.attributes, "attributes"),
            (self.provenance, "provenance"),
        )
        for tuple_value, tuple_name in tuple_fields:
            if not isinstance(tuple_value, tuple):
                raise ValueError(f"{tuple_name} must be an immutable tuple")
        if len(self.time_seconds) < 2:
            raise ValueError("an experimental trace requires at least two observations")
        if not (len(self.time_seconds) == len(self.signal) == len(self.magnet_current_ampere)):
            raise ValueError("time, signal, and magnet-current arrays must have equal length")
        if any(not isfinite(value) for value in self.time_seconds):
            raise ValueError("time values must be finite")
        if any(
            right <= left
            for left, right in zip(self.time_seconds, self.time_seconds[1:], strict=False)
        ):
            raise ValueError("time values must be strictly increasing")
        if any(not isfinite(value) for value in (*self.signal, *self.magnet_current_ampere)):
            raise ValueError("signal and magnet-current values must be finite")
        if not self.provenance:
            raise ValueError("experimental trace must retain at least one source")
        attribute_keys: list[str] = []
        for attribute in self.attributes:
            if not isinstance(attribute, tuple) or len(attribute) != 2:
                raise ValueError("each attribute must be an immutable key/value pair")
            key, value = attribute
            _require_text(key, "attribute key")
            _require_text(value, "attribute value")
            attribute_keys.append(key)
        if len(attribute_keys) != len(set(attribute_keys)):
            raise ValueError("attribute keys must be unique")
