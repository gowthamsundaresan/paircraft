"""Declared transformations from raw magnetic-control traces to MFE summaries."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from typing import Final

import numpy as np
from scipy.interpolate import make_interp_spline

from paircraft import __version__
from paircraft.experiments.model import ExperimentalTrace, TraceClass

_ALGORITHM_IDENTIFIER: Final = "paircraft.experiments.analyze_folded_mfe"
_ALGORITHM_VERSION: Final = __version__


@dataclass(frozen=True, slots=True)
class MfeAnalysisConfig:
    """Complete configuration for folded high-field MFE analysis."""

    frames_per_state: int
    discard_periods: int = 1
    steady_state_frames: int = 10
    spline_degree: int = 3

    def __post_init__(self) -> None:
        if self.frames_per_state <= 0:
            raise ValueError("frames_per_state must be positive")
        if self.discard_periods < 0:
            raise ValueError("discard_periods must be nonnegative")
        if not 1 <= self.steady_state_frames <= self.frames_per_state:
            raise ValueError("steady_state_frames must fit within one field state")
        if not 1 <= self.spline_degree <= 3:
            raise ValueError("spline_degree must lie in [1, 3]")


@dataclass(frozen=True, slots=True)
class MfeAnalysisResult:
    """A calculated folded trace retaining its exact raw input and configuration."""

    input_trace: ExperimentalTrace
    config: MfeAnalysisConfig
    folded_relative_signal: tuple[float, ...]
    field_off_mean: float
    field_on_mean: float
    relative_effect: float
    sign_convention: str = "(I_on - I_off) / I_off"
    evidence_status: str = "calculated"
    uncertainty_status: str = "not_calculated"
    uncertainty_note: str = (
        "No replicate hierarchy or joint measurement-error model is declared for this scalar trace."
    )
    algorithm_identifier: str = _ALGORITHM_IDENTIFIER
    algorithm_version: str = _ALGORITHM_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.folded_relative_signal, tuple):
            raise ValueError("folded signal must be an immutable tuple")
        expected = self.config.frames_per_state * 2
        if len(self.folded_relative_signal) != expected:
            raise ValueError("folded signal length must equal one complete OFF/ON period")
        if any(
            not isfinite(value)
            for value in (
                *self.folded_relative_signal,
                self.field_off_mean,
                self.field_on_mean,
                self.relative_effect,
            )
        ):
            raise ValueError("MFE analysis outputs must be finite")
        if self.sign_convention != "(I_on - I_off) / I_off":
            raise ValueError("MFE result must retain the executed sign convention")
        if self.evidence_status != "calculated":
            raise ValueError("MFE result evidence status must be calculated")
        if self.uncertainty_status != "not_calculated" or not self.uncertainty_note.strip():
            raise ValueError("MFE result must state that uncertainty was not calculated and why")
        if (
            self.algorithm_identifier != _ALGORITHM_IDENTIFIER
            or self.algorithm_version != _ALGORITHM_VERSION
        ):
            raise ValueError("MFE result must retain the executing algorithm identity and version")
        folded = np.asarray(self.folded_relative_signal, dtype=np.float64)
        off_stop = self.config.frames_per_state
        off_start = off_stop - self.config.steady_state_frames
        on_start = len(folded) - self.config.steady_state_frames
        expected_off = float(folded[off_start:off_stop].mean())
        expected_on = float(folded[on_start:].mean())
        if expected_off == 0.0:
            raise ValueError("MFE result field-OFF mean must be nonzero")
        expected_effect = (expected_on - expected_off) / expected_off
        if not (
            isclose(self.field_off_mean, expected_off, rel_tol=1e-12, abs_tol=0.0)
            and isclose(self.field_on_mean, expected_on, rel_tol=1e-12, abs_tol=0.0)
            and isclose(self.relative_effect, expected_effect, rel_tol=1e-12, abs_tol=0.0)
        ):
            raise ValueError("MFE result summary contradicts its folded trace")

    @property
    def relative_effect_percent(self) -> float:
        """Return the signed MFE in percentage points."""

        return 100.0 * self.relative_effect


def analyze_folded_mfe(trace: ExperimentalTrace, config: MfeAnalysisConfig) -> MfeAnalysisResult:
    """Detrend repeated OFF/ON periods and calculate signed steady-state MFE.

    A spline through the final sample of each OFF block estimates the photobleaching baseline, as
    in the deposited Oxford figure workflow. The first configured periods are discarded before
    fitting. No coil-current-to-field conversion is performed.
    """

    if trace.trace_class is not TraceClass.INSTRUMENT_EXPORT_RAW_TRACE:
        raise ValueError("folded MFE analysis requires an instrument-export raw trace")
    period = config.frames_per_state * 2
    if len(trace.signal) % period:
        raise ValueError("trace length must contain complete OFF/ON periods")
    period_count = len(trace.signal) // period
    retained_periods = period_count - config.discard_periods
    if retained_periods < config.spline_degree + 1:
        raise ValueError("not enough retained periods for the configured baseline spline")

    current = np.asarray(trace.magnet_current_ampere, dtype=np.float64).reshape(
        period_count, period
    )
    off_current = current[:, : config.frames_per_state]
    on_current = current[:, config.frames_per_state :]
    if not np.all(off_current == 0.0):
        raise ValueError("each period must begin with a zero-current OFF state")
    if np.any(on_current == 0.0) or not np.all(on_current == on_current[0, 0]):
        raise ValueError("all periods must share one constant nonzero-current ON state")

    start = config.discard_periods * period
    signal = np.asarray(trace.signal[start:], dtype=np.float64)
    sample_index = np.arange(signal.size, dtype=np.float64)
    anchors = np.arange(config.frames_per_state - 1, signal.size, period)
    baseline = make_interp_spline(anchors, signal[anchors], k=config.spline_degree)(sample_index)
    if not np.isfinite(baseline).all() or np.any(baseline == 0.0):
        raise ValueError("photobleaching baseline must be finite and nonzero")
    normalized = signal / baseline
    folded = normalized.reshape(retained_periods, period).mean(axis=0)

    off_stop = config.frames_per_state
    off_start = off_stop - config.steady_state_frames
    on_stop = period
    on_start = on_stop - config.steady_state_frames
    field_off_mean = float(folded[off_start:off_stop].mean())
    field_on_mean = float(folded[on_start:on_stop].mean())
    if field_off_mean == 0.0:
        raise ValueError("field-OFF reference mean must be nonzero")
    relative_effect = (field_on_mean - field_off_mean) / field_off_mean
    return MfeAnalysisResult(
        input_trace=trace,
        config=config,
        folded_relative_signal=tuple(float(value) for value in folded),
        field_off_mean=field_off_mean,
        field_on_mean=field_on_mean,
        relative_effect=relative_effect,
    )
