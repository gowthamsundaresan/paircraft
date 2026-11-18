from collections.abc import Callable
from dataclasses import replace
from math import nan
from typing import Any, cast

import pytest

from paircraft.experiments import (
    ExperimentalTrace,
    MfeAnalysisConfig,
    SourceProvenance,
    TraceClass,
    analyze_folded_mfe,
)


def _source() -> SourceProvenance:
    return SourceProvenance(
        source_id="synthetic-fixed-answer",
        source_url="https://example.test/synthetic",
        source_version="v1",
        exact_location="generated in test",
        sha256="a" * 64,
        license="test fixture",
    )


def _trace() -> ExperimentalTrace:
    periods = 6
    frames_per_state = 4
    one_period_signal = (10.0,) * frames_per_state + (8.0,) * frames_per_state
    one_period_current = (0.0,) * frames_per_state + (0.11,) * frames_per_state
    signal = one_period_signal * periods
    return ExperimentalTrace(
        trace_id="synthetic-mfe",
        sample_label="synthetic sample",
        construct_id="synthetic construct",
        trace_class=TraceClass.INSTRUMENT_EXPORT_RAW_TRACE,
        observable="fluorescence intensity",
        signal_unit="camera_count",
        time_seconds=tuple(index * 0.25 for index in range(len(signal))),
        signal=signal,
        magnet_current_ampere=one_period_current * periods,
        attributes=(("fixture", "fixed-answer"),),
        conditions="constant baseline with an exact negative 20 percent ON response",
        provenance=(_source(),),
    )


def test_folded_mfe_recovers_signed_fixed_answer_and_retains_input() -> None:
    trace = _trace()
    config = MfeAnalysisConfig(frames_per_state=4, steady_state_frames=2)

    result = analyze_folded_mfe(trace, config)

    assert result.input_trace is trace
    assert result.config is config
    assert result.sign_convention == "(I_on - I_off) / I_off"
    assert result.evidence_status == "calculated"
    assert result.uncertainty_status == "not_calculated"
    assert result.algorithm_identifier == "paircraft.experiments.analyze_folded_mfe"
    assert result.algorithm_version == "0.1.0"
    assert result.field_off_mean == pytest.approx(1.0, abs=1e-15)
    assert result.field_on_mean == pytest.approx(0.8, abs=1e-15)
    assert result.relative_effect == pytest.approx(-0.2, abs=1e-15)
    assert result.relative_effect_percent == pytest.approx(-20.0, abs=1e-13)
    with pytest.raises(ValueError, match="summary contradicts"):
        replace(result, relative_effect=0.2)


def test_folded_mfe_rejects_reversed_off_on_schedule() -> None:
    trace = _trace()
    reversed_current = (0.11,) * 4 + (0.0,) * 4
    trace = replace(trace, magnet_current_ampere=reversed_current * 6)

    with pytest.raises(ValueError, match="begin with a zero-current OFF state"):
        analyze_folded_mfe(trace, MfeAnalysisConfig(frames_per_state=4, steady_state_frames=2))


def test_folded_mfe_refuses_derived_trace() -> None:
    trace = replace(_trace(), trace_class=TraceClass.ROI_DERIVED_TRACE)

    with pytest.raises(ValueError, match="instrument-export raw trace"):
        analyze_folded_mfe(trace, MfeAnalysisConfig(frames_per_state=4, steady_state_frames=2))


def test_experimental_contract_rejects_invalid_identity_shape_and_values() -> None:
    trace = _trace()
    source = _source()
    with pytest.raises(ValueError, match="source ID must be explicit"):
        replace(source, source_id=" ")
    with pytest.raises(ValueError, match="source sha256"):
        replace(source, sha256="bad")
    with pytest.raises(ValueError, match="time must be an immutable tuple"):
        replace(trace, time_seconds=cast(Any, list(trace.time_seconds)))
    with pytest.raises(ValueError, match="at least two observations"):
        replace(trace, time_seconds=(0.0,), signal=(1.0,), magnet_current_ampere=(0.0,))
    with pytest.raises(ValueError, match="equal length"):
        replace(trace, signal=trace.signal[:-1])
    with pytest.raises(ValueError, match="time values must be finite"):
        replace(trace, time_seconds=(nan, *trace.time_seconds[1:]))
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(trace, time_seconds=(0.0,) * len(trace.time_seconds))
    with pytest.raises(ValueError, match="signal and magnet-current values must be finite"):
        replace(trace, signal=(nan, *trace.signal[1:]))
    with pytest.raises(ValueError, match="at least one source"):
        replace(trace, provenance=())
    with pytest.raises(ValueError, match="immutable key/value pair"):
        replace(trace, attributes=cast(Any, (("key",),)))
    with pytest.raises(ValueError, match="attribute key must be explicit"):
        replace(trace, attributes=(("", "value"),))
    with pytest.raises(ValueError, match="attribute keys must be unique"):
        replace(trace, attributes=(("key", "one"), ("key", "two")))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frames_per_state": 0}, "frames_per_state"),
        ({"frames_per_state": 4, "discard_periods": -1}, "discard_periods"),
        ({"frames_per_state": 4, "steady_state_frames": 5}, "steady_state_frames"),
        ({"frames_per_state": 4, "steady_state_frames": 2, "spline_degree": 4}, "spline_degree"),
    ],
)
def test_mfe_analysis_config_rejects_invalid_windows(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MfeAnalysisConfig(**kwargs)


def test_folded_mfe_rejects_incomplete_or_unusable_periods() -> None:
    trace = _trace()
    config = MfeAnalysisConfig(frames_per_state=4, steady_state_frames=2)
    with pytest.raises(ValueError, match="complete OFF/ON periods"):
        analyze_folded_mfe(
            replace(
                trace,
                time_seconds=trace.time_seconds[:-1],
                signal=trace.signal[:-1],
                magnet_current_ampere=trace.magnet_current_ampere[:-1],
            ),
            config,
        )
    with pytest.raises(ValueError, match="not enough retained periods"):
        analyze_folded_mfe(
            replace(
                trace,
                time_seconds=trace.time_seconds[:24],
                signal=trace.signal[:24],
                magnet_current_ampere=trace.magnet_current_ampere[:24],
            ),
            config,
        )
    variable_on = (0.0,) * 4 + (0.11, 0.12, 0.11, 0.12)
    with pytest.raises(ValueError, match="constant nonzero-current ON state"):
        analyze_folded_mfe(replace(trace, magnet_current_ampere=variable_on * 6), config)
    inconsistent_periods = ((0.0,) * 4 + (0.11,) * 4) * 5 + (0.0,) * 4 + (0.12,) * 4
    with pytest.raises(ValueError, match="constant nonzero-current ON state"):
        analyze_folded_mfe(replace(trace, magnet_current_ampere=inconsistent_periods), config)
    with pytest.raises(ValueError, match="baseline must be finite and nonzero"):
        analyze_folded_mfe(replace(trace, signal=(0.0,) * len(trace.signal)), config)


def test_mfe_result_rejects_shape_and_epistemic_mutations() -> None:
    result = analyze_folded_mfe(
        _trace(),
        MfeAnalysisConfig(frames_per_state=4, steady_state_frames=2),
    )
    invalid_results: tuple[tuple[Callable[[], object], str], ...] = (
        (
            lambda: replace(
                result,
                folded_relative_signal=cast(Any, list(result.folded_relative_signal)),
            ),
            "immutable",
        ),
        (
            lambda: replace(result, folded_relative_signal=result.folded_relative_signal[:-1]),
            "length",
        ),
        (
            lambda: replace(
                result,
                folded_relative_signal=(nan, *result.folded_relative_signal[1:]),
            ),
            "finite",
        ),
        (lambda: replace(result, sign_convention="(I_off-I_on)/I_off"), "sign convention"),
        (lambda: replace(result, evidence_status="measured"), "calculated"),
        (lambda: replace(result, uncertainty_status="reported"), "uncertainty"),
        (lambda: replace(result, uncertainty_note=" "), "uncertainty"),
        (lambda: replace(result, algorithm_identifier="other"), "algorithm identity"),
    )
    for construct, message in invalid_results:
        with pytest.raises(ValueError, match=message):
            construct()


def test_mfe_result_rejects_zero_off_reference_and_analysis_rejects_zero_on_current() -> None:
    result = analyze_folded_mfe(
        _trace(),
        MfeAnalysisConfig(frames_per_state=4, steady_state_frames=2),
    )
    zero_off = (0.0,) * 4 + result.folded_relative_signal[4:]
    with pytest.raises(ValueError, match="field-OFF mean must be nonzero"):
        replace(result, folded_relative_signal=zero_off)

    all_off_current = (0.0,) * len(_trace().magnet_current_ampere)
    with pytest.raises(ValueError, match="constant nonzero-current ON state"):
        analyze_folded_mfe(
            replace(_trace(), magnet_current_ampere=all_off_current),
            MfeAnalysisConfig(frames_per_state=4, steady_state_frames=2),
        )
