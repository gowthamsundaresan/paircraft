import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from paircraft.experiments import (
    OxfordMaryProvenance,
    SourceProvenance,
    TraceClass,
    load_oxford_mary_scalar_trace,
    load_oxford_photodiode_csv,
)


def _provenance(path: Path, source_id: str) -> SourceProvenance:
    return SourceProvenance(
        source_id=source_id,
        source_url=f"https://example.test/{path.name}",
        source_version="fixed test fixture v1",
        exact_location=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        license="CC BY 4.0",
    )


def _mary_fixture(tmp_path: Path) -> tuple[Path, Path, Path, OxfordMaryProvenance]:
    periods = 5
    frames_per_state = 4
    frame_count = periods * frames_per_state * 2
    raw_path = tmp_path / "MARY_0.ome.tiff"
    config_path = tmp_path / "MARY_config.json"
    schedule_path = tmp_path / "MARY_data.npz"
    with tifffile.TiffWriter(raw_path) as writer:
        for value in np.arange(frame_count, dtype=np.float32):
            writer.write(np.asarray([[value]], dtype=np.float32), photometric="minisblack")
    config_path.write_text(
        json.dumps(
            {
                "Sample": "T7 MagLOV11 10s",
                "NumPeriods": periods,
                "ImagesPerOnPeriod": frames_per_state,
                "MagnetCurrent_A": [0.11] * periods,
                "ImageExposure_ms": 200,
            }
        ),
        encoding="utf-8",
    )
    current = np.tile(np.concatenate((np.zeros(4), np.full(4, 0.11))), periods)
    np.savez(
        schedule_path,
        timings=np.concatenate((np.full(frame_count, 250), np.array([10]))),
        camera_triggers=np.concatenate((np.full(frame_count, 5), np.array([-1]))),
        magnet_current=np.concatenate((current, np.array([0.0]))),
    )
    provenance = OxfordMaryProvenance(
        raw_trace=_provenance(raw_path, "raw"),
        configuration=_provenance(config_path, "config"),
        schedule=_provenance(schedule_path, "schedule"),
    )
    return raw_path, config_path, schedule_path, provenance


def test_mary_reader_aligns_terminal_event_without_calling_current_a_field(tmp_path: Path) -> None:
    raw, config, schedule, provenance = _mary_fixture(tmp_path)

    trace = load_oxford_mary_scalar_trace(
        raw,
        config,
        schedule,
        provenance=provenance,
        construct_id="construct-maglov2-domain-aa",
        trace_id="oxford-mary-maglov2",
    )

    assert trace.trace_class is TraceClass.INSTRUMENT_EXPORT_RAW_TRACE
    assert trace.sample_label == "T7 MagLOV11 10s"
    assert trace.signal_unit == "camera_count"
    assert len(trace.signal) == 40
    assert trace.time_seconds[0] == 0.0
    assert trace.time_seconds[-1] == pytest.approx(9.75)
    assert trace.magnet_current_ampere[:8] == (0.0,) * 4 + (0.11,) * 4
    assert len(trace.provenance) == 3


def test_mary_reader_rejects_ambiguous_extra_schedule_event(tmp_path: Path) -> None:
    raw, config, schedule, provenance = _mary_fixture(tmp_path)
    with np.load(schedule, allow_pickle=False) as payload:
        timings = payload["timings"]
        triggers = payload["camera_triggers"]
        current = payload["magnet_current"].copy()
    current[-1] = 0.11
    np.savez(schedule, timings=timings, camera_triggers=triggers, magnet_current=current)
    provenance = OxfordMaryProvenance(
        provenance.raw_trace,
        provenance.configuration,
        _provenance(schedule, "schedule"),
    )

    with pytest.raises(ValueError, match="terminal schedule event"):
        load_oxford_mary_scalar_trace(
            raw,
            config,
            schedule,
            provenance=provenance,
            construct_id="construct-maglov2-domain-aa",
            trace_id="oxford-mary-maglov2",
        )


def test_mary_reader_refuses_non_scalar_tiff_frames(tmp_path: Path) -> None:
    raw, config, schedule, provenance = _mary_fixture(tmp_path)
    with tifffile.TiffWriter(raw) as writer:
        for _ in range(40):
            writer.write(np.ones((2, 2), dtype=np.float32), photometric="minisblack")
    provenance = OxfordMaryProvenance(
        _provenance(raw, "raw"),
        provenance.configuration,
        provenance.schedule,
    )

    with pytest.raises(ValueError, match="refuses non-scalar TIFF"):
        load_oxford_mary_scalar_trace(
            raw,
            config,
            schedule,
            provenance=provenance,
            construct_id="construct-maglov2-domain-aa",
            trace_id="oxford-mary-maglov2",
        )


def test_photodiode_reader_preserves_metadata_and_normalizes_elapsed_time(tmp_path: Path) -> None:
    path = tmp_path / "measurement_0.csv"
    path.write_text(
        "Parameter,Value\n"
        "Current Reading (A),0.910078\n"
        "Sample Description,ML2-R11-f back only\n"
        "Date,2025-07-03 11:34:37\n"
        "\n"
        "Time (s),Measurement (W)\n"
        "1851.641174,0.0000000565443120\n"
        "1851.741174,0.0000000578210690\n",
        encoding="utf-8",
    )

    trace = load_oxford_photodiode_csv(
        path,
        provenance=_provenance(path, "oxford-measurement-0"),
        construct_id="construct-fast-domain-aa",
        trace_id="oxford-photodiode-0",
    )

    assert trace.observable == "optical power"
    assert trace.signal_unit == "W"
    assert trace.time_seconds == pytest.approx((0.0, 0.1))
    assert trace.magnet_current_ampere == (0.910078, 0.910078)
    assert ("Source time origin (s)", "1851.641174") in trace.attributes


def test_reader_checks_exact_payload_hash_before_parsing(tmp_path: Path) -> None:
    path = tmp_path / "measurement.csv"
    path.write_text("not the recorded payload", encoding="utf-8")
    provenance = SourceProvenance(
        source_id="changed",
        source_url="https://example.test/changed",
        source_version="v1",
        exact_location="measurement.csv",
        sha256="0" * 64,
        license="CC BY 4.0",
    )

    with pytest.raises(ValueError, match="sha256 does not match"):
        load_oxford_photodiode_csv(
            path,
            provenance=provenance,
            construct_id="construct-fast-domain-aa",
            trace_id="changed",
        )


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("Wrong,Header\n", "Parameter,Value header"),
        (
            "Parameter,Value\nCurrent Reading (A),1\nSample Description,x\n",
            "blank row",
        ),
        (
            "Parameter,Value\nDuplicate,one\nDuplicate,two\n\nTime (s),Measurement (W)\n0,1\n1,2\n",
            "unique key/value pairs",
        ),
        (
            "Parameter,Value\nCurrent Reading (A),1\nSample Description,x\n\n"
            "Wrong,Units\n0,1\n1,2\n",
            "seconds and watts",
        ),
        (
            "Parameter,Value\nCurrent Reading (A),1\nSample Description,x\n\n"
            "Time (s),Measurement (W)\n0,1\n",
            "at least two observations",
        ),
        (
            "Parameter,Value\nCurrent Reading (A),not-a-number\nSample Description,x\n\n"
            "Time (s),Measurement (W)\n0,1\n1,2\n",
            "invalid metadata or observations",
        ),
        (
            "Parameter,Value\nCurrent Reading (A),1\nSample Description,x\n\n"
            "Time (s),Measurement (W)\n0,1,extra\n1,2\n",
            "finite two-column values",
        ),
    ],
)
def test_photodiode_reader_rejects_malformed_external_records(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "malformed.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_oxford_photodiode_csv(
            path,
            provenance=_provenance(path, "malformed"),
            construct_id="construct-fast-domain-aa",
            trace_id="malformed",
        )
