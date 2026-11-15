import hashlib
import itertools
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from scipy.interpolate import make_interp_spline as scipy_make_interp_spline

from paircraft.experiments import oxford_corpus
from paircraft.experiments.oxford_corpus import (
    AcquisitionSpec,
    ExposureState,
    ResultStatus,
    analyze_trace_row,
    append_access_event,
    build_oxford_calibration_corpus,
    freeze_document,
    join_exposure,
    persist_freeze,
    preflight_npz,
    read_access_journal,
    read_admitted_arrays,
    require_resource_budget,
    summarize_acquisition,
    validate_component_partitions,
    validate_partitions,
)


def _fixture_spec(path: Path, *, rows: int = 1) -> AcquisitionSpec:
    return AcquisitionSpec(
        acquisition_id="fixture-acquisition",
        provider_file_id=1,
        provider_path="upload/fixture.npz",
        local_name=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        provider_md5=hashlib.md5(path.read_bytes()).hexdigest(),
        row_count=rows,
        metadata_path=("fixture",),
        construct_id="construct:fixture",
        initial_exposure=ExposureState.OUTCOME_SEALED,
    )


def _write_npz(
    path: Path,
    *,
    rows: int = 1,
    extra: bool = False,
    dtype: Any = np.float32,
    off_value: float = 100,
    on_value: float = 80,
) -> None:
    signal = np.tile(np.concatenate((np.full(45, off_value), np.full(45, on_value))), 10)
    arrays: dict[str, np.ndarray] = {
        "time_traces": np.tile(signal, (rows, 1)).astype(dtype),
        "mask": np.zeros((3200, 3200), dtype=np.bool_),
        "background": np.asarray(5, dtype=np.float32),
    }
    if extra:
        arrays["surprise"] = np.asarray(1)
    cast(Any, np.savez_compressed)(path, **arrays)


def _write_metadata(path: Path) -> str:
    path.write_text(
        json.dumps(
            {
                "fixture": {
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
            }
        ),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exposure_join_is_a_monotone_semilattice() -> None:
    states = tuple(ExposureState)
    for left, right, third in itertools.product(states, repeat=3):
        assert join_exposure(left, right) == join_exposure(right, left)
        assert join_exposure(left, left) == left
        assert join_exposure(join_exposure(left, right), third) == join_exposure(
            left, join_exposure(right, third)
        )
    assert (
        join_exposure(ExposureState.FIXED_ANSWER_FIXTURE, ExposureState.OUTCOME_INSPECTED)
        is ExposureState.FIXED_ANSWER_FIXTURE
    )


def test_preflight_authenticates_headers_without_loading_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "fixture.npz"
    _write_npz(path)
    spec = _fixture_spec(path)

    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("numeric load attempted"))
    record = preflight_npz(path, spec)

    assert record.acquisition_id == spec.acquisition_id
    assert {header.name for header in record.headers} == {
        "time_traces.npy",
        "mask.npy",
        "background.npy",
    }
    assert next(header for header in record.headers if header.name == "time_traces.npy").shape == (
        1,
        900,
    )


def test_preflight_refuses_identity_members_and_dtype_drift(tmp_path: Path) -> None:
    path = tmp_path / "fixture.npz"
    _write_npz(path)
    spec = _fixture_spec(path)
    with pytest.raises(ValueError, match="source_sha256_mismatch"):
        preflight_npz(path, replace(spec, sha256="0" * 64))

    _write_npz(path, extra=True)
    with pytest.raises(ValueError, match="unexpected_zip_members"):
        preflight_npz(path, _fixture_spec(path))

    _write_npz(path, dtype=np.float64)
    with pytest.raises(ValueError, match="invalid_npy_contract"):
        preflight_npz(path, _fixture_spec(path))

    _write_npz(path, dtype=object)
    with pytest.raises(ValueError, match="unsafe_npy_dtype"):
        preflight_npz(path, _fixture_spec(path))


def test_frozen_transform_uses_nine_retained_anchors_and_preserves_coordinates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "identity.npz"
    path.write_bytes(b"fixture")
    spec = _fixture_spec(path)
    signal = np.tile(np.concatenate((np.full(45, 100.0), np.full(45, 80.0))), 10)
    observed: dict[str, object] = {}

    def capture(x: np.ndarray, y: np.ndarray, **kwargs: Any) -> Any:
        observed.update(x=x.copy(), y=y.copy(), kwargs=kwargs)
        return scipy_make_interp_spline(x, y, **kwargs)

    monkeypatch.setattr(oxford_corpus, "make_interp_spline", capture)
    result = analyze_trace_row(signal, spec, 0)

    assert cast(np.ndarray[Any, Any], observed["x"]).tolist() == [
        43 + 90 * index for index in range(9)
    ]
    assert cast(np.ndarray[Any, Any], observed["y"]).dtype == np.float64
    assert observed["kwargs"] == {"k": 3, "bc_type": None}
    assert result.row.result_status is ResultStatus.PRODUCED
    assert result.row.median_cycle_ratio == pytest.approx(-0.2)
    assert len(result.cycles) == 9
    assert len(result.frames) == 810
    first = result.frames[0]
    assert (first.source_frame_index, first.source_period_index, first.nominal_state) == (
        90,
        1,
        "OFF",
    )
    assert [window.start_seconds for window in first.exposure_windows] == [20.0, 20.0, 21.0]
    last = result.frames[44]
    assert last.exposure_windows[0].start_seconds == pytest.approx(20 + 44 * 10 / 45)
    assert all(window.timestamp_source == "nominal_metadata" for window in last.exposure_windows)


def test_invalid_baseline_preserves_row_cycles_and_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "identity.npz"
    path.write_bytes(b"fixture")
    spec = _fixture_spec(path)
    signal = np.full(900, 100.0)

    class InvalidSpline:
        def __call__(self, indices: np.ndarray, **kwargs: object) -> np.ndarray:
            baseline = np.full(indices.shape, 100.0)
            baseline[100] = 0.0
            return baseline

    monkeypatch.setattr(
        oxford_corpus, "make_interp_spline", lambda *args, **kwargs: InvalidSpline()
    )
    result = analyze_trace_row(signal, spec, 0)

    assert result.row.result_status is ResultStatus.EXCLUDED
    assert result.row.reason_codes == ("nonpositive_baseline_retained_frame",)
    assert len(result.cycles) == 9 and all(not cycle.eligible for cycle in result.cycles)
    assert len(result.frames) == 810 and all(not frame.eligible for frame in result.frames)
    summary = summarize_acquisition(spec, [result])
    assert summary.result_status is ResultStatus.EXCLUDED
    assert summary.median_row_median is None
    assert summary.denominator_weighted_reconciliation is None


def test_freeze_precedes_hash_chained_access_and_corruption_refuses(tmp_path: Path) -> None:
    source = tmp_path / "identity.npz"
    source.write_bytes(b"fixture")
    spec = _fixture_spec(source)
    preflight = oxford_corpus.PreflightRecord(
        spec.acquisition_id, 7, spec.sha256, spec.provider_md5, ()
    )
    document = freeze_document(
        [preflight], {spec.connected_group_id: spec.initial_exposure}, specs=[spec]
    )
    digest, freeze_path = persist_freeze(tmp_path, document)
    assert hashlib.sha256(freeze_path.read_bytes()).hexdigest() == digest

    first = append_access_event(tmp_path, spec, preflight, digest, run_id="run")
    second = append_access_event(tmp_path, spec, preflight, digest, run_id="run")
    journal = tmp_path / "access" / "outcome-access.jsonl"
    events = read_access_journal(journal)
    assert len(events) == 2
    assert second["previous_event_sha256"] == first["event_sha256"]

    lines = journal.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])
    payload["source_id"] = "tampered"
    lines[0] = json.dumps(payload)
    journal.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="journal_hash"):
        read_access_journal(journal)


@pytest.mark.parametrize("damage", ["missing_newline", "whole_tail_event"])
def test_journal_head_detects_tail_loss(tmp_path: Path, damage: str) -> None:
    source = tmp_path / "identity.npz"
    source.write_bytes(b"fixture")
    spec = _fixture_spec(source)
    preflight = oxford_corpus.PreflightRecord(
        spec.acquisition_id, 7, spec.sha256, spec.provider_md5, ()
    )
    document = freeze_document(
        [preflight], {spec.connected_group_id: spec.initial_exposure}, specs=[spec]
    )
    digest, _ = persist_freeze(tmp_path, document)
    append_access_event(tmp_path, spec, preflight, digest, run_id="run")
    append_access_event(tmp_path, spec, preflight, digest, run_id="run")
    journal = tmp_path / "access/outcome-access.jsonl"
    raw = journal.read_bytes()
    if damage == "missing_newline":
        journal.write_bytes(raw[:-1])
        expected = "journal_tail"
    else:
        journal.write_bytes(raw.splitlines(keepends=True)[0])
        expected = "journal_head_mismatch"
    with pytest.raises(ValueError, match=expected):
        read_access_journal(journal)


def test_content_address_alone_cannot_authorize_outcome_access(tmp_path: Path) -> None:
    source = tmp_path / "identity.npz"
    source.write_bytes(b"fixture")
    spec = _fixture_spec(source)
    preflight = oxford_corpus.PreflightRecord(
        spec.acquisition_id, 7, spec.sha256, spec.provider_md5, ()
    )
    digest, _ = persist_freeze(tmp_path, {})
    with pytest.raises(ValueError, match="freeze_schema_mismatch"):
        append_access_event(tmp_path, spec, preflight, digest, run_id="run")


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("tier", "freeze_tier_mismatch"),
        ("transform", "freeze_transform_mismatch"),
        ("implementation", "freeze_implementation_mismatch"),
        ("timing", "freeze_timing_mismatch"),
        ("resources", "freeze_resource_contract_mismatch"),
        ("evidence", "freeze_evidence_identity_mismatch"),
        ("graph", "freeze_graph_status_mismatch"),
        ("development", "freeze_development_group_missing"),
        ("partition", "freeze_partition_mismatch"),
        ("acquisition", "freeze_acquisition_identity_mismatch"),
        ("preflight", "freeze_preflight_identity_mismatch"),
    ],
)
def test_freeze_semantics_are_authorization_not_merely_content_address(
    tmp_path: Path, mutation: str, reason: str
) -> None:
    source = tmp_path / "identity.npz"
    source.write_bytes(b"fixture")
    spec = _fixture_spec(source)
    preflight = oxford_corpus.PreflightRecord(
        spec.acquisition_id, 7, spec.sha256, spec.provider_md5, ()
    )
    document = freeze_document(
        [preflight], {spec.connected_group_id: spec.initial_exposure}, specs=[spec]
    )
    replacements: dict[str, tuple[str, object]] = {
        "tier": ("feasibility_tier", "V_unseen_variant"),
        "transform": ("transform", {}),
        "implementation": ("implementation_sha256", "0" * 64),
        "timing": ("timing", {}),
        "resources": ("resource_contract", {}),
        "evidence": ("bound_evidence_hashes", {}),
        "graph": ("source_graph_status", "reconciled"),
        "development": ("development_group_ids", []),
        "partition": ("holdout_group_ids", [spec.connected_group_id]),
        "acquisition": ("acquisitions", []),
        "preflight": ("preflights", []),
    }
    key, value = replacements[mutation]
    document[key] = value
    digest, _ = persist_freeze(tmp_path, document)
    with pytest.raises(ValueError, match=reason):
        append_access_event(tmp_path, spec, preflight, digest, run_id="run")


def test_bound_production_evidence_is_verified_from_exact_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "experiments/payloads/oxford-v3"
    provider_root = tmp_path / "experiments/provider-records"
    input_root.mkdir(parents=True)
    provider_root.mkdir(parents=True)
    paths = {
        "provider_record": provider_root / "oxford-30344995-v3.json",
        "inventory": tmp_path / "experiments/inventory.jsonl",
        "transformations": tmp_path / "experiments/transformations.jsonl",
        "metadata": input_root / "59258273_metadata.json",
        "utils": input_root / "59258276_utils.py",
        "fig1_notebook": input_root / "59258753_Fig1_PanelC.ipynb",
        "fig2_notebook": input_root / "59258738_Fig2.ipynb",
        "alias_readme": input_root / "59258675_README.txt",
        "zenodo_record": provider_root / "zenodo-14500235.json",
        "archive_members": tmp_path / "experiments/archive-members.jsonl",
    }
    for identity, path in paths.items():
        path.write_text(identity, encoding="utf-8")
    hashes = {
        identity: hashlib.sha256(path.read_bytes()).hexdigest() for identity, path in paths.items()
    }
    monkeypatch.setattr(oxford_corpus, "EVIDENCE_HASHES", hashes)
    verifier = cast(Any, oxford_corpus._verify_production_evidence)
    verifier(input_root)
    paths["inventory"].write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="bound_evidence_mismatch:inventory"):
        verifier(input_root)


def test_access_and_partition_guards_refuse_before_use(tmp_path: Path) -> None:
    source = tmp_path / "identity.npz"
    source.write_bytes(b"fixture")
    spec = _fixture_spec(source)
    preflight = oxford_corpus.PreflightRecord(
        spec.acquisition_id, 7, spec.sha256, spec.provider_md5, ()
    )
    with pytest.raises(ValueError, match="durable_pre_read_freeze_missing"):
        append_access_event(tmp_path, spec, preflight, "0" * 64, run_id="run")
    with pytest.raises(ValueError, match="connected_group_partition_overlap"):
        validate_partitions(["cg:one"], [], ["cg:one"])
    with pytest.raises(ValueError, match="connected_component_partition_overlap"):
        validate_component_partitions(
            {"source": "component", "alias": "component"},
            {"training": ["source"], "holdout": ["alias"]},
        )


def test_resource_budget_includes_threshold_and_all_caps(
    tmp_path: Path,
) -> None:
    threshold = oxford_corpus.MINIMUM_FREE_BYTES + 10 + 20
    require_resource_budget(
        tmp_path,
        persistent_bytes=10,
        scratch_bytes=20,
        transient_bytes=30,
        free_bytes=threshold,
    )
    with pytest.raises(ValueError, match="free_space_reserve_breached"):
        require_resource_budget(
            tmp_path,
            persistent_bytes=10,
            scratch_bytes=20,
            transient_bytes=30,
            free_bytes=threshold - 1,
        )
    with pytest.raises(ValueError, match="transient_allocation_budget_exceeded"):
        require_resource_budget(
            tmp_path,
            persistent_bytes=0,
            scratch_bytes=0,
            transient_bytes=oxford_corpus.MAX_TRANSIENT_BYTES + 1,
        )
    with pytest.raises(ValueError, match="rss_budget_exceeded"):
        require_resource_budget(
            tmp_path,
            persistent_bytes=0,
            scratch_bytes=0,
            transient_bytes=0,
            rss_bytes=oxford_corpus.MAX_RSS_BYTES + 1,
        )


def test_guarded_reader_never_opens_early_and_journal_survives_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "fixture.npz"
    _write_npz(path)
    spec = _fixture_spec(path)
    preflight = preflight_npz(path, spec)
    opened: list[bool] = []

    def fail_after_event(source: object) -> tuple[np.ndarray, np.ndarray, float]:
        opened.append(True)
        raise RuntimeError("injected numeric-open failure")

    monkeypatch.setattr(oxford_corpus, "_read_numeric_members", fail_after_event)
    with pytest.raises(ValueError, match="durable_pre_read_freeze_missing"):
        read_admitted_arrays(path, spec, preflight, tmp_path / "out", "0" * 64, run_id="run")
    assert opened == []

    document = freeze_document(
        [preflight], {spec.connected_group_id: spec.initial_exposure}, specs=[spec]
    )
    digest, _ = persist_freeze(tmp_path / "out", document)
    with pytest.raises(RuntimeError, match="numeric-open failure"):
        read_admitted_arrays(path, spec, preflight, tmp_path / "out", digest, run_id="run")
    assert opened == [True]
    assert len(read_access_journal(tmp_path / "out/access/outcome-access.jsonl")) == 1


def test_guarded_reader_pins_authenticated_descriptor_across_path_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "fixture.npz"
    replacement = tmp_path / "replacement.npz"
    _write_npz(path, off_value=100, on_value=80)
    _write_npz(replacement, off_value=300, on_value=200)
    spec = _fixture_spec(path)
    preflight = preflight_npz(path, spec)
    document = freeze_document(
        [preflight], {spec.connected_group_id: spec.initial_exposure}, specs=[spec]
    )
    output = tmp_path / "out"
    digest, _ = persist_freeze(output, document)
    original_append = oxford_corpus.append_access_event

    def append_then_swap(*args: Any, **kwargs: Any) -> dict[str, object]:
        event = original_append(*args, **kwargs)
        replacement.replace(path)
        return event

    monkeypatch.setattr(oxford_corpus, "append_access_event", append_then_swap)
    traces, _, _, event = read_admitted_arrays(path, spec, preflight, output, digest, run_id="run")
    assert traces[0, 0] == pytest.approx(100)
    assert event["source_id"] == spec.source_id


def test_guarded_reader_returns_its_event_not_concurrent_journal_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "fixture.npz"
    _write_npz(path)
    spec = _fixture_spec(path)
    preflight = preflight_npz(path, spec)
    document = freeze_document(
        [preflight], {spec.connected_group_id: spec.initial_exposure}, specs=[spec]
    )
    output = tmp_path / "out"
    digest, _ = persist_freeze(output, document)
    original_append = oxford_corpus.append_access_event

    def append_with_competing_writer(*args: Any, **kwargs: Any) -> dict[str, object]:
        own_event = original_append(*args, **kwargs)
        original_append(*args, **kwargs)
        return own_event

    monkeypatch.setattr(oxford_corpus, "append_access_event", append_with_competing_writer)
    _, _, _, own_event = read_admitted_arrays(path, spec, preflight, output, digest, run_id="run")
    events = read_access_journal(output / "access/outcome-access.jsonl")
    assert own_event["event_id"] == events[0]["event_id"]
    assert own_event["event_id"] != events[-1]["event_id"]


def test_atomic_compact_build_and_post_event_failure_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source = inputs / "fixture.npz"
    _write_npz(source)
    spec = _fixture_spec(source)
    metadata = inputs / "metadata.json"
    metadata_hash = _write_metadata(metadata)
    output = tmp_path / "output"

    destination = build_oxford_calibration_corpus(
        inputs,
        metadata,
        output,
        "complete",
        specs=[spec],
        metadata_sha256=metadata_hash,
    )
    corpus = json.loads((destination / "corpus.json").read_text(encoding="utf-8"))
    assert corpus["feasibility_tier"] == "D_descriptive"
    assert corpus["deposited_row_count"] == 1
    assert (destination / "frames.csv.gz").stat().st_size < 128 * 1024**2
    assert not list((output / "results").glob(".staging-*"))

    def fail_transform(*args: object, **kwargs: object) -> object:
        raise RuntimeError("injected transform failure")

    monkeypatch.setattr(oxford_corpus, "analyze_trace_row", fail_transform)
    with pytest.raises(RuntimeError, match="transform failure"):
        build_oxford_calibration_corpus(
            inputs,
            metadata,
            output,
            "failed",
            specs=[spec],
            metadata_sha256=metadata_hash,
        )
    assert not (output / "results/failed").exists()
    assert not list((output / "results").glob(".staging-*"))
    assert len(read_access_journal(output / "access/outcome-access.jsonl")) == 2
