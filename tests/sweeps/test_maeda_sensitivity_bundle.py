from __future__ import annotations

import fcntl
import json
import os
import stat
from collections.abc import Callable, Iterator
from hashlib import sha256
from pathlib import Path

import pytest

import paircraft.sweeps.maeda_sensitivity_bundle as bundle_module
import paircraft.sweeps.storage as storage_module
from paircraft.sweeps.maeda_sensitivity import (
    MaedaSensitivityJob,
    MaedaSensitivityPlan,
    default_maeda_local_sensitivity,
    materialize_maeda_local_sensitivity,
)
from paircraft.sweeps.maeda_sensitivity_bundle import (
    MaedaSensitivityBundle,
    _cleanup_root_leftovers,
    inspect_maeda_sensitivity_bundle,
    load_authenticated_maeda_sensitivity_bundle,
    run_maeda_sensitivity_bundle,
)
from paircraft.sweeps.storage import (
    DirectoryIdentity,
    PublishedDurabilityError,
    canonical_json_bytes,
    canonical_sha256,
    directory_identity,
    operational_write,
    publish_canonical_json_once,
)

type FastReplay = Callable[[MaedaSensitivityPlan, tuple[MaedaSensitivityJob, ...]], Iterator[int]]


@pytest.fixture(scope="module")
def plan() -> MaedaSensitivityPlan:
    return materialize_maeda_local_sensitivity(default_maeda_local_sensitivity())


@pytest.fixture
def fast_exact_replay(
    monkeypatch: pytest.MonkeyPatch,
    plan: MaedaSensitivityPlan,
) -> FastReplay:
    data = tuple(range(len(plan.jobs)))
    indices = {id(job): index for index, job in enumerate(plan.jobs)}
    parameter_families = tuple(job.parameter_family_sha256 for job in plan.jobs)
    design_dependencies = tuple(job.design_dependency_sha256 for job in plan.jobs)
    job_hashes = tuple(job.job_sha256 for job in plan.jobs)

    def execute(expected: MaedaSensitivityPlan) -> tuple[int, ...]:
        assert expected is plan
        return data

    def iterate(
        expected: MaedaSensitivityPlan,
        jobs: tuple[MaedaSensitivityJob, ...],
    ) -> Iterator[int]:
        assert expected is plan
        return iter(indices[id(job)] for job in jobs)

    def digest(value: str) -> str:
        return sha256(value.encode()).hexdigest()

    def document(index: int) -> dict[str, object]:
        return {
            "schema_version": "paircraft-simulation-datum-v1",
            "evidence_origin": "simulation",
            "membership": "reference_only_not_training",
            "prediction": {
                "oracle": {"artifact_sha256": digest("fixed-oracle")},
                "lineage": [
                    {
                        "artifact_id": "hypothetical-parameter-family",
                        "sha256": parameter_families[index],
                    },
                    {
                        "artifact_id": "sensitivity-design-dependency-group",
                        "sha256": design_dependencies[index],
                    },
                    {
                        "artifact_id": "sensitivity-plan-job",
                        "sha256": job_hashes[index],
                    },
                ],
                "distribution": {"mean": 0.4 + index / 10_000},
            },
            "execution": {},
            "hashes": {
                "datum_sha256": digest(f"datum:{index}"),
                "query_sha256": digest(f"query:{index}"),
                "prediction_sha256": digest(f"prediction:{index}"),
                "configuration_sha256": digest("configuration"),
                "execution_sha256": digest(f"execution:{index}"),
                "dependency_group_sha256": parameter_families[index],
            },
        }

    monkeypatch.setattr(bundle_module, "execute_maeda_sensitivity_oracle_plan", execute)
    monkeypatch.setattr(bundle_module, "iter_maeda_sensitivity_oracle_jobs", iterate)
    monkeypatch.setattr(bundle_module, "simulation_datum_document", document)
    return iterate


def test_runner_and_inspector_preserve_complete_scientific_grouping(
    tmp_path: Path,
    plan: MaedaSensitivityPlan,
    fast_exact_replay: FastReplay,
) -> None:
    del fast_exact_replay
    path = run_maeda_sensitivity_bundle(plan, tmp_path)
    bundle, documents = load_authenticated_maeda_sensitivity_bundle(path, expected_plan=plan)

    assert path.name == f"maeda-sensitivity-{plan.plan_sha256}.bundle.json"
    assert bundle.plan_sha256 == plan.plan_sha256
    assert len(bundle.results) == len(documents) == 204
    assert len({result.parameter_family_sha256 for result in bundle.results}) == 34
    assert len({result.design_dependency_sha256 for result in bundle.results}) == 2
    assert inspect_maeda_sensitivity_bundle(path, expected_plan=plan) == bundle
    assert all(document["membership"] == "reference_only_not_training" for document in documents)


def test_existing_bundle_is_never_overwritten(
    tmp_path: Path,
    plan: MaedaSensitivityPlan,
    fast_exact_replay: FastReplay,
) -> None:
    del fast_exact_replay
    run_maeda_sensitivity_bundle(plan, tmp_path)
    with pytest.raises(FileExistsError):
        run_maeda_sensitivity_bundle(plan, tmp_path)


def test_failure_preserves_partial_and_resume_replays_then_skips_completed_jobs(
    tmp_path: Path,
    plan: MaedaSensitivityPlan,
    fast_exact_replay: FastReplay,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact_iterate = fast_exact_replay

    def fail_after_three(
        expected: MaedaSensitivityPlan,
        jobs: tuple[MaedaSensitivityJob, ...],
    ) -> Iterator[int]:
        for position, value in enumerate(exact_iterate(expected, jobs)):
            if position == 3:
                raise ArithmeticError("injected sensitivity solve failure")
            yield value

    monkeypatch.setattr(bundle_module, "iter_maeda_sensitivity_oracle_jobs", fail_after_three)
    with pytest.raises(ArithmeticError, match="injected"):
        run_maeda_sensitivity_bundle(plan, tmp_path)
    partial = tmp_path / f".maeda-sensitivity-{plan.plan_sha256}.partial"
    assert len(tuple((partial / "results").iterdir())) == 3

    calls: list[tuple[str, ...]] = []

    def recording_iterate(
        expected: MaedaSensitivityPlan,
        jobs: tuple[MaedaSensitivityJob, ...],
    ) -> Iterator[int]:
        calls.append(tuple(job.job_sha256 for job in jobs))
        return exact_iterate(expected, jobs)

    monkeypatch.setattr(bundle_module, "iter_maeda_sensitivity_oracle_jobs", recording_iterate)
    final = run_maeda_sensitivity_bundle(plan, tmp_path, resume=True)

    assert final.is_file()
    assert len(calls) == 2
    assert calls[0] == tuple(job.job_sha256 for job in plan.jobs[:3])
    assert calls[1] == tuple(job.job_sha256 for job in plan.jobs[3:])
    assert not partial.exists()


def test_resume_rejects_tampered_partial_result(
    tmp_path: Path,
    plan: MaedaSensitivityPlan,
    fast_exact_replay: FastReplay,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact_iterate = fast_exact_replay

    def fail_after_one(
        expected: MaedaSensitivityPlan,
        jobs: tuple[MaedaSensitivityJob, ...],
    ) -> Iterator[int]:
        for position, value in enumerate(exact_iterate(expected, jobs)):
            if position == 1:
                raise ArithmeticError("injected sensitivity solve failure")
            yield value

    monkeypatch.setattr(bundle_module, "iter_maeda_sensitivity_oracle_jobs", fail_after_one)
    with pytest.raises(ArithmeticError):
        run_maeda_sensitivity_bundle(plan, tmp_path)
    partial = tmp_path / f".maeda-sensitivity-{plan.plan_sha256}.partial"
    result = next((partial / "results").iterdir())
    document = json.loads(result.read_bytes())
    document["simulation_datum"]["prediction"]["distribution"]["mean"] += 0.01
    result.write_bytes(canonical_json_bytes(document))
    monkeypatch.setattr(bundle_module, "iter_maeda_sensitivity_oracle_jobs", exact_iterate)

    with pytest.raises(ValueError, match="exact canonical sensitivity replay"):
        run_maeda_sensitivity_bundle(plan, tmp_path, resume=True)


def test_final_hard_link_is_commit_point_even_if_directory_fsync_fails(
    tmp_path: Path,
    plan: MaedaSensitivityPlan,
    fast_exact_replay: FastReplay,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fast_exact_replay
    real_fsync = os.fsync

    def fail_directory_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode) and any(
            path.name.endswith(".bundle.json") for path in tmp_path.iterdir()
        ):
            raise OSError("injected directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_directory_fsync)
    with pytest.raises(PublishedDurabilityError, match="published"):
        run_maeda_sensitivity_bundle(plan, tmp_path)
    final = tmp_path / f"maeda-sensitivity-{plan.plan_sha256}.bundle.json"
    assert final.is_file()


def test_post_commit_partial_cleanup_failure_does_not_report_unpublished(
    tmp_path: Path,
    plan: MaedaSensitivityPlan,
    fast_exact_replay: FastReplay,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fast_exact_replay

    def fail_cleanup(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(bundle_module, "remove_authenticated_directory", fail_cleanup)
    final = run_maeda_sensitivity_bundle(plan, tmp_path)

    assert final.is_file()


def test_partial_symlink_is_refused(
    tmp_path: Path,
    plan: MaedaSensitivityPlan,
    fast_exact_replay: FastReplay,
) -> None:
    del fast_exact_replay
    partial = tmp_path / f".maeda-sensitivity-{plan.plan_sha256}.partial"
    target = tmp_path / "elsewhere"
    target.mkdir()
    partial.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="real non-symlink directory"):
        run_maeda_sensitivity_bundle(plan, tmp_path, resume=True)


def test_results_directory_replacement_is_detected_without_writing_replacement(
    tmp_path: Path,
    plan: MaedaSensitivityPlan,
    fast_exact_replay: FastReplay,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fast_exact_replay
    real_publish = bundle_module.atomic_link_bytes
    preserved = tmp_path / "preserved-results"
    replaced = False

    def replace_before_publish(
        directory: Path,
        destination: Path,
        payload: bytes,
        *,
        expected_identity: DirectoryIdentity,
    ) -> None:
        nonlocal replaced
        if directory.name == "results" and not replaced:
            replaced = True
            directory.rename(preserved)
            directory.mkdir()
        real_publish(
            directory,
            destination,
            payload,
            expected_identity=expected_identity,
        )

    monkeypatch.setattr(bundle_module, "atomic_link_bytes", replace_before_publish)
    with pytest.raises(ValueError, match="replaced"):
        run_maeda_sensitivity_bundle(plan, tmp_path)

    assert preserved.is_dir()
    replacement = tmp_path / f".maeda-sensitivity-{plan.plan_sha256}.partial" / "results"
    assert replacement.is_dir()
    assert tuple(replacement.iterdir()) == ()


def test_plan_lock_refuses_concurrent_runner(
    tmp_path: Path,
    plan: MaedaSensitivityPlan,
    fast_exact_replay: FastReplay,
) -> None:
    del fast_exact_replay
    lock = tmp_path / f".maeda-sensitivity-{plan.plan_sha256}.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            run_maeda_sensitivity_bundle(plan, tmp_path)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_unlinked_and_recreated_named_lock_is_rejected(
    tmp_path: Path,
    plan: MaedaSensitivityPlan,
    fast_exact_replay: FastReplay,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fast_exact_replay
    real_acquire = bundle_module.acquire_lock
    preserved = tmp_path / "preserved-lock"

    def replace_lock(
        path: Path,
        *,
        expected_parent_identity: DirectoryIdentity,
    ) -> int:
        descriptor = real_acquire(path, expected_parent_identity=expected_parent_identity)
        path.rename(preserved)
        path.write_bytes(b"replacement")
        return descriptor

    monkeypatch.setattr(bundle_module, "acquire_lock", replace_lock)
    with pytest.raises(ValueError, match="named lock path"):
        run_maeda_sensitivity_bundle(plan, tmp_path)

    assert preserved.is_file()
    assert (tmp_path / f".maeda-sensitivity-{plan.plan_sha256}.lock").read_bytes() == (
        b"replacement"
    )


def test_cleanup_quarantines_and_rejects_substituted_publication_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = tmp_path / "result.bundle.json"
    partial = tmp_path / ".result.partial"
    temporary = tmp_path / f".tmp-{final.name}-0123456789abcdef"
    preserved = tmp_path / "preserved-original"
    temporary.write_bytes(b"original")
    real_rename = os.rename

    def substitute(
        source: str | bytes,
        destination: str | bytes,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        if source == temporary.name:
            real_rename(
                source,
                preserved.name,
                src_dir_fd=src_dir_fd,
                dst_dir_fd=dst_dir_fd,
            )
            assert src_dir_fd is not None
            descriptor = os.open(
                temporary.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=src_dir_fd,
            )
            os.write(descriptor, b"replacement")
            os.close(descriptor)
        real_rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(os, "rename", substitute)
    with pytest.raises(ValueError, match="quarantined publication temporary was replaced"):
        _cleanup_root_leftovers(
            tmp_path,
            final,
            partial,
            expected_root_identity=directory_identity(tmp_path, "test root"),
        )

    assert preserved.read_bytes() == b"original"


def test_inspector_rejects_even_coherently_rehashed_noncanonical_result(
    tmp_path: Path,
    plan: MaedaSensitivityPlan,
    fast_exact_replay: FastReplay,
) -> None:
    del fast_exact_replay
    original = run_maeda_sensitivity_bundle(plan, tmp_path / "source")
    document = json.loads(original.read_bytes())
    document["results"][0]["simulation_datum"]["prediction"]["distribution"]["mean"] += 1e-6
    tampered = tmp_path / "tampered.bundle.json"
    tampered.write_bytes(canonical_json_bytes(document))

    with pytest.raises(ValueError, match="exact canonical sensitivity replay"):
        inspect_maeda_sensitivity_bundle(tampered, expected_plan=plan)


def test_inspector_rejects_coherently_rehashed_oracle_artifact_forgery(
    tmp_path: Path,
    plan: MaedaSensitivityPlan,
    fast_exact_replay: FastReplay,
) -> None:
    del fast_exact_replay
    original = run_maeda_sensitivity_bundle(plan, tmp_path / "source")
    document = json.loads(original.read_bytes())
    document["oracle_artifact_sha256"] = "0" * 64
    identity = {
        key: document[key]
        for key in (
            "schema_version",
            "status",
            "plan_sha256",
            "specification_sha256",
            "oracle_artifact_sha256",
            "results_sha256",
            "claim_boundary",
        )
    }
    document["hashes"]["bundle_sha256"] = canonical_sha256(identity)
    forged = tmp_path / "forged.bundle.json"
    forged.write_bytes(canonical_json_bytes(document))

    with pytest.raises(ValueError, match="oracle artifact does not match exact replay"):
        inspect_maeda_sensitivity_bundle(forged, expected_plan=plan)


def test_generic_publisher_is_canonical_crash_safe_and_no_replace(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "artifact.json"
    value = {"z": 1, "a": [2, 3]}

    assert publish_canonical_json_once(path, value) == path
    assert path.read_bytes() == b'{"a":[2,3],"z":1}'
    with pytest.raises(FileExistsError):
        publish_canonical_json_once(path, {"replacement": True})
    assert path.read_bytes() == b'{"a":[2,3],"z":1}'


def test_publishers_reject_nonobject_json(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="top-level object"):
        publish_canonical_json_once(tmp_path / "array.json", [1, 2, 3])

    directory = tmp_path / "atomic"
    directory.mkdir()
    identity = directory_identity(directory, "atomic test directory")
    with pytest.raises(ValueError, match="must contain one JSON object"):
        bundle_module.atomic_link_bytes(
            directory,
            directory / "array.json",
            b"[1,2,3]",
            expected_identity=identity,
        )


def test_operational_write_rejects_oversize_receipt_before_creating_a_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(storage_module, "_MAX_JSON_BYTES", 64)
    identity = directory_identity(tmp_path, "operational-write test directory")

    with pytest.raises(ValueError, match="invalid size"):
        operational_write(
            tmp_path / "failure.json",
            {"message": "x" * 128},
            expected_parent_identity=identity,
        )

    assert tuple(tmp_path.iterdir()) == ()


def test_inspector_rejects_noncanonical_and_duplicate_json(
    tmp_path: Path,
    plan: MaedaSensitivityPlan,
) -> None:
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text('{"x": 1}')
    with pytest.raises(ValueError, match="canonical"):
        inspect_maeda_sensitivity_bundle(noncanonical, expected_plan=plan)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"x":1,"x":2}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        inspect_maeda_sensitivity_bundle(duplicate, expected_plan=plan)


def test_bundle_identity_cannot_be_constructed_without_exact_replay() -> None:
    with pytest.raises(ValueError, match="exact replay authentication"):
        MaedaSensitivityBundle(
            plan_sha256="0" * 64,
            specification_sha256="1" * 64,
            oracle_artifact_sha256="2" * 64,
            results=(),
            results_sha256="3" * 64,
            bundle_sha256="4" * 64,
            _construction_key=object(),
        )
