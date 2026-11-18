from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

import paircraft.sweeps.bundle as bundle_module
from paircraft.oracle import SimulationDatum
from paircraft.spin import MaedaProteinSystem
from paircraft.sweeps.bundle import (
    MaedaSweepBundle,
    StoredSimulationDatumRef,
    inspect_maeda_sweep_bundle,
    load_authenticated_maeda_sweep_bundle,
    load_authenticated_maeda_sweep_documents,
    run_maeda_sweep_bundle,
)
from paircraft.sweeps.maeda import (
    IdealFieldPoint,
    MaedaSweepJob,
    MaedaSweepPlan,
    MaedaSweepSpecification,
    execute_maeda_sweep_job,
    maeda_sweep_target,
    materialize_maeda_sweep,
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _configuration(execution: dict[str, object]) -> dict[str, object]:
    return {
        key: execution[key]
        for key in (
            "model_id",
            "model_family",
            "solver_name",
            "solver_versions",
            "code_version",
            "code_artifact_sha256",
            "settings",
            "deterministic",
            "random_seed",
            "randomness_note",
        )
    }


@pytest.fixture(scope="module")
def plan_and_results() -> tuple[
    MaedaSweepPlan,
    dict[str, SimulationDatum],
]:
    specification = MaedaSweepSpecification(
        sweep_id="bundle-test",
        targets=(maeda_sweep_target(MaedaProteinSystem.ATCRY1),),
        field_points=(IdealFieldPoint("zero", 0.0), IdealFieldPoint("fifty-ut", 50e-6)),
        claim_boundary="Exact simulator reference sweep for bundle tests.",
    )
    plan = materialize_maeda_sweep(specification)
    return plan, {job.job_sha256: execute_maeda_sweep_job(plan, job) for job in plan.jobs}


def _install_executor(
    monkeypatch: pytest.MonkeyPatch,
    results: dict[str, SimulationDatum],
    calls: list[str],
    *,
    fail_job: str | None = None,
) -> None:
    def execute(
        plan: MaedaSweepPlan,
        job: MaedaSweepJob,
        oracle: object,
    ) -> SimulationDatum:
        del plan, oracle
        calls.append(job.job_sha256)
        if job.job_sha256 == fail_job:
            raise ArithmeticError("sanitized injected failure")
        return results[job.job_sha256]

    monkeypatch.setattr(bundle_module, "execute_maeda_sweep_job", execute)


def _rehash_datum(document: dict[str, object]) -> bytes:
    prediction = document["prediction"]
    assert isinstance(prediction, dict)
    execution = document["execution"]
    assert isinstance(execution, dict)
    query = prediction["query"]
    configuration = _configuration(execution)
    hashes = document["hashes"]
    assert isinstance(hashes, dict)
    query_hash = _sha(query)
    prediction_hash = _sha(prediction)
    configuration_hash = _sha(configuration)
    execution_hash = _sha(
        {
            "configuration_sha256": configuration_hash,
            "achieved_relative_residual": execution["achieved_relative_residual"],
            "iteration_count": execution["iteration_count"],
        }
    )
    hashes.update(
        {
            "query_sha256": query_hash,
            "prediction_sha256": prediction_hash,
            "configuration_sha256": configuration_hash,
            "execution_sha256": execution_hash,
            "datum_sha256": _sha(
                {
                    "schema_version": document["schema_version"],
                    "query_sha256": query_hash,
                    "prediction_sha256": prediction_hash,
                    "execution_sha256": execution_hash,
                    "dependency_group_sha256": hashes["dependency_group_sha256"],
                    "evidence_origin": document["evidence_origin"],
                    "membership": document["membership"],
                }
            ),
        }
    )
    return _canonical(document)


def test_success_publishes_one_authenticated_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan_and_results: tuple[MaedaSweepPlan, dict[str, SimulationDatum]],
) -> None:
    plan, results = plan_and_results
    calls: list[str] = []
    _install_executor(monkeypatch, results, calls)

    path = run_maeda_sweep_bundle(plan, tmp_path)
    bundle = inspect_maeda_sweep_bundle(path, expected_plan=plan)

    assert path.name == f"maeda-{plan.plan_sha256}.bundle.json"
    assert calls == [job.job_sha256 for job in plan.jobs]
    assert bundle.plan_sha256 == plan.plan_sha256
    assert len(bundle.results) == 2
    assert len(load_authenticated_maeda_sweep_documents(path, expected_plan=plan)) == 2
    loaded_bundle, documents = load_authenticated_maeda_sweep_bundle(path, expected_plan=plan)
    assert loaded_bundle == bundle
    assert len(documents) == 2
    assert not any(item.name.endswith(".partial") for item in tmp_path.iterdir())


def test_existing_final_is_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan_and_results: tuple[MaedaSweepPlan, dict[str, SimulationDatum]],
) -> None:
    plan, results = plan_and_results
    _install_executor(monkeypatch, results, [])
    path = run_maeda_sweep_bundle(plan, tmp_path)
    before = path.read_bytes()

    with pytest.raises(FileExistsError):
        run_maeda_sweep_bundle(plan, tmp_path, resume=True)

    assert path.read_bytes() == before


def test_failure_preserves_authenticated_partial_and_resume_skips_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan_and_results: tuple[MaedaSweepPlan, dict[str, SimulationDatum]],
) -> None:
    plan, results = plan_and_results
    first, second = plan.jobs
    failed_calls: list[str] = []
    _install_executor(
        monkeypatch,
        results,
        failed_calls,
        fail_job=second.job_sha256,
    )
    with pytest.raises(ArithmeticError, match="injected failure"):
        run_maeda_sweep_bundle(plan, tmp_path)

    final = tmp_path / f"maeda-{plan.plan_sha256}.bundle.json"
    partial = tmp_path / f".maeda-{plan.plan_sha256}.partial"
    assert not final.exists()
    assert (partial / "results" / f"{first.job_sha256}.json").is_file()
    resumed_calls: list[str] = []
    _install_executor(monkeypatch, results, resumed_calls)

    path = run_maeda_sweep_bundle(plan, tmp_path, resume=True)

    assert resumed_calls == [second.job_sha256]
    assert inspect_maeda_sweep_bundle(path, expected_plan=plan).plan_sha256 == plan.plan_sha256


def test_tampered_partial_fails_instead_of_recomputing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan_and_results: tuple[MaedaSweepPlan, dict[str, SimulationDatum]],
) -> None:
    plan, results = plan_and_results
    second = plan.jobs[1]
    _install_executor(monkeypatch, results, [], fail_job=second.job_sha256)
    with pytest.raises(ArithmeticError):
        run_maeda_sweep_bundle(plan, tmp_path)
    partial = tmp_path / f".maeda-{plan.plan_sha256}.partial"
    result = next((partial / "results").iterdir())
    payload = json.loads(result.read_text())
    payload["prediction"]["distribution"]["mean"] = 0.25
    result.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))

    with pytest.raises(ValueError, match=r"point yield|derived hash"):
        run_maeda_sweep_bundle(plan, tmp_path, resume=True)


@pytest.mark.parametrize(
    "mutation",
    [
        "calibration",
        "uncertainty",
        "applicability",
        "lineage",
        "claim",
        "yield",
        "rights",
        "provenance",
        "residual",
    ],
)
def test_coherently_rehashed_semantic_tampering_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan_and_results: tuple[MaedaSweepPlan, dict[str, SimulationDatum]],
    mutation: str,
) -> None:
    plan, results = plan_and_results
    _install_executor(monkeypatch, results, [], fail_job=plan.jobs[1].job_sha256)
    with pytest.raises(ArithmeticError):
        run_maeda_sweep_bundle(plan, tmp_path)
    partial = tmp_path / f".maeda-{plan.plan_sha256}.partial"
    result = partial / "results" / f"{plan.jobs[0].job_sha256}.json"
    payload = json.loads(result.read_bytes())
    prediction = payload["prediction"]
    if mutation == "calibration":
        prediction["distribution"]["calibration_status"] = "calibrated"
    elif mutation == "uncertainty":
        prediction["distribution"]["epistemic_standard_uncertainty"] = 0.0
    elif mutation == "applicability":
        prediction["applicability"]["status"] = "boundary"
    elif mutation == "lineage":
        prediction["lineage"][0]["sha256"] = "0" * 64
    elif mutation == "yield":
        prediction["distribution"].update({"mean": 0.123, "lower": 0.123, "upper": 0.123})
    elif mutation == "rights":
        prediction["lineage"][1]["license"] = "rewritten rights assertion"
    elif mutation == "provenance":
        prediction["lineage"][1]["source_uri"] = "https://example.invalid/rewritten"
    elif mutation == "residual":
        execution = payload["execution"]
        execution["achieved_relative_residual"] /= 2
        prediction["execution"] = execution
        prediction["distribution"]["method"] = (
            "deterministic infinite-time Maeda RP2 solve; "
            f"residual={execution['achieved_relative_residual']:.6g}; "
            f"iterations={execution['iteration_count']}"
        )
        prediction["lineage"][-1]["sha256"] = _sha(
            {
                "configuration_sha256": _sha(_configuration(execution)),
                "residual": execution["achieved_relative_residual"],
                "iterations": execution["iteration_count"],
            }
        )
    else:
        prediction["claim_boundary"] = "broader claim"
    result.write_bytes(_rehash_datum(payload))

    with pytest.raises(ValueError):
        run_maeda_sweep_bundle(plan, tmp_path, resume=True)


def test_partial_symlink_is_refused(
    tmp_path: Path,
    plan_and_results: tuple[MaedaSweepPlan, dict[str, SimulationDatum]],
) -> None:
    plan, _ = plan_and_results
    partial = tmp_path / f".maeda-{plan.plan_sha256}.partial"
    tmp_path.mkdir(exist_ok=True)
    target = tmp_path / "elsewhere"
    target.mkdir()
    partial.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="real non-symlink directory"):
        run_maeda_sweep_bundle(plan, tmp_path, resume=True)


def test_inspector_rejects_noncanonical_and_duplicate_key_json(
    tmp_path: Path,
    plan_and_results: tuple[MaedaSweepPlan, dict[str, SimulationDatum]],
) -> None:
    plan, _ = plan_and_results
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text('{"x": 1}')
    with pytest.raises(ValueError, match="canonical"):
        inspect_maeda_sweep_bundle(noncanonical, expected_plan=plan)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"x":1,"x":2}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        inspect_maeda_sweep_bundle(duplicate, expected_plan=plan)


def test_inspection_requires_expected_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan_and_results: tuple[MaedaSweepPlan, dict[str, SimulationDatum]],
) -> None:
    plan, results = plan_and_results
    _install_executor(monkeypatch, results, [])
    path = run_maeda_sweep_bundle(plan, tmp_path)

    with pytest.raises(TypeError, match="expected_plan"):
        inspect_maeda_sweep_bundle(path)  # type: ignore[call-arg]


def test_inspection_rejects_wrong_expected_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan_and_results: tuple[MaedaSweepPlan, dict[str, SimulationDatum]],
) -> None:
    plan, results = plan_and_results
    _install_executor(monkeypatch, results, [])
    path = run_maeda_sweep_bundle(plan, tmp_path)
    wrong = materialize_maeda_sweep(
        MaedaSweepSpecification(
            sweep_id="different-plan",
            targets=plan.specification.targets,
            field_points=plan.specification.field_points,
            claim_boundary=plan.specification.claim_boundary,
        )
    )

    with pytest.raises(ValueError, match="expected plan"):
        inspect_maeda_sweep_bundle(path, expected_plan=wrong)


def test_resume_removes_regular_crash_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan_and_results: tuple[MaedaSweepPlan, dict[str, SimulationDatum]],
) -> None:
    plan, results = plan_and_results
    _install_executor(monkeypatch, results, [], fail_job=plan.jobs[1].job_sha256)
    with pytest.raises(ArithmeticError):
        run_maeda_sweep_bundle(plan, tmp_path)
    partial = tmp_path / f".maeda-{plan.plan_sha256}.partial"
    temporary = partial / "results" / ".tmp-crash-leftover"
    temporary.write_text("uncommitted")
    _install_executor(monkeypatch, results, [])

    final = run_maeda_sweep_bundle(plan, tmp_path, resume=True)

    assert final.is_file()
    assert not partial.exists()


def test_published_bundle_survives_partial_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan_and_results: tuple[MaedaSweepPlan, dict[str, SimulationDatum]],
) -> None:
    plan, results = plan_and_results
    _install_executor(monkeypatch, results, [])
    real_remove_tree = bundle_module._remove_tree_at

    def fail_partial_cleanup(
        parent_fd: int,
        name: str,
        *,
        expected_identity: bundle_module._DirectoryIdentity | None = None,
    ) -> None:
        if name.startswith(".cleanup-"):
            raise OSError("injected cleanup failure")
        real_remove_tree(parent_fd, name, expected_identity=expected_identity)

    monkeypatch.setattr(bundle_module, "_remove_tree_at", fail_partial_cleanup)

    final = run_maeda_sweep_bundle(plan, tmp_path)

    assert inspect_maeda_sweep_bundle(final, expected_plan=plan).plan_sha256 == plan.plan_sha256
    assert (tmp_path / f".maeda-{plan.plan_sha256}.partial").is_dir()


def test_post_commit_temporary_unlink_failure_is_nonfatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan_and_results: tuple[MaedaSweepPlan, dict[str, SimulationDatum]],
) -> None:
    plan, results = plan_and_results
    _install_executor(monkeypatch, results, [])
    real_unlink = os.unlink

    def fail_final_temp_once(
        path: str | bytes,
        *,
        dir_fd: int | None = None,
    ) -> None:
        if isinstance(path, str) and ".bundle.json-" in path:
            raise OSError("injected committed-temp cleanup failure")
        real_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "unlink", fail_final_temp_once)

    final = run_maeda_sweep_bundle(plan, tmp_path)

    assert inspect_maeda_sweep_bundle(final, expected_plan=plan).plan_sha256 == plan.plan_sha256
    assert any(item.name.startswith(f".tmp-{final.name}-") for item in tmp_path.iterdir())


def test_resume_cleans_strict_root_temp_and_init_leftovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan_and_results: tuple[MaedaSweepPlan, dict[str, SimulationDatum]],
) -> None:
    plan, results = plan_and_results
    _install_executor(monkeypatch, results, [], fail_job=plan.jobs[1].job_sha256)
    with pytest.raises(ArithmeticError):
        run_maeda_sweep_bundle(plan, tmp_path)
    final_name = f"maeda-{plan.plan_sha256}.bundle.json"
    root_temp = tmp_path / f".tmp-{final_name}-0123456789abcdef"
    init = tmp_path / f".init-.maeda-{plan.plan_sha256}.partial-abcdef12"
    root_temp.write_text("interrupted publication")
    init.mkdir()
    (init / "incomplete").write_text("interrupted initialization")
    _install_executor(monkeypatch, results, [])

    final = run_maeda_sweep_bundle(plan, tmp_path, resume=True)

    assert final.is_file()
    assert not root_temp.exists()
    assert not init.exists()


def test_root_cleanup_refuses_replaced_initialization_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = tmp_path / "maeda-abc.bundle.json"
    partial = tmp_path / ".maeda-abc.partial"
    initialization = tmp_path / ".init-.maeda-abc.partial-abcdef12"
    preserved = tmp_path / "preserved-initialization"
    initialization.mkdir()
    (initialization / "original").write_text("preserve me")
    real_remove_tree = bundle_module._remove_tree_at

    def replace_then_remove(
        parent_fd: int,
        name: str,
        *,
        expected_identity: bundle_module._DirectoryIdentity | None = None,
    ) -> None:
        if name == initialization.name:
            initialization.rename(preserved)
            initialization.mkdir()
            (initialization / "replacement").write_text("do not delete")
        real_remove_tree(parent_fd, name, expected_identity=expected_identity)

    monkeypatch.setattr(bundle_module, "_remove_tree_at", replace_then_remove)

    with pytest.raises(ValueError, match="replaced"):
        bundle_module._cleanup_root_leftovers(
            tmp_path,
            final,
            partial,
            expected_root_identity=bundle_module._directory_identity(tmp_path, "test root"),
        )

    assert (preserved / "original").is_file()
    assert (initialization / "replacement").is_file()


def test_mutators_reject_wrong_parent_identity(tmp_path: Path) -> None:
    actual = bundle_module._directory_identity(tmp_path, "test root")
    wrong = bundle_module._DirectoryIdentity(actual.device, actual.inode + 1)

    with pytest.raises(ValueError, match="replaced"):
        bundle_module._operational_write(
            tmp_path / "receipt.json",
            {"safe": True},
            expected_parent_identity=wrong,
        )
    with pytest.raises(ValueError, match="replaced"):
        bundle_module._cleanup_root_leftovers(
            tmp_path,
            tmp_path / "final.json",
            tmp_path / ".partial",
            expected_root_identity=wrong,
        )

    assert not (tmp_path / "receipt.json").exists()


def test_recursive_cleanup_refuses_substituted_regular_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    victim = tree / "victim"
    preserved = tree / "preserved"
    victim.write_text("original")
    real_open = os.open
    substituted = False

    def substitute_before_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal substituted
        if path == "victim" and not substituted:
            substituted = True
            victim.rename(preserved)
            victim.write_text("replacement")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", substitute_before_open)
    root_fd = real_open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        identity = bundle_module._directory_identity(tree, "test tree")
        with pytest.raises(ValueError, match="regular file was replaced"):
            bundle_module._remove_tree_at(root_fd, tree.name, expected_identity=identity)
    finally:
        os.close(root_fd)

    assert preserved.read_text() == "original"
    assert victim.read_text() == "replacement"


def test_resume_rejects_noncanonical_failure_receipt_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan_and_results: tuple[MaedaSweepPlan, dict[str, SimulationDatum]],
) -> None:
    plan, results = plan_and_results
    _install_executor(monkeypatch, results, [], fail_job=plan.jobs[1].job_sha256)
    with pytest.raises(ArithmeticError):
        run_maeda_sweep_bundle(plan, tmp_path)
    receipt = tmp_path / f".maeda-{plan.plan_sha256}.partial" / "last-failure.json"
    payload = json.loads(receipt.read_bytes())
    payload["detail"] = "not permitted"
    receipt.write_bytes(_canonical(payload))

    with pytest.raises(ValueError, match="failure receipt"):
        run_maeda_sweep_bundle(plan, tmp_path, resume=True)


def test_replaced_results_directory_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan_and_results: tuple[MaedaSweepPlan, dict[str, SimulationDatum]],
) -> None:
    plan, results = plan_and_results
    _install_executor(monkeypatch, results, [])
    real_partial_results = bundle_module._partial_results
    replaced = False

    def replace_then_read(
        active_plan: MaedaSweepPlan,
        results_directory: Path,
        *,
        expected_identity: bundle_module._DirectoryIdentity,
    ) -> object:
        nonlocal replaced
        if not replaced:
            replaced = True
            moved = results_directory.with_name("results-replaced")
            results_directory.rename(moved)
            results_directory.mkdir()
        return real_partial_results(
            active_plan,
            results_directory,
            expected_identity=expected_identity,
        )

    monkeypatch.setattr(bundle_module, "_partial_results", replace_then_read)

    with pytest.raises(ValueError, match="replaced"):
        run_maeda_sweep_bundle(plan, tmp_path)


def test_replaced_output_root_is_detected_at_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan_and_results: tuple[MaedaSweepPlan, dict[str, SimulationDatum]],
) -> None:
    plan, results = plan_and_results
    _install_executor(monkeypatch, results, [])
    real_atomic_link = bundle_module._atomic_link_bytes
    moved_root = tmp_path.with_name(f"{tmp_path.name}-moved")
    replaced = False

    def replace_root_then_publish(
        directory: Path,
        destination: Path,
        payload: bytes,
        *,
        expected_identity: bundle_module._DirectoryIdentity,
    ) -> None:
        nonlocal replaced
        if destination.name.endswith(".bundle.json") and not replaced:
            replaced = True
            directory.rename(moved_root)
            directory.mkdir()
        real_atomic_link(
            directory,
            destination,
            payload,
            expected_identity=expected_identity,
        )

    monkeypatch.setattr(bundle_module, "_atomic_link_bytes", replace_root_then_publish)

    try:
        with pytest.raises(ValueError, match="replaced"):
            run_maeda_sweep_bundle(plan, tmp_path)
    finally:
        shutil.rmtree(moved_root, ignore_errors=True)


def test_inspector_rejects_coherently_rehashed_mixed_configurations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan_and_results: tuple[MaedaSweepPlan, dict[str, SimulationDatum]],
) -> None:
    plan, results = plan_and_results
    _install_executor(monkeypatch, results, [])
    final = run_maeda_sweep_bundle(plan, tmp_path)
    document = json.loads(final.read_bytes())
    record = document["results"][1]
    datum = record["simulation_datum"]
    execution = datum["execution"]
    execution["code_version"] = "different-build"
    datum["prediction"]["execution"] = execution
    configuration_hash = _sha(_configuration(execution))
    datum["prediction"]["lineage"][-1]["sha256"] = _sha(
        {
            "configuration_sha256": configuration_hash,
            "residual": execution["achieved_relative_residual"],
            "iterations": execution["iteration_count"],
        }
    )
    raw_datum = _rehash_datum(datum)
    record["simulation_datum"] = json.loads(raw_datum)
    hashes = record["simulation_datum"]["hashes"]
    record["reference"].update(
        {
            "document_sha256": hashlib.sha256(raw_datum).hexdigest(),
            "datum_sha256": hashes["datum_sha256"],
            "query_sha256": hashes["query_sha256"],
            "prediction_sha256": hashes["prediction_sha256"],
            "configuration_sha256": hashes["configuration_sha256"],
            "execution_sha256": hashes["execution_sha256"],
            "dependency_group_sha256": hashes["dependency_group_sha256"],
        }
    )
    results_hash = _sha(document["results"])
    document["results_sha256"] = results_hash
    document["hashes"]["results_sha256"] = results_hash
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
    document["hashes"]["bundle_sha256"] = _sha(identity)
    final.write_bytes(_canonical(document))

    with pytest.raises(
        ValueError,
        match=r"canonical deterministic Maeda result|share one execution configuration",
    ):
        inspect_maeda_sweep_bundle(final, expected_plan=plan)


@pytest.mark.parametrize("corruption", ["unknown", "control", "result", "failure"])
def test_resume_refuses_unknown_or_symlinked_partial_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan_and_results: tuple[MaedaSweepPlan, dict[str, SimulationDatum]],
    corruption: str,
) -> None:
    plan, results = plan_and_results
    _install_executor(monkeypatch, results, [], fail_job=plan.jobs[1].job_sha256)
    with pytest.raises(ArithmeticError):
        run_maeda_sweep_bundle(plan, tmp_path)
    partial = tmp_path / f".maeda-{plan.plan_sha256}.partial"
    target = tmp_path / "attacker-file"
    target.write_text("attacker")
    if corruption == "unknown":
        (partial / "unknown").write_text("unexpected")
    elif corruption == "control":
        (partial / "control.json").unlink()
        (partial / "control.json").symlink_to(target)
    elif corruption == "result":
        result = next((partial / "results").iterdir())
        result.unlink()
        result.symlink_to(target)
    else:
        (partial / "last-failure.json").unlink()
        (partial / "last-failure.json").symlink_to(target)

    with pytest.raises(ValueError):
        run_maeda_sweep_bundle(plan, tmp_path, resume=True)


@pytest.mark.parametrize("entry", ["final", "lock"])
def test_final_and_lock_symlinks_are_refused(
    tmp_path: Path,
    plan_and_results: tuple[MaedaSweepPlan, dict[str, SimulationDatum]],
    entry: str,
) -> None:
    plan, _ = plan_and_results
    target = tmp_path / "attacker-file"
    target.write_text("attacker")
    name = (
        f"maeda-{plan.plan_sha256}.bundle.json"
        if entry == "final"
        else f".maeda-{plan.plan_sha256}.lock"
    )
    (tmp_path / name).symlink_to(target)

    with pytest.raises((FileExistsError, OSError)):
        run_maeda_sweep_bundle(plan, tmp_path)


def test_concurrent_lock_is_nonblocking(
    tmp_path: Path,
    plan_and_results: tuple[MaedaSweepPlan, dict[str, SimulationDatum]],
) -> None:
    plan, _ = plan_and_results
    lock = tmp_path / f".maeda-{plan.plan_sha256}.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(BlockingIOError):
            run_maeda_sweep_bundle(plan, tmp_path)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@pytest.mark.parametrize(
    "raw, message",
    [
        (b"", "invalid size"),
        (b"{", "strict JSON"),
        (b"[]", "one JSON object"),
        (b'{"x":NaN}', "non-finite JSON constant"),
    ],
)
def test_inspector_rejects_malformed_json_classes(
    tmp_path: Path,
    plan_and_results: tuple[MaedaSweepPlan, dict[str, SimulationDatum]],
    raw: bytes,
    message: str,
) -> None:
    plan, _ = plan_and_results
    path = tmp_path / "malformed.json"
    path.write_bytes(raw)

    with pytest.raises(ValueError, match=message):
        inspect_maeda_sweep_bundle(path, expected_plan=plan)


def test_public_bundle_types_reject_invalid_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan_and_results: tuple[MaedaSweepPlan, dict[str, SimulationDatum]],
) -> None:
    plan, results = plan_and_results
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        StoredSimulationDatumRef(*(["not-a-hash"] * 8))

    _install_executor(monkeypatch, results, [])
    path = run_maeda_sweep_bundle(plan, tmp_path)
    bundle = inspect_maeda_sweep_bundle(path, expected_plan=plan)
    with pytest.raises(ValueError, match="immutable typed tuple"):
        replace(bundle, results=[*bundle.results])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one result"):
        replace(bundle, results=())
    with pytest.raises(ValueError, match="claim boundary must be explicit"):
        replace(bundle, claim_boundary=" ")
    with pytest.raises(ValueError, match="claim boundary is unsupported"):
        replace(bundle, claim_boundary="fabricated")
    with pytest.raises(ValueError, match="bundle identity hash"):
        replace(bundle, bundle_sha256="0" * 64)


@pytest.mark.parametrize(
    "plan, output",
    [(object(), Path("unused")), (None, Path("unused")), ("plan", "output")],
)
def test_runner_rejects_untyped_public_inputs(plan: object, output: object) -> None:
    with pytest.raises(ValueError, match=r"MaedaSweepPlan|pathlib\.Path"):
        run_maeda_sweep_bundle(plan, output)  # type: ignore[arg-type]


def test_inspector_rejects_untyped_public_inputs(
    tmp_path: Path,
    plan_and_results: tuple[MaedaSweepPlan, dict[str, SimulationDatum]],
) -> None:
    plan, _ = plan_and_results
    with pytest.raises(ValueError, match=r"pathlib\.Path"):
        inspect_maeda_sweep_bundle("bundle.json", expected_plan=plan)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expected MaedaSweepPlan"):
        inspect_maeda_sweep_bundle(tmp_path / "missing", expected_plan=object())  # type: ignore[arg-type]


def test_bundle_constructor_rejects_empty_results() -> None:
    with pytest.raises(ValueError, match="at least one result"):
        MaedaSweepBundle(
            plan_sha256="0" * 64,
            specification_sha256="0" * 64,
            oracle_artifact_sha256="0" * 64,
            results=(),
            results_sha256="0" * 64,
            bundle_sha256="0" * 64,
            claim_boundary="irrelevant before empty-result rejection",
        )
