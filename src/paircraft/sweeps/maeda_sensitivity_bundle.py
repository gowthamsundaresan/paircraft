"""Authenticated persistence for the complete hypothetical Maeda sensitivity plan."""

from __future__ import annotations

import fcntl
import os
import re
import stat
import tempfile
from contextlib import suppress
from dataclasses import InitVar, asdict, dataclass
from hashlib import sha256
from pathlib import Path
from secrets import token_hex
from typing import cast

from paircraft.oracle.maeda_sensitivity import execute_maeda_sensitivity_oracle_plan
from paircraft.oracle.maeda_sensitivity import (
    iter_maeda_sensitivity_oracle_jobs as iter_maeda_sensitivity_oracle_jobs,
)
from paircraft.oracle.simulation import SimulationDatum, simulation_datum_document
from paircraft.sweeps.maeda_sensitivity import (
    MaedaSensitivityJob,
    MaedaSensitivityPlan,
    maeda_sensitivity_plan_document,
)
from paircraft.sweeps.storage import (
    DirectoryIdentity,
    PublishedDurabilityError,
    canonical_json_bytes,
    canonical_sha256,
    directory_identity,
    fsync_directory,
    load_canonical_json_bytes,
    lstat_directory,
    operational_write,
    read_canonical_json,
    read_regular,
    remove_authenticated_directory,
    remove_tree_at,
    require_directory_identity,
    require_locked_path,
    unlink_regular,
)
from paircraft.sweeps.storage import acquire_lock as acquire_lock
from paircraft.sweeps.storage import atomic_link_bytes as atomic_link_bytes

_SCHEMA = "paircraft-maeda-sensitivity-bundle-v1"
_AUTHENTICATED_CONSTRUCTION = object()
_CLAIM_BOUNDARY = (
    "Complete deterministic simulator-only hypothetical OAT bundle; computational completeness "
    "and exact canonical replay only, not uncertainty, ranking, training data, mutation effects, "
    "experimental calibration, biological variability, or prospective evidence."
)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    return cast(dict[str, object], value)


def _sequence(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return cast(list[object], value)


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class StoredSensitivityDatumRef:
    """Authenticated stored identity retaining both scientific grouping levels."""

    job_sha256: str
    document_sha256: str
    datum_sha256: str
    query_sha256: str
    prediction_sha256: str
    configuration_sha256: str
    execution_sha256: str
    parameter_family_sha256: str
    design_dependency_sha256: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.job_sha256, "job sha256"),
            (self.document_sha256, "document sha256"),
            (self.datum_sha256, "datum sha256"),
            (self.query_sha256, "query sha256"),
            (self.prediction_sha256, "prediction sha256"),
            (self.configuration_sha256, "configuration sha256"),
            (self.execution_sha256, "execution sha256"),
            (self.parameter_family_sha256, "parameter-family sha256"),
            (self.design_dependency_sha256, "design-dependency sha256"),
        ):
            _digest(value, label)


@dataclass(frozen=True, slots=True)
class MaedaSensitivityBundle:
    """Identity of one fully replay-authenticated sensitivity bundle."""

    plan_sha256: str
    specification_sha256: str
    oracle_artifact_sha256: str
    results: tuple[StoredSensitivityDatumRef, ...]
    results_sha256: str
    bundle_sha256: str
    _construction_key: InitVar[object]
    claim_boundary: str = _CLAIM_BOUNDARY

    def __post_init__(self, _construction_key: object) -> None:
        if _construction_key is not _AUTHENTICATED_CONSTRUCTION:
            raise ValueError("sensitivity bundle requires exact replay authentication")
        for value, label in (
            (self.plan_sha256, "plan sha256"),
            (self.specification_sha256, "specification sha256"),
            (self.oracle_artifact_sha256, "oracle artifact sha256"),
            (self.results_sha256, "results sha256"),
            (self.bundle_sha256, "bundle sha256"),
        ):
            _digest(value, label)
        if len(self.results) != 204 or any(
            not isinstance(item, StoredSensitivityDatumRef) for item in self.results
        ):
            raise ValueError("sensitivity bundle requires exactly 204 typed results")
        if len({item.parameter_family_sha256 for item in self.results}) != 34:
            raise ValueError("sensitivity bundle requires exactly 34 parameter families")
        if len({item.design_dependency_sha256 for item in self.results}) != 2:
            raise ValueError("sensitivity bundle requires exactly two design dependencies")
        if self.claim_boundary != _CLAIM_BOUNDARY:
            raise ValueError("sensitivity bundle claim boundary is unsupported")
        expected = canonical_sha256(
            {
                "schema_version": _SCHEMA,
                "status": "complete",
                "plan_sha256": self.plan_sha256,
                "specification_sha256": self.specification_sha256,
                "oracle_artifact_sha256": self.oracle_artifact_sha256,
                "results_sha256": self.results_sha256,
                "claim_boundary": self.claim_boundary,
            }
        )
        if self.bundle_sha256 != expected:
            raise ValueError("sensitivity bundle identity hash is invalid")


def _lineage_sha(document: dict[str, object], artifact_id: str) -> str:
    prediction = _mapping(document.get("prediction"), "simulation prediction")
    lineage = _sequence(prediction.get("lineage"), "prediction lineage")
    matches = tuple(
        _mapping(item, "lineage artifact")
        for item in lineage
        if _mapping(item, "lineage artifact").get("artifact_id") == artifact_id
    )
    if len(matches) != 1:
        raise ValueError(f"prediction requires exactly one {artifact_id} lineage artifact")
    return _digest(matches[0].get("sha256"), f"{artifact_id} sha256")


def _reference(
    job: MaedaSensitivityJob,
    document: dict[str, object],
) -> StoredSensitivityDatumRef:
    hashes = _mapping(document.get("hashes"), "simulation datum hashes")
    parameter_family = _lineage_sha(document, "hypothetical-parameter-family")
    design_dependency = _lineage_sha(document, "sensitivity-design-dependency-group")
    planned_job = _lineage_sha(document, "sensitivity-plan-job")
    if (
        parameter_family != job.parameter_family_sha256
        or design_dependency != job.design_dependency_sha256
        or planned_job != job.job_sha256
    ):
        raise ValueError("stored datum contradicts its planned scientific identities")
    return StoredSensitivityDatumRef(
        job_sha256=job.job_sha256,
        document_sha256=sha256(canonical_json_bytes(document)).hexdigest(),
        datum_sha256=_digest(hashes.get("datum_sha256"), "datum sha256"),
        query_sha256=_digest(hashes.get("query_sha256"), "query sha256"),
        prediction_sha256=_digest(hashes.get("prediction_sha256"), "prediction sha256"),
        configuration_sha256=_digest(hashes.get("configuration_sha256"), "configuration sha256"),
        execution_sha256=_digest(hashes.get("execution_sha256"), "execution sha256"),
        parameter_family_sha256=parameter_family,
        design_dependency_sha256=design_dependency,
    )


def _record(job: MaedaSensitivityJob, datum: SimulationDatum) -> dict[str, object]:
    document = simulation_datum_document(datum)
    return {"reference": asdict(_reference(job, document)), "simulation_datum": document}


def _bundle_document(
    plan: MaedaSensitivityPlan,
    records: list[dict[str, object]],
) -> dict[str, object]:
    oracle_artifacts = {
        _mapping(
            _mapping(
                _mapping(record["simulation_datum"], "simulation datum")["prediction"],
                "prediction",
            )["oracle"],
            "oracle",
        )["artifact_sha256"]
        for record in records
    }
    if len(oracle_artifacts) != 1:
        raise ValueError("all sensitivity results must share one oracle artifact")
    oracle_artifact = _digest(oracle_artifacts.pop(), "oracle artifact sha256")
    results_hash = canonical_sha256(records)
    identity = {
        "schema_version": _SCHEMA,
        "status": "complete",
        "plan_sha256": plan.plan_sha256,
        "specification_sha256": plan.specification.specification_sha256,
        "oracle_artifact_sha256": oracle_artifact,
        "results_sha256": results_hash,
        "claim_boundary": _CLAIM_BOUNDARY,
    }
    return {
        **identity,
        "plan": maeda_sensitivity_plan_document(plan),
        "results": records,
        "hashes": {
            "results_sha256": results_hash,
            "bundle_sha256": canonical_sha256(identity),
        },
    }


def _paths(plan: MaedaSensitivityPlan, root: Path) -> tuple[Path, Path, Path]:
    stem = f"maeda-sensitivity-{plan.plan_sha256}"
    return (
        root / f"{stem}.bundle.json",
        root / f".{stem}.partial",
        root / f".{stem}.lock",
    )


def _cleanup_root_leftovers(
    output_root: Path,
    final: Path,
    partial: Path,
    *,
    expected_root_identity: DirectoryIdentity,
) -> None:
    temporary = re.compile(rf"{re.escape(f'.tmp-{final.name}-')}[0-9a-f]{{16}}")
    initialization = re.compile(rf"{re.escape(f'.init-{partial.name}-')}[A-Za-z0-9_]{{6,}}")
    root_fd = os.open(
        output_root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        root = os.fstat(root_fd)
        if (root.st_dev, root.st_ino) != (
            expected_root_identity.device,
            expected_root_identity.inode,
        ):
            raise ValueError("sensitivity output root was replaced before cleanup")
        for name in os.listdir(root_fd):
            metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            if temporary.fullmatch(name):
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValueError("publication temporary is not a regular file")
                quarantine = f".cleanup-{name}-{token_hex(8)}"
                os.rename(name, quarantine, src_dir_fd=root_fd, dst_dir_fd=root_fd)
                os.fsync(root_fd)
                moved = os.stat(quarantine, dir_fd=root_fd, follow_symlinks=False)
                if (moved.st_dev, moved.st_ino) != (metadata.st_dev, metadata.st_ino):
                    raise ValueError("quarantined publication temporary was replaced")
                os.unlink(quarantine, dir_fd=root_fd)
                os.fsync(root_fd)
            elif initialization.fullmatch(name):
                if not stat.S_ISDIR(metadata.st_mode):
                    raise ValueError("initialization temporary is not a directory")
                remove_tree_at(
                    root_fd,
                    name,
                    expected_identity=DirectoryIdentity(metadata.st_dev, metadata.st_ino),
                )
    finally:
        os.close(root_fd)


def _real_directory(path: Path, label: str) -> None:
    lstat_directory(path, label)


def _prepare_workspace(
    plan: MaedaSensitivityPlan,
    partial: Path,
    *,
    resume: bool,
) -> Path:
    control = partial / "control.json"
    if partial.exists() or partial.is_symlink():
        _real_directory(partial, "sensitivity partial workspace")
        if not resume:
            raise FileExistsError(f"partial sensitivity workspace already exists: {partial}")
        allowed = {"control.json", "results", "last-failure.json"}
        if {item.name for item in partial.iterdir()} - allowed:
            raise ValueError("partial sensitivity workspace contains unexpected entries")
        if read_regular(control, "sensitivity partial control") != canonical_json_bytes(
            maeda_sensitivity_plan_document(plan)
        ):
            raise ValueError("partial sensitivity workspace belongs to another plan")
        results = partial / "results"
        _real_directory(results, "sensitivity partial results")
        failure = partial / "last-failure.json"
        if failure.exists() or failure.is_symlink():
            receipt = read_canonical_json(failure, label="sensitivity failure receipt")
            if set(receipt) != {"job_sha256", "category", "message"}:
                raise ValueError("sensitivity failure receipt has unexpected fields")
            if (
                receipt["job_sha256"] not in {job.job_sha256 for job in plan.jobs}
                or receipt["category"]
                not in {"numerical_failure", "validation_failure", "execution_failure"}
                or receipt["message"] != "Job failed before authenticated result publication."
            ):
                raise ValueError("sensitivity failure receipt has invalid semantics")
            unlink_regular(
                partial,
                failure.name,
                "sensitivity failure receipt",
                expected_parent_identity=directory_identity(
                    partial, "sensitivity partial workspace"
                ),
            )
        return results
    staging = Path(tempfile.mkdtemp(prefix=f".init-{partial.name}-", dir=partial.parent))
    staging_identity = directory_identity(staging, "sensitivity initialization workspace")
    try:
        results = staging / "results"
        results.mkdir(mode=0o700)
        atomic_link_bytes(
            staging,
            staging / "control.json",
            canonical_json_bytes(maeda_sensitivity_plan_document(plan)),
            expected_identity=staging_identity,
        )
        fsync_directory(results)
        fsync_directory(staging)
        os.rename(staging, partial)
        fsync_directory(partial.parent)
    finally:
        root_fd = os.open(
            partial.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            with suppress(FileNotFoundError):
                remove_tree_at(
                    root_fd,
                    staging.name,
                    expected_identity=staging_identity,
                )
        finally:
            os.close(root_fd)
    return partial / "results"


def _partial_records(
    plan: MaedaSensitivityPlan,
    results: Path,
    *,
    expected_identity: DirectoryIdentity,
) -> dict[str, dict[str, object]]:
    require_directory_identity(results, expected_identity, "sensitivity partial results")
    jobs = {job.job_sha256: job for job in plan.jobs}
    loaded: dict[str, dict[str, object]] = {}
    for path in results.iterdir():
        if path.name.startswith(".tmp-"):
            unlink_regular(
                results,
                path.name,
                "interrupted sensitivity temporary",
                expected_parent_identity=expected_identity,
            )
            continue
        if not path.name.endswith(".json") or path.stem not in jobs:
            raise ValueError("partial sensitivity results contain an unexpected filename")
        record = load_canonical_json_bytes(
            read_regular(path, "partial sensitivity result"),
            label="partial sensitivity result",
        )
        if set(record) != {"reference", "simulation_datum"}:
            raise ValueError("partial sensitivity result has unexpected fields")
        loaded[path.stem] = record
    ordered_jobs = tuple(job for job in plan.jobs if job.job_sha256 in loaded)
    completed: dict[str, dict[str, object]] = {}
    for job, datum in zip(
        ordered_jobs,
        iter_maeda_sensitivity_oracle_jobs(plan, ordered_jobs),
        strict=True,
    ):
        record = loaded[job.job_sha256]
        expected = simulation_datum_document(datum)
        stored = _mapping(record.get("simulation_datum"), "simulation datum")
        if stored != expected:
            raise ValueError("stored datum does not equal exact canonical sensitivity replay")
        reference = _reference(job, stored)
        if record.get("reference") != asdict(reference):
            raise ValueError("stored sensitivity reference does not authenticate its datum")
        completed[job.job_sha256] = record
    require_directory_identity(results, expected_identity, "sensitivity partial results")
    return completed


def run_maeda_sensitivity_bundle(
    plan: MaedaSensitivityPlan,
    output_root: Path,
    *,
    resume: bool = False,
) -> Path:
    """Execute with per-job checkpoints and publish one complete canonical bundle."""

    if not isinstance(plan, MaedaSensitivityPlan):
        raise ValueError("sensitivity bundle execution requires a typed plan")
    if not isinstance(output_root, Path):
        raise ValueError("output root must be a pathlib.Path")
    if not output_root.exists():
        output_root.mkdir(parents=True, mode=0o700)
    _real_directory(output_root, "sensitivity output root")
    fsync_directory(output_root)
    root_identity = directory_identity(output_root, "sensitivity output root")
    output, partial, lock = _paths(plan, output_root)
    descriptor = acquire_lock(lock, expected_parent_identity=root_identity)
    try:
        require_locked_path(lock, descriptor, expected_parent_identity=root_identity)
        require_directory_identity(output_root, root_identity, "sensitivity output root")
        if resume:
            _cleanup_root_leftovers(
                output_root,
                output,
                partial,
                expected_root_identity=root_identity,
            )
            require_directory_identity(output_root, root_identity, "sensitivity output root")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"final sensitivity bundle already exists: {output}")
        results = _prepare_workspace(plan, partial, resume=resume)
        require_locked_path(lock, descriptor, expected_parent_identity=root_identity)
        partial_identity = directory_identity(partial, "sensitivity partial workspace")
        results_identity = directory_identity(results, "sensitivity partial results")
        completed = _partial_records(
            plan,
            results,
            expected_identity=results_identity,
        )
        pending = tuple(job for job in plan.jobs if job.job_sha256 not in completed)
        try:
            for job, datum in zip(
                pending,
                iter_maeda_sensitivity_oracle_jobs(plan, pending),
                strict=True,
            ):
                record = _record(job, datum)
                require_directory_identity(results, results_identity, "sensitivity partial results")
                atomic_link_bytes(
                    results,
                    results / f"{job.job_sha256}.json",
                    canonical_json_bytes(record),
                    expected_identity=results_identity,
                )
                completed[job.job_sha256] = record
        except BaseException as error:
            if isinstance(error, PublishedDurabilityError):
                raise
            require_directory_identity(partial, partial_identity, "sensitivity partial workspace")
            failed_job = next(
                (job for job in pending if job.job_sha256 not in completed),
                pending[-1],
            )
            category = (
                "numerical_failure"
                if isinstance(error, ArithmeticError)
                else "validation_failure"
                if isinstance(error, ValueError)
                else "execution_failure"
            )
            operational_write(
                partial / "last-failure.json",
                {
                    "job_sha256": failed_job.job_sha256,
                    "category": category,
                    "message": "Job failed before authenticated result publication.",
                },
                expected_parent_identity=partial_identity,
            )
            raise
        if set(completed) != {job.job_sha256 for job in plan.jobs}:
            raise ValueError("partial sensitivity workspace is not complete")
        ordered = [completed[job.job_sha256] for job in plan.jobs]
        require_directory_identity(output_root, root_identity, "sensitivity output root")
        require_locked_path(lock, descriptor, expected_parent_identity=root_identity)
        atomic_link_bytes(
            output_root,
            output,
            canonical_json_bytes(_bundle_document(plan, ordered)),
            expected_identity=root_identity,
        )
        with suppress(OSError, ValueError):
            require_directory_identity(partial, partial_identity, "sensitivity partial workspace")
            remove_authenticated_directory(
                output_root,
                partial,
                partial_identity,
                expected_root_identity=root_identity,
            )
        return output
    finally:
        with suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _inspect_document(
    document: dict[str, object],
    *,
    expected_plan: MaedaSensitivityPlan,
) -> tuple[MaedaSensitivityBundle, tuple[dict[str, object], ...]]:
    expected_keys = {
        "schema_version",
        "status",
        "plan_sha256",
        "specification_sha256",
        "oracle_artifact_sha256",
        "results_sha256",
        "claim_boundary",
        "plan",
        "results",
        "hashes",
    }
    if set(document) != expected_keys:
        raise ValueError("sensitivity bundle has unexpected or missing fields")
    if document["schema_version"] != _SCHEMA or document["status"] != "complete":
        raise ValueError("sensitivity bundle schema or status is unsupported")
    if document["plan"] != maeda_sensitivity_plan_document(expected_plan):
        raise ValueError("sensitivity bundle does not contain the exact expected plan")
    records = _sequence(document["results"], "sensitivity bundle results")
    if len(records) != len(expected_plan.jobs):
        raise ValueError("sensitivity bundle result count contradicts the plan")
    references: list[StoredSensitivityDatumRef] = []
    documents: list[dict[str, object]] = []
    replay_oracle_artifacts: set[str] = set()
    replay = execute_maeda_sensitivity_oracle_plan(expected_plan)
    for job, expected_datum, raw_record in zip(expected_plan.jobs, replay, records, strict=True):
        record = _mapping(raw_record, "sensitivity result record")
        if set(record) != {"reference", "simulation_datum"}:
            raise ValueError("sensitivity result record has unexpected fields")
        stored = _mapping(record["simulation_datum"], "simulation datum")
        expected = simulation_datum_document(expected_datum)
        if stored != expected:
            raise ValueError("stored datum does not equal exact canonical sensitivity replay")
        oracle = _mapping(
            _mapping(expected["prediction"], "replayed prediction")["oracle"],
            "replayed oracle",
        )
        replay_oracle_artifacts.add(
            _digest(oracle.get("artifact_sha256"), "replayed oracle artifact sha256")
        )
        reference = _reference(job, stored)
        if record["reference"] != asdict(reference):
            raise ValueError("stored sensitivity reference does not authenticate its datum")
        references.append(reference)
        documents.append(stored)
    if len(replay_oracle_artifacts) != 1 or document["oracle_artifact_sha256"] != next(
        iter(replay_oracle_artifacts)
    ):
        raise ValueError("bundle oracle artifact does not match exact replay")
    results_hash = canonical_sha256(records)
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
    expected_hashes = {
        "results_sha256": results_hash,
        "bundle_sha256": canonical_sha256(identity),
    }
    if document["results_sha256"] != results_hash or document["hashes"] != expected_hashes:
        raise ValueError("sensitivity bundle content hashes are invalid")
    bundle = MaedaSensitivityBundle(
        plan_sha256=_digest(document["plan_sha256"], "plan sha256"),
        specification_sha256=_digest(document["specification_sha256"], "specification sha256"),
        oracle_artifact_sha256=_digest(
            document["oracle_artifact_sha256"], "oracle artifact sha256"
        ),
        results=tuple(references),
        results_sha256=results_hash,
        bundle_sha256=expected_hashes["bundle_sha256"],
        _construction_key=_AUTHENTICATED_CONSTRUCTION,
    )
    if (
        bundle.plan_sha256 != expected_plan.plan_sha256
        or bundle.specification_sha256 != expected_plan.specification.specification_sha256
    ):
        raise ValueError("sensitivity bundle identity contradicts the expected plan")
    return bundle, tuple(documents)


def load_authenticated_maeda_sensitivity_bundle(
    path: Path,
    *,
    expected_plan: MaedaSensitivityPlan,
) -> tuple[MaedaSensitivityBundle, tuple[dict[str, object], ...]]:
    """Strictly load and exactly replay every result in one sensitivity bundle."""

    if not isinstance(expected_plan, MaedaSensitivityPlan):
        raise ValueError("sensitivity bundle inspection requires a typed expected plan")
    return _inspect_document(
        read_canonical_json(path, label="sensitivity bundle"),
        expected_plan=expected_plan,
    )


def inspect_maeda_sensitivity_bundle(
    path: Path,
    *,
    expected_plan: MaedaSensitivityPlan,
) -> MaedaSensitivityBundle:
    """Return bundle identity only after strict parsing and exact oracle replay."""

    bundle, _ = load_authenticated_maeda_sensitivity_bundle(path, expected_plan=expected_plan)
    return bundle
