"""Crash-safe, content-addressed persistence for executed Maeda sweep plans."""

from __future__ import annotations

import fcntl
import json
import os
import re
import stat
import tempfile
from contextlib import suppress
from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
from math import isfinite
from pathlib import Path
from secrets import token_hex
from typing import Final, cast

from paircraft.oracle.maeda2012 import (
    Maeda2012SimulatorOracle,
    maeda_2012_query,
)
from paircraft.oracle.model import (
    ObservableKind,
    OraclePrediction,
)
from paircraft.oracle.simulation import (
    SimulationDatum,
    simulation_datum_json,
)
from paircraft.spin import maeda_2012_specification
from paircraft.sweeps.maeda import (
    MaedaSweepJob,
    MaedaSweepPlan,
    execute_maeda_sweep_job,
    maeda_sweep_plan_document,
    maeda_sweep_plan_json,
    validate_maeda_job_result,
)
from paircraft.sweeps.storage import (
    PublishedDurabilityError,
    canonical_json_bytes,
    canonical_sha256,
    load_canonical_json_bytes,
)

_BUNDLE_SCHEMA: Final = "paircraft-maeda-sweep-bundle-v1"
_DATUM_SCHEMA: Final = "paircraft-simulation-datum-v1"
_MAX_JSON_BYTES: Final = 16 * 1024 * 1024
_CLAIM_BOUNDARY: Final = (
    "Complete deterministic simulator reference bundle; computational completeness and numerical "
    "convergence only, not experimental calibration, predictive validation, training data, "
    "biological variability, mutation effects, or MagLOV evidence."
)
_PREDICTION_CLAIM_BOUNDARY: Final = (
    "simulator-only reproduction of a published reduced point-estimate model; not "
    "experimental calibration, sequence prediction, exact Figure 6 refitting, or "
    "sub-Earth-field validation"
)
_MODEL_ID: Final = "maeda-2012-five-nucleus-haberkorn-std-rp2-v1"
_MODEL_FAMILY: Final = "haberkorn_static_pair_with_st_dephasing"
_SOLVER_NAME: Final = "energy-basis Jacobi-preconditioned GMRES"
_RANDOMNESS_NOTE: Final = "Deterministic linear-algebra solve; no random sampling is used."
_EXPECTED_SETTINGS: Final = [
    {
        "name": "maximum_restart_cycles",
        "value": "50",
        "unit": "1",
        "definition": "Maximum number of restarted GMRES cycles.",
    },
    {
        "name": "relative_tolerance",
        "value": "1e-08",
        "unit": "1",
        "definition": "Requested true relative residual tolerance.",
    },
    {
        "name": "restart",
        "value": "100",
        "unit": "1",
        "definition": "Krylov subspace dimension before GMRES restart.",
    },
]


def _canonical_bytes(payload: object) -> bytes:
    return canonical_json_bytes(payload)


def _canonical_sha256(payload: object) -> str:
    return canonical_sha256(payload)


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _normalized(value: object) -> object:
    return json.loads(_canonical_bytes(value))


def _load_canonical_json(raw: bytes, *, label: str) -> dict[str, object]:
    return load_canonical_json_bytes(raw, label=label, maximum_bytes=_MAX_JSON_BYTES)


def _require_exact_keys(value: dict[str, object], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} has unexpected or missing fields")


def _dict(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return cast(list[object], value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class _DirectoryIdentity:
    device: int
    inode: int


def _directory_identity(path: Path, label: str) -> _DirectoryIdentity:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise ValueError(f"{label} must be an anchored real directory") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{label} must be an anchored real directory")
        return _DirectoryIdentity(metadata.st_dev, metadata.st_ino)
    finally:
        os.close(descriptor)


def _require_directory_identity(
    path: Path,
    expected: _DirectoryIdentity,
    label: str,
) -> None:
    if _directory_identity(path, label) != expected:
        raise ValueError(f"{label} was replaced during bundle execution")


def _lstat_directory(path: Path, label: str) -> None:
    mode = path.lstat().st_mode
    if not stat.S_ISDIR(mode):
        raise ValueError(f"{label} must be a real non-symlink directory")


def _read_regular(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} must be a readable regular non-symlink file") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file")
        if metadata.st_size <= 0 or metadata.st_size > _MAX_JSON_BYTES:
            raise ValueError(f"{label} has an invalid size")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError(f"{label} changed while it was read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError(f"{label} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _unlink_regular(
    directory: Path,
    name: str,
    label: str,
    *,
    expected_parent_identity: _DirectoryIdentity,
) -> None:
    """Unlink one exact regular entry without following a substituted symlink."""

    directory_fd = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor: int | None = None
    try:
        parent = os.fstat(directory_fd)
        if (parent.st_dev, parent.st_ino) != (
            expected_parent_identity.device,
            expected_parent_identity.inode,
        ):
            raise ValueError(f"{label} parent directory was replaced")
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        opened = os.fstat(descriptor)
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise ValueError(f"{label} must be a regular non-symlink file")
        os.unlink(name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    except OSError as error:
        raise ValueError(f"{label} must be a removable regular non-symlink file") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)


def _remove_tree_at(
    parent_fd: int,
    name: str,
    *,
    expected_identity: _DirectoryIdentity | None = None,
) -> None:
    """Recursively unlink a tree without following pathnames or symlinks."""

    child_fd = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_fd,
    )
    try:
        opened = os.fstat(child_fd)
        opened_pair = (opened.st_dev, opened.st_ino)
        if expected_identity is not None and opened_pair != (
            expected_identity.device,
            expected_identity.inode,
        ):
            raise ValueError("recursive cleanup target was replaced")
        for child_name in os.listdir(child_fd):
            child = os.stat(child_name, dir_fd=child_fd, follow_symlinks=False)
            child_identity = _DirectoryIdentity(child.st_dev, child.st_ino)
            if stat.S_ISDIR(child.st_mode):
                _remove_tree_at(
                    child_fd,
                    child_name,
                    expected_identity=child_identity,
                )
            elif stat.S_ISREG(child.st_mode):
                regular_fd = os.open(
                    child_name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=child_fd,
                )
                try:
                    opened_regular = os.fstat(regular_fd)
                    current_regular = os.stat(
                        child_name,
                        dir_fd=child_fd,
                        follow_symlinks=False,
                    )
                    if (
                        not stat.S_ISREG(opened_regular.st_mode)
                        or (opened_regular.st_dev, opened_regular.st_ino)
                        != (child.st_dev, child.st_ino)
                        or (current_regular.st_dev, current_regular.st_ino)
                        != (child.st_dev, child.st_ino)
                    ):
                        raise ValueError("recursive cleanup regular file was replaced")
                    os.unlink(child_name, dir_fd=child_fd)
                finally:
                    os.close(regular_fd)
            elif stat.S_ISLNK(child.st_mode):
                current_link = os.stat(
                    child_name,
                    dir_fd=child_fd,
                    follow_symlinks=False,
                )
                if (current_link.st_dev, current_link.st_ino) != (
                    child.st_dev,
                    child.st_ino,
                ):
                    raise ValueError("recursive cleanup symlink was replaced")
                os.unlink(child_name, dir_fd=child_fd)
            else:
                raise ValueError("recursive cleanup encountered an unsafe file type")
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != opened_pair:
            raise ValueError("recursive cleanup target was replaced")
    finally:
        os.close(child_fd)
    os.rmdir(name, dir_fd=parent_fd)
    os.fsync(parent_fd)


def _atomic_link_bytes(
    directory: Path,
    destination: Path,
    payload: bytes,
    *,
    expected_identity: _DirectoryIdentity,
) -> None:
    _load_canonical_json(payload, label="publication payload")
    if destination.parent != directory:
        raise ValueError("atomic destination must be directly inside its anchored directory")
    directory_fd = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    opened_directory = os.fstat(directory_fd)
    if (opened_directory.st_dev, opened_directory.st_ino) != (
        expected_identity.device,
        expected_identity.inode,
    ):
        os.close(directory_fd)
        raise ValueError("atomic destination directory was replaced before publication")
    temporary_name = f".tmp-{destination.name}-{token_hex(8)}"
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(
            temporary_name,
            destination.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        # The no-replace hard link is the publication commit point. A later
        # directory-fsync failure cannot make an extant final artifact
        # unpublished, so cleanup/durability deepening is best effort here.
        try:
            os.fsync(directory_fd)
        except OSError as error:
            raise PublishedDurabilityError(
                f"published {destination}; directory fsync failed"
            ) from error
    finally:
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
            os.fsync(directory_fd)
        except FileNotFoundError:
            pass
        except OSError:
            # The hardlink is the commit point. A leftover temporary is safely
            # recognizable and is recovered under the plan lock on resume.
            pass
        os.close(directory_fd)


def _operational_write(
    path: Path,
    payload: object,
    *,
    expected_parent_identity: _DirectoryIdentity,
) -> None:
    raw = _canonical_bytes(payload)
    parent_fd = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    parent = os.fstat(parent_fd)
    if (parent.st_dev, parent.st_ino) != (
        expected_parent_identity.device,
        expected_parent_identity.inode,
    ):
        os.close(parent_fd)
        raise ValueError("operational receipt parent directory was replaced")
    temporary_name = f".failure-{path.name}-{token_hex(8)}"
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.fsync(parent_fd)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=parent_fd)
        os.close(parent_fd)


@dataclass(frozen=True, slots=True)
class StoredSimulationDatumRef:
    """Authenticated identity of one canonical stored simulation datum."""

    job_sha256: str
    document_sha256: str
    datum_sha256: str
    query_sha256: str
    prediction_sha256: str
    configuration_sha256: str
    execution_sha256: str
    dependency_group_sha256: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.job_sha256, "job sha256"),
            (self.document_sha256, "document sha256"),
            (self.datum_sha256, "datum sha256"),
            (self.query_sha256, "query sha256"),
            (self.prediction_sha256, "prediction sha256"),
            (self.configuration_sha256, "configuration sha256"),
            (self.execution_sha256, "execution sha256"),
            (self.dependency_group_sha256, "dependency-group sha256"),
        ):
            _require_sha256(value, label)


@dataclass(frozen=True, slots=True)
class MaedaSweepBundle:
    """Validated identity of one complete, immutable sweep bundle."""

    plan_sha256: str
    specification_sha256: str
    oracle_artifact_sha256: str
    results: tuple[StoredSimulationDatumRef, ...]
    results_sha256: str
    bundle_sha256: str
    claim_boundary: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.plan_sha256, "plan sha256"),
            (self.specification_sha256, "specification sha256"),
            (self.oracle_artifact_sha256, "oracle artifact sha256"),
            (self.results_sha256, "results sha256"),
            (self.bundle_sha256, "bundle sha256"),
        ):
            _require_sha256(value, label)
        if not isinstance(self.results, tuple) or any(
            not isinstance(item, StoredSimulationDatumRef) for item in self.results
        ):
            raise ValueError("bundle result references must be an immutable typed tuple")
        if not self.results:
            raise ValueError("a complete bundle requires at least one result")
        if not self.claim_boundary.strip():
            raise ValueError("bundle claim boundary must be explicit")
        if self.claim_boundary != _CLAIM_BOUNDARY:
            raise ValueError("bundle claim boundary is unsupported")
        expected_bundle_hash = _canonical_sha256(
            {
                "schema_version": _BUNDLE_SCHEMA,
                "status": "complete",
                "plan_sha256": self.plan_sha256,
                "specification_sha256": self.specification_sha256,
                "oracle_artifact_sha256": self.oracle_artifact_sha256,
                "results_sha256": self.results_sha256,
                "claim_boundary": self.claim_boundary,
            }
        )
        if self.bundle_sha256 != expected_bundle_hash:
            raise ValueError("bundle identity hash is invalid")


def _execution_hashes(execution: dict[str, object]) -> tuple[str, str]:
    required = {
        "model_id",
        "model_family",
        "solver_name",
        "solver_versions",
        "code_version",
        "code_artifact_sha256",
        "settings",
        "achieved_relative_residual",
        "iteration_count",
        "deterministic",
        "random_seed",
        "randomness_note",
    }
    _require_exact_keys(execution, required, "simulation execution")
    for key in ("model_id", "model_family", "solver_name", "code_version", "randomness_note"):
        _string(execution[key], f"execution {key}")
    code_artifact = _string(execution["code_artifact_sha256"], "code artifact sha256")
    _require_sha256(code_artifact, "code artifact sha256")
    if (
        execution["model_id"] != _MODEL_ID
        or execution["model_family"] != _MODEL_FAMILY
        or execution["solver_name"] != _SOLVER_NAME
        or execution["randomness_note"] != _RANDOMNESS_NOTE
    ):
        raise ValueError("stored execution has unsupported model or solver semantics")
    versions = _list(execution["solver_versions"], "solver versions")
    normalized_versions: list[tuple[str, str]] = []
    for item in versions:
        pair = _list(item, "solver version")
        if len(pair) != 2:
            raise ValueError("solver version must be a two-string pair")
        normalized_versions.append(
            (_string(pair[0], "solver dependency"), _string(pair[1], "solver version"))
        )
    if normalized_versions != sorted(set(normalized_versions)):
        raise ValueError("solver versions must be unique sorted pairs")
    if execution["settings"] != _EXPECTED_SETTINGS:
        raise ValueError("stored execution settings do not match the fixed Maeda solver")
    configuration = {
        "model_id": execution["model_id"],
        "model_family": execution["model_family"],
        "solver_name": execution["solver_name"],
        "solver_versions": execution["solver_versions"],
        "code_version": execution["code_version"],
        "code_artifact_sha256": execution["code_artifact_sha256"],
        "settings": execution["settings"],
        "deterministic": execution["deterministic"],
        "random_seed": execution["random_seed"],
        "randomness_note": execution["randomness_note"],
    }
    configuration_hash = _canonical_sha256(configuration)
    execution_hash = _canonical_sha256(
        {
            "configuration_sha256": configuration_hash,
            "achieved_relative_residual": execution["achieved_relative_residual"],
            "iteration_count": execution["iteration_count"],
        }
    )
    if execution["deterministic"] is not True or execution["random_seed"] is not None:
        raise ValueError("stored sweep execution must be deterministic without a seed")
    residual = execution["achieved_relative_residual"]
    iterations = execution["iteration_count"]
    if (
        isinstance(residual, bool)
        or not isinstance(residual, int | float)
        or not isfinite(residual)
    ):
        raise ValueError("stored execution residual must be finite")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("stored execution iteration count must be positive")
    tolerance = float(_EXPECTED_SETTINGS[1]["value"])
    if not isfinite(tolerance) or not 0 < tolerance < 1 or residual < 0 or residual > tolerance:
        raise ValueError("stored execution did not meet its recorded tolerance")
    return configuration_hash, execution_hash


def _validate_distribution(prediction: dict[str, object]) -> None:
    distribution = _dict(prediction.get("distribution"), "prediction distribution")
    _require_exact_keys(
        distribution,
        {
            "mean",
            "lower",
            "upper",
            "coverage_probability",
            "unit",
            "method",
            "epistemic_standard_uncertainty",
            "aleatoric_standard_uncertainty",
            "simulator_discrepancy_standard_uncertainty",
            "numerical_standard_uncertainty",
            "decomposition_status",
            "uncertainty_assumptions",
            "calibration_status",
            "note",
        },
        "prediction distribution",
    )
    mean = distribution.get("mean")
    lower = distribution.get("lower")
    upper = distribution.get("upper")
    execution = _dict(prediction.get("execution"), "prediction execution")
    residual = execution.get("achieved_relative_residual")
    iterations = execution.get("iteration_count")
    if isinstance(residual, bool) or not isinstance(residual, int | float):
        raise ValueError("prediction execution residual must be numeric")
    expected_method = (
        "deterministic infinite-time Maeda RP2 solve; "
        f"residual={residual:.6g}; iterations={iterations}"
    )
    if (
        isinstance(mean, bool)
        or not isinstance(mean, int | float)
        or not isfinite(mean)
        or not 0 <= mean <= 1
        or lower != mean
        or upper != mean
        or distribution.get("unit") != "1"
        or distribution.get("coverage_probability") is not None
        or distribution.get("method") != expected_method
        or distribution.get("epistemic_standard_uncertainty") is not None
        or distribution.get("aleatoric_standard_uncertainty") is not None
        or distribution.get("simulator_discrepancy_standard_uncertainty") is not None
        or distribution.get("numerical_standard_uncertainty") is not None
        or distribution.get("decomposition_status")
        != "unavailable for published point-estimate model"
        or distribution.get("uncertainty_assumptions")
        != (
            "Fitted-rate covariance is unavailable; common rate-scaling "
            "non-identifiability and model discrepancy are not propagated."
        )
        or distribution.get("calibration_status")
        != "uncalibrated simulator-only point result; no predictive coverage"
        or distribution.get("note")
        != (
            "GMRES residual is a convergence diagnostic, not predictive uncertainty. "
            "Millitesla model agreement does not validate sub-Earth-field biology."
        )
    ):
        raise ValueError("stored prediction must be an uncalibrated unitless point yield")


def _validate_applicability(prediction: dict[str, object], job: MaedaSweepJob) -> None:
    status = "boundary" if job.field_point.magnitude_tesla == 0.05 else "in_domain"
    expected = {
        "status": status,
        "reasons": [
            "Exact published point-estimate parameter set and ideal protocol are supported."
        ],
        "unsupported_dimensions": [],
        "support_evidence_ids": [
            "maeda-2012-article",
            "maeda-2012-supporting-information",
        ],
        "support_bounds": [
            {
                "dimension": "magnetic flux-density magnitude",
                "lower": 0.0,
                "upper": 0.05,
                "unit": "T",
                "definition": "Euclidean magnitude of the ideal Cartesian B vector.",
            }
        ],
        "model_family_disagreement": "not assessed; this oracle executes one model family",
    }
    if prediction.get("applicability") != expected:
        raise ValueError("stored prediction has invalid Maeda applicability semantics")


def _validate_lineage(
    prediction: dict[str, object],
    job: MaedaSweepJob,
    configuration_hash: str,
) -> None:
    specification = maeda_2012_specification(job.target.protein_system)
    parameter_ids = [
        *(item.equivalent_field.parameter_id for item in specification.hyperfine_parameters),
        specification.forward_rate.parameter_id,
        specification.back_recombination_rate.parameter_id,
        specification.st_dephasing_rate.parameter_id,
        specification.electron_g_factor.parameter_id,
        specification.electron_gyromagnetic_ratio.parameter_id,
    ]
    expected_ids = [
        f"{specification.protein_system.value}-parameter-set",
        *parameter_ids,
        "initial-nuclear-state-assumption",
        "maeda-2012-supporting-information",
        "maeda-2012-article",
        "maeda-2012-numerical-execution",
    ]
    lineage = _list(prediction.get("lineage"), "prediction lineage")
    if len(lineage) != 15:
        raise ValueError("stored prediction must retain exactly 15 Maeda lineage artifacts")
    artifact_keys = {
        "artifact_id",
        "sha256",
        "evidence_role",
        "evidence_status",
        "reuse_status",
        "membership",
        "exact_location",
        "conditions",
        "uncertainty_note",
        "source_uri",
        "source_artifact_sha256",
        "license",
    }
    artifacts = [_dict(item, "lineage artifact") for item in lineage]
    actual_ids: list[str] = []
    for artifact in artifacts:
        _require_exact_keys(artifact, artifact_keys, "lineage artifact")
        for key in artifact_keys - {"source_artifact_sha256"}:
            _string(artifact[key], f"lineage {key}")
        artifact_id = _string(artifact["artifact_id"], "lineage artifact ID")
        digest = _string(artifact["sha256"], "lineage sha256")
        _require_sha256(digest, "lineage sha256")
        source_digest = artifact["source_artifact_sha256"]
        if source_digest is not None:
            _require_sha256(
                _string(source_digest, "source artifact sha256"), "source artifact sha256"
            )
        actual_ids.append(artifact_id)
    if actual_ids != expected_ids or len(set(actual_ids)) != 15:
        raise ValueError("stored prediction lineage IDs are not the exact Maeda set")
    if artifacts[0]["sha256"] != job.target.parameter_set_sha256:
        raise ValueError("parameter-set lineage does not match the query system")
    for artifact, hyperfine in zip(artifacts[1:6], specification.hyperfine_parameters, strict=True):
        if (
            artifact["sha256"] != _canonical_sha256(asdict(hyperfine))
            or artifact["source_artifact_sha256"]
            != hyperfine.equivalent_field.source_artifact_sha256
        ):
            raise ValueError("hyperfine lineage hash is invalid")
    scalar_parameters = (
        specification.forward_rate,
        specification.back_recombination_rate,
        specification.st_dephasing_rate,
        specification.electron_g_factor,
        specification.electron_gyromagnetic_ratio,
    )
    for artifact, parameter in zip(artifacts[6:11], scalar_parameters, strict=True):
        if (
            artifact["sha256"] != _canonical_sha256(asdict(parameter))
            or artifact["source_artifact_sha256"] != parameter.source_artifact_sha256
        ):
            raise ValueError("scalar-parameter lineage hash is invalid")
    if any(artifacts[index]["source_artifact_sha256"] is not None for index in (0, 11, 14)):
        raise ValueError("internal lineage artifacts cannot claim a source artifact hash")
    if (
        artifacts[11]["sha256"]
        != sha256(specification.initial_nuclear_state_assumption.encode()).hexdigest()
    ):
        raise ValueError("initial-state lineage hash is invalid")
    supporting_hash = specification.hyperfine_parameters[0].equivalent_field.source_artifact_sha256
    article_hash = specification.back_recombination_rate.source_artifact_sha256
    if (
        artifacts[12]["sha256"] != supporting_hash
        or artifacts[12]["source_artifact_sha256"] != supporting_hash
        or artifacts[13]["sha256"] != article_hash
        or artifacts[13]["source_artifact_sha256"] != article_hash
    ):
        raise ValueError("published-source lineage hashes are invalid")
    execution = _dict(prediction["execution"], "prediction execution")
    expected_execution_hash = _canonical_sha256(
        {
            "configuration_sha256": configuration_hash,
            "residual": execution["achieved_relative_residual"],
            "iterations": execution["iteration_count"],
        }
    )
    if artifacts[14]["sha256"] != expected_execution_hash:
        raise ValueError("numerical-execution lineage hash is invalid")


@lru_cache(maxsize=256)
def _canonical_job_result(job: MaedaSweepJob) -> bytes:
    """Replay one deterministic job through the trusted canonical oracle."""

    query = maeda_2012_query(
        job.target.protein_system,
        job.field_point.vector_tesla,
        observable=ObservableKind.SECONDARY_RADICAL_YIELD,
        query_id=job.query_id,
    )
    prediction = Maeda2012SimulatorOracle().predict(query)
    if not isinstance(prediction, OraclePrediction) or prediction.execution is None:
        raise ArithmeticError("canonical Maeda replay did not produce an executable prediction")
    return simulation_datum_json(SimulationDatum(prediction, prediction.execution))


def _validate_raw_datum(
    raw: bytes,
    *,
    plan: MaedaSweepPlan,
    job: MaedaSweepJob,
) -> tuple[StoredSimulationDatumRef, dict[str, object]]:
    document = _load_canonical_json(raw, label="stored simulation datum")
    _require_exact_keys(
        document,
        {"schema_version", "evidence_origin", "membership", "prediction", "execution", "hashes"},
        "stored simulation datum",
    )
    if (
        document["schema_version"] != _DATUM_SCHEMA
        or document["evidence_origin"] != "simulation"
        or document["membership"] != "reference_only_not_training"
    ):
        raise ValueError("stored datum has an unsupported schema, origin, or membership")
    prediction = _dict(document["prediction"], "stored prediction")
    _require_exact_keys(
        prediction,
        {
            "applicability",
            "claim_boundary",
            "distribution",
            "executed_model_family",
            "executed_model_id",
            "execution",
            "lineage",
            "oracle",
            "query",
            "resolved_setup",
        },
        "stored prediction",
    )
    execution = _dict(document["execution"], "stored execution")
    hashes = _dict(document["hashes"], "stored hashes")
    _require_exact_keys(
        hashes,
        {
            "query_sha256",
            "prediction_sha256",
            "configuration_sha256",
            "execution_sha256",
            "dependency_group_sha256",
            "datum_sha256",
        },
        "stored hashes",
    )
    if prediction.get("execution") != execution:
        raise ValueError("stored prediction and execution payloads disagree")
    query = _dict(prediction.get("query"), "stored query")
    expected_query = _normalized(
        asdict(
            maeda_2012_query(
                job.target.protein_system,
                job.field_point.vector_tesla,
                observable=ObservableKind.SECONDARY_RADICAL_YIELD,
                query_id=job.query_id,
            )
        )
    )
    if query != expected_query:
        raise ValueError("stored query does not equal the exact planned query")
    system = _dict(query.get("system"), "stored query system")
    protocol = _dict(query.get("target_protocol"), "stored target protocol")
    field = _dict(protocol.get("field"), "stored field")
    expected_system = job.target
    if (
        query.get("query_id") != job.query_id
        or query.get("observable") != "secondary_radical_yield"
        or query.get("model_family") != job.target.model_family.value
        or query.get("reference_protocol") is not None
        or system.get("system_id") != expected_system.system_id
        or system.get("parameter_set_id") != expected_system.parameter_set_id
        or system.get("parameter_set_sha256") != expected_system.parameter_set_sha256
        or system.get("parameter_catalog_version") != expected_system.parameter_catalog_version
        or system.get("parameter_catalog_sha256") != expected_system.parameter_catalog_sha256
        or (
            field.get("x_tesla"),
            field.get("y_tesla"),
            field.get("z_tesla"),
        )
        != job.field_point.vector_tesla
        or field.get("intent") != "ideal_simulation_target"
    ):
        raise ValueError("stored datum contradicts its exact sweep job")
    if prediction.get("resolved_setup") != {
        "system": query["system"],
        "target_protocol": query["target_protocol"],
        "reference_protocol": query["reference_protocol"],
    }:
        raise ValueError("stored resolved setup does not preserve the query")
    if prediction.get("oracle") != _normalized(asdict(plan.execution_oracle)):
        raise ValueError("stored oracle identity contradicts the parent plan")
    if (
        prediction.get("executed_model_family") != query.get("model_family")
        or prediction.get("executed_model_id") != execution.get("model_id")
        or execution.get("model_family") != query.get("model_family")
    ):
        raise ValueError("stored execution model identity is inconsistent")
    if prediction.get("claim_boundary") != _PREDICTION_CLAIM_BOUNDARY:
        raise ValueError("stored prediction has an invalid claim boundary")
    _validate_distribution(prediction)
    _validate_applicability(prediction, job)
    query_hash = _canonical_sha256(query)
    prediction_hash = _canonical_sha256(prediction)
    configuration_hash, execution_hash = _execution_hashes(execution)
    _validate_lineage(prediction, job, configuration_hash)
    dependency_hash = _canonical_sha256(
        {
            "system_id": system["system_id"],
            "parameter_set_id": system["parameter_set_id"],
            "parameter_set_sha256": system["parameter_set_sha256"],
            "parameter_catalog_version": system["parameter_catalog_version"],
            "parameter_catalog_sha256": system["parameter_catalog_sha256"],
            "model_family": query["model_family"],
            "oracle_artifact_sha256": plan.execution_oracle.artifact_sha256,
        }
    )
    expected_hashes = {
        "query_sha256": query_hash,
        "prediction_sha256": prediction_hash,
        "configuration_sha256": configuration_hash,
        "execution_sha256": execution_hash,
        "dependency_group_sha256": dependency_hash,
    }
    if any(hashes.get(key) != value for key, value in expected_hashes.items()):
        raise ValueError("stored simulation datum contains an invalid derived hash")
    datum_hash = _canonical_sha256(
        {
            "schema_version": _DATUM_SCHEMA,
            "query_sha256": query_hash,
            "prediction_sha256": prediction_hash,
            "execution_sha256": execution_hash,
            "dependency_group_sha256": dependency_hash,
            "evidence_origin": "simulation",
            "membership": "reference_only_not_training",
        }
    )
    if hashes.get("datum_sha256") != datum_hash or query_hash != job.query_sha256:
        raise ValueError("stored datum identity does not match the planned query")
    if raw != _canonical_job_result(job):
        raise ValueError("stored datum does not equal the canonical deterministic Maeda result")
    reference = StoredSimulationDatumRef(
        job_sha256=job.job_sha256,
        document_sha256=sha256(raw).hexdigest(),
        datum_sha256=datum_hash,
        query_sha256=query_hash,
        prediction_sha256=prediction_hash,
        configuration_sha256=configuration_hash,
        execution_sha256=execution_hash,
        dependency_group_sha256=dependency_hash,
    )
    return reference, document


def _record_document(
    reference: StoredSimulationDatumRef,
    document: dict[str, object],
) -> dict[str, object]:
    return {"reference": asdict(reference), "simulation_datum": document}


def _bundle_document(
    plan: MaedaSweepPlan,
    records: list[dict[str, object]],
) -> dict[str, object]:
    results_hash = _canonical_sha256(records)
    identity = {
        "schema_version": _BUNDLE_SCHEMA,
        "status": "complete",
        "plan_sha256": plan.plan_sha256,
        "specification_sha256": plan.specification.specification_sha256,
        "oracle_artifact_sha256": plan.execution_oracle.artifact_sha256,
        "results_sha256": results_hash,
        "claim_boundary": _CLAIM_BOUNDARY,
    }
    bundle_hash = _canonical_sha256(identity)
    return {
        **identity,
        "plan": maeda_sweep_plan_document(plan),
        "results": records,
        "hashes": {"results_sha256": results_hash, "bundle_sha256": bundle_hash},
    }


def _bundle_paths(plan: MaedaSweepPlan, output_root: Path) -> tuple[Path, Path, Path]:
    stem = f"maeda-{plan.plan_sha256}"
    return (
        output_root / f"{stem}.bundle.json",
        output_root / f".{stem}.partial",
        output_root / f".{stem}.lock",
    )


def _cleanup_root_leftovers(
    output_root: Path,
    final: Path,
    partial: Path,
    *,
    expected_root_identity: _DirectoryIdentity,
) -> None:
    temporary_pattern = re.compile(rf"{re.escape(f'.tmp-{final.name}-')}[0-9a-f]{{16}}")
    initialization_pattern = re.compile(rf"{re.escape(f'.init-{partial.name}-')}[A-Za-z0-9_]{{6,}}")
    root_fd = os.open(
        output_root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened_root = os.fstat(root_fd)
        if (opened_root.st_dev, opened_root.st_ino) != (
            expected_root_identity.device,
            expected_root_identity.inode,
        ):
            raise ValueError("output root was replaced before cleanup")
        for name in os.listdir(root_fd):
            if temporary_pattern.fullmatch(name):
                metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValueError("interrupted publication temporary must be a regular file")
                os.unlink(name, dir_fd=root_fd)
                os.fsync(root_fd)
            elif initialization_pattern.fullmatch(name):
                metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                if not stat.S_ISDIR(metadata.st_mode):
                    raise ValueError("interrupted initialization workspace must be a directory")
                _remove_tree_at(
                    root_fd,
                    name,
                    expected_identity=_DirectoryIdentity(metadata.st_dev, metadata.st_ino),
                )
    finally:
        os.close(root_fd)


def _remove_authenticated_partial(
    output_root: Path,
    partial: Path,
    expected_identity: _DirectoryIdentity,
    *,
    expected_root_identity: _DirectoryIdentity,
) -> None:
    """Detach then remove only the exact authenticated partial workspace."""

    directory_fd = os.open(
        output_root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    detached_name = f".cleanup-{partial.name}-{token_hex(8)}"
    detached = output_root / detached_name
    source_fd: int | None = None
    try:
        opened_root = os.fstat(directory_fd)
        if (opened_root.st_dev, opened_root.st_ino) != (
            expected_root_identity.device,
            expected_root_identity.inode,
        ):
            raise ValueError("output root was replaced before partial cleanup")
        source_fd = os.open(
            partial.name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        source = os.fstat(source_fd)
        current = os.stat(partial.name, dir_fd=directory_fd, follow_symlinks=False)
        expected_pair = (expected_identity.device, expected_identity.inode)
        if (source.st_dev, source.st_ino) != expected_pair or (
            current.st_dev,
            current.st_ino,
        ) != expected_pair:
            raise ValueError("partial workspace was replaced before cleanup")
        os.rename(
            partial.name,
            detached_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
        if _directory_identity(detached, "detached partial workspace") != expected_identity:
            os.rename(
                detached_name,
                partial.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            os.fsync(directory_fd)
            raise ValueError("partial workspace was replaced before cleanup")
        try:
            _remove_tree_at(directory_fd, detached_name, expected_identity=expected_identity)
        except (OSError, ValueError):
            os.rename(
                detached_name,
                partial.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            os.fsync(directory_fd)
            raise
    finally:
        if source_fd is not None:
            os.close(source_fd)
        os.close(directory_fd)


def _prepare_root(output_root: Path) -> _DirectoryIdentity:
    if output_root.exists():
        _lstat_directory(output_root, "output root")
    else:
        output_root.mkdir(parents=True, mode=0o700)
    _fsync_directory(output_root)
    return _directory_identity(output_root, "output root")


def _acquire_lock(path: Path) -> int:
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("sweep lock must be a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _prepare_partial(plan: MaedaSweepPlan, partial: Path, *, resume: bool) -> Path:
    control = partial / "control.json"
    results = partial / "results"
    expected_control = maeda_sweep_plan_json(plan)
    if partial.exists() or partial.is_symlink():
        _lstat_directory(partial, "partial workspace")
        if not resume:
            raise FileExistsError(f"partial sweep workspace already exists: {partial}")
        allowed = {"control.json", "results", "last-failure.json"}
        if {item.name for item in partial.iterdir()} - allowed:
            raise ValueError("partial workspace contains unexpected files")
        if _read_regular(control, "partial control") != expected_control:
            raise ValueError("partial workspace belongs to a different plan")
        _lstat_directory(results, "partial results directory")
        failure = partial / "last-failure.json"
        if failure.exists() or failure.is_symlink():
            receipt = _load_canonical_json(
                _read_regular(failure, "failure receipt"), label="failure receipt"
            )
            _require_exact_keys(
                receipt,
                {"job_sha256", "category", "message"},
                "failure receipt",
            )
            if (
                receipt["job_sha256"] not in {job.job_sha256 for job in plan.jobs}
                or receipt["category"]
                not in {"numerical_failure", "validation_failure", "execution_failure"}
                or receipt["message"] != "Job failed before authenticated result publication."
            ):
                raise ValueError("failure receipt has invalid sanitized semantics")
            _unlink_regular(
                partial,
                failure.name,
                "failure receipt",
                expected_parent_identity=_directory_identity(partial, "partial workspace"),
            )
    else:
        staging = Path(tempfile.mkdtemp(prefix=f".init-{partial.name}-", dir=partial.parent))
        staging_identity = _directory_identity(staging, "initialization workspace")
        try:
            staged_results = staging / "results"
            staged_results.mkdir(mode=0o700)
            _atomic_link_bytes(
                staging,
                staging / "control.json",
                expected_control,
                expected_identity=staging_identity,
            )
            _fsync_directory(staged_results)
            _fsync_directory(staging)
            os.rename(staging, partial)
            _fsync_directory(partial.parent)
        finally:
            root_fd = os.open(
                partial.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                with suppress(FileNotFoundError):
                    _remove_tree_at(
                        root_fd,
                        staging.name,
                        expected_identity=staging_identity,
                    )
            finally:
                os.close(root_fd)
        results = partial / "results"
    return results


def _partial_results(
    plan: MaedaSweepPlan,
    results_directory: Path,
    *,
    expected_identity: _DirectoryIdentity,
) -> dict[str, tuple[StoredSimulationDatumRef, dict[str, object]]]:
    _require_directory_identity(results_directory, expected_identity, "partial results directory")
    jobs = {job.job_sha256: job for job in plan.jobs}
    found: dict[str, tuple[StoredSimulationDatumRef, dict[str, object]]] = {}
    for path in results_directory.iterdir():
        if path.name.startswith(".tmp-"):
            _unlink_regular(
                results_directory,
                path.name,
                "interrupted result temporary",
                expected_parent_identity=expected_identity,
            )
            continue
        if not path.name.endswith(".json") or path.stem not in jobs:
            raise ValueError("partial results contain an unexpected filename")
        raw = _read_regular(path, "partial simulation datum")
        found[path.stem] = _validate_raw_datum(raw, plan=plan, job=jobs[path.stem])
    _require_directory_identity(results_directory, expected_identity, "partial results directory")
    return found


def _failure_category(error: BaseException) -> str:
    if isinstance(error, ArithmeticError):
        return "numerical_failure"
    if isinstance(error, ValueError):
        return "validation_failure"
    return "execution_failure"


def run_maeda_sweep_bundle(
    plan: MaedaSweepPlan,
    output_root: Path,
    *,
    resume: bool = False,
) -> Path:
    """Execute and atomically publish one authenticated sweep bundle."""

    if not isinstance(plan, MaedaSweepPlan):
        raise ValueError("bundle execution requires a MaedaSweepPlan")
    if not isinstance(output_root, Path):
        raise ValueError("output root must be a pathlib.Path")
    root_identity = _prepare_root(output_root)
    final, partial, lock = _bundle_paths(plan, output_root)
    descriptor = _acquire_lock(lock)
    try:
        _require_directory_identity(output_root, root_identity, "output root")
        if resume:
            _cleanup_root_leftovers(
                output_root,
                final,
                partial,
                expected_root_identity=root_identity,
            )
            _require_directory_identity(output_root, root_identity, "output root")
        if final.exists() or final.is_symlink():
            raise FileExistsError(f"final sweep bundle already exists: {final}")
        results_directory = _prepare_partial(plan, partial, resume=resume)
        partial_identity = _directory_identity(partial, "partial workspace")
        results_identity = _directory_identity(results_directory, "partial results directory")
        completed = _partial_results(
            plan,
            results_directory,
            expected_identity=results_identity,
        )
        executor = Maeda2012SimulatorOracle()
        for job in plan.jobs:
            if job.job_sha256 in completed:
                continue
            try:
                datum = execute_maeda_sweep_job(plan, job, executor)
                validate_maeda_job_result(plan, job, datum)
                raw = simulation_datum_json(datum)
                validated = _validate_raw_datum(raw, plan=plan, job=job)
                destination = results_directory / f"{job.job_sha256}.json"
                _require_directory_identity(
                    results_directory, results_identity, "partial results directory"
                )
                _atomic_link_bytes(
                    results_directory,
                    destination,
                    raw,
                    expected_identity=results_identity,
                )
                completed[job.job_sha256] = validated
            except BaseException as error:
                if isinstance(error, PublishedDurabilityError):
                    raise
                _require_directory_identity(partial, partial_identity, "partial workspace")
                _operational_write(
                    partial / "last-failure.json",
                    {
                        "job_sha256": job.job_sha256,
                        "category": _failure_category(error),
                        "message": "Job failed before authenticated result publication.",
                    },
                    expected_parent_identity=partial_identity,
                )
                raise
        _require_directory_identity(partial, partial_identity, "partial workspace")
        completed = _partial_results(
            plan,
            results_directory,
            expected_identity=results_identity,
        )
        if set(completed) != {job.job_sha256 for job in plan.jobs}:
            raise ValueError("partial workspace does not contain the complete planned job set")
        records = [_record_document(*completed[job.job_sha256]) for job in plan.jobs]
        configuration_hashes = {
            _dict(record["reference"], "result reference")["configuration_sha256"]
            for record in records
        }
        if len(configuration_hashes) != 1:
            raise ValueError("all sweep results must share one execution configuration")
        document = _bundle_document(plan, records)
        _require_directory_identity(output_root, root_identity, "output root")
        _atomic_link_bytes(
            output_root,
            final,
            _canonical_bytes(document),
            expected_identity=root_identity,
        )
        try:
            _require_directory_identity(partial, partial_identity, "partial workspace")
            _remove_authenticated_partial(
                output_root,
                partial,
                partial_identity,
                expected_root_identity=root_identity,
            )
        except (OSError, ValueError):
            # Publication commits at the no-replace hardlink plus output-root fsync above.
            # Cleanup is best-effort and must not turn a published bundle into a reported failure.
            pass
        return final
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _inspect_document(
    document: dict[str, object],
    *,
    expected_plan: MaedaSweepPlan,
) -> MaedaSweepBundle:
    required = {
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
    _require_exact_keys(document, required, "sweep bundle")
    if document["schema_version"] != _BUNDLE_SCHEMA or document["status"] != "complete":
        raise ValueError("sweep bundle schema or status is unsupported")
    records = _list(document["results"], "bundle results")
    plan_document = _dict(document["plan"], "bundle plan")
    plan_job_hashes = _validate_embedded_plan(plan_document, document)
    if len(records) != len(plan_job_hashes):
        raise ValueError("bundle result count does not match the embedded plan")
    results_hash = _canonical_sha256(records)
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
    bundle_hash = _canonical_sha256(identity)
    hashes = _dict(document["hashes"], "bundle hashes")
    if document["results_sha256"] != results_hash or hashes != {
        "results_sha256": results_hash,
        "bundle_sha256": bundle_hash,
    }:
        raise ValueError("sweep bundle content hashes are invalid")
    references: list[StoredSimulationDatumRef] = []
    if (
        document["plan_sha256"] != expected_plan.plan_sha256
        or document["specification_sha256"] != expected_plan.specification.specification_sha256
        or document["oracle_artifact_sha256"] != expected_plan.execution_oracle.artifact_sha256
        or document["plan"] != maeda_sweep_plan_document(expected_plan)
        or len(records) != len(expected_plan.jobs)
    ):
        raise ValueError("sweep bundle does not match the expected plan")
    for record_value, job in zip(records, expected_plan.jobs, strict=True):
        record = _dict(record_value, "bundle result record")
        _require_exact_keys(record, {"reference", "simulation_datum"}, "bundle result record")
        datum_document = _dict(record["simulation_datum"], "bundled simulation datum")
        raw = _canonical_bytes(datum_document)
        reference, _ = _validate_raw_datum(raw, plan=expected_plan, job=job)
        if record["reference"] != asdict(reference):
            raise ValueError("bundle result reference does not authenticate its datum")
        references.append(reference)
    if len({reference.configuration_sha256 for reference in references}) != 1:
        raise ValueError("all bundled results must share one execution configuration")
    return MaedaSweepBundle(
        plan_sha256=_string(document["plan_sha256"], "plan sha256"),
        specification_sha256=_string(document["specification_sha256"], "specification sha256"),
        oracle_artifact_sha256=_string(
            document["oracle_artifact_sha256"], "oracle artifact sha256"
        ),
        results=tuple(references),
        results_sha256=results_hash,
        bundle_sha256=bundle_hash,
        claim_boundary=_string(document["claim_boundary"], "claim boundary"),
    )


def _validate_embedded_plan(
    plan_document: dict[str, object],
    bundle_document: dict[str, object],
) -> tuple[str, ...]:
    _require_exact_keys(
        plan_document,
        {"schema_version", "specification", "execution_oracle", "jobs", "hashes"},
        "bundle plan",
    )
    specification = _dict(plan_document["specification"], "bundle plan specification")
    jobs = _list(plan_document["jobs"], "bundle plan jobs")
    plan_hashes = _dict(plan_document["hashes"], "bundle plan hashes")
    specification_hash = _canonical_sha256(specification)
    job_hashes: list[str] = []
    for value in jobs:
        job = dict(_dict(value, "bundle plan job"))
        scientific_hash = job.pop("scientific_family_sha256", None)
        stored_job_hash = job.pop("job_sha256", None)
        calculated_job_hash = _canonical_sha256(job)
        target = _dict(job.get("target"), "bundle plan job target")
        calculated_family_hash = _canonical_sha256(
            {
                "system_id": target.get("system_id"),
                "parameter_set_id": target.get("parameter_set_id"),
                "parameter_set_sha256": target.get("parameter_set_sha256"),
                "model_family": target.get("model_family"),
            }
        )
        if scientific_hash != calculated_family_hash or stored_job_hash != calculated_job_hash:
            raise ValueError("bundle plan contains an invalid job identity")
        job_hashes.append(calculated_job_hash)
    plan_hash = _canonical_sha256(
        {
            "specification_sha256": specification_hash,
            "execution_oracle": plan_document["execution_oracle"],
            "job_sha256": job_hashes,
        }
    )
    if (
        plan_hashes != {"specification_sha256": specification_hash, "plan_sha256": plan_hash}
        or bundle_document["specification_sha256"] != specification_hash
        or bundle_document["plan_sha256"] != plan_hash
        or bundle_document["oracle_artifact_sha256"]
        != _dict(plan_document["execution_oracle"], "bundle execution oracle").get(
            "artifact_sha256"
        )
    ):
        raise ValueError("embedded sweep plan hashes are invalid")
    return tuple(job_hashes)


def inspect_maeda_sweep_bundle(
    path: Path,
    *,
    expected_plan: MaedaSweepPlan,
) -> MaedaSweepBundle:
    """Fully authenticate a canonical bundle against its expected immutable plan."""

    bundle, _ = load_authenticated_maeda_sweep_bundle(path, expected_plan=expected_plan)
    return bundle


def load_authenticated_maeda_sweep_documents(
    path: Path,
    *,
    expected_plan: MaedaSweepPlan,
) -> tuple[dict[str, object], ...]:
    """Load ordered raw datum documents only after full plan-bound authentication."""

    _, documents = load_authenticated_maeda_sweep_bundle(path, expected_plan=expected_plan)
    return documents


def load_authenticated_maeda_sweep_bundle(
    path: Path,
    *,
    expected_plan: MaedaSweepPlan,
) -> tuple[MaedaSweepBundle, tuple[dict[str, object], ...]]:
    """Authenticate once and return both bundle identity and ordered datum documents."""

    if not isinstance(path, Path):
        raise ValueError("bundle path must be a pathlib.Path")
    if not isinstance(expected_plan, MaedaSweepPlan):
        raise ValueError("bundle inspection requires an expected MaedaSweepPlan")
    raw = _read_regular(path, "sweep bundle")
    document = _load_canonical_json(raw, label="sweep bundle")
    bundle = _inspect_document(document, expected_plan=expected_plan)
    records = _list(document["results"], "bundle results")
    documents = tuple(
        _dict(_dict(record, "bundle result record")["simulation_datum"], "simulation datum")
        for record in records
    )
    return bundle, documents
