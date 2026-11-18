"""Generic canonical, crash-safe publication for generated JSON artifacts.

Filesystem locks and cleanup coordinate cooperative same-user writers. Identity
checks detect replacement at the transitions where they run; portable name-based
deletion is not an adversarial security boundary against a malicious same-user
process racing after the final check.
"""

from __future__ import annotations

import fcntl
import json
import os
import stat
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from secrets import token_hex
from typing import cast

_MAX_JSON_BYTES = 64 * 1024 * 1024


class PublishedDurabilityError(OSError):
    """Final name exists, but its containing-directory durability is indeterminate."""


@dataclass(frozen=True, slots=True)
class DirectoryIdentity:
    """Device/inode identity anchoring a directory across mutations."""

    device: int
    inode: int


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def directory_identity(path: Path, label: str) -> DirectoryIdentity:
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
        return DirectoryIdentity(metadata.st_dev, metadata.st_ino)
    finally:
        os.close(descriptor)


def require_directory_identity(path: Path, expected: DirectoryIdentity, label: str) -> None:
    if directory_identity(path, label) != expected:
        raise ValueError(f"{label} was replaced during bundle execution")


def lstat_directory(path: Path, label: str) -> None:
    if not stat.S_ISDIR(path.lstat().st_mode):
        raise ValueError(f"{label} must be a real non-symlink directory")


def read_regular(path: Path, label: str) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise ValueError(f"{label} must be a readable regular non-symlink file") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= _MAX_JSON_BYTES:
            raise ValueError(f"{label} has an invalid size or file type")
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


def unlink_regular(
    directory: Path,
    name: str,
    label: str,
    *,
    expected_parent_identity: DirectoryIdentity,
) -> None:
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
        descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
        opened = os.fstat(descriptor)
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            current.st_dev,
            current.st_ino,
        ):
            raise ValueError(f"{label} must be a regular non-symlink file")
        os.unlink(name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)


def remove_tree_at(
    parent_fd: int,
    name: str,
    *,
    expected_identity: DirectoryIdentity | None = None,
) -> None:
    """Remove one checked tree while rechecking identities at declared transitions."""

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
            if stat.S_ISDIR(child.st_mode):
                remove_tree_at(
                    child_fd,
                    child_name,
                    expected_identity=DirectoryIdentity(child.st_dev, child.st_ino),
                )
            elif stat.S_ISREG(child.st_mode) or stat.S_ISLNK(child.st_mode):
                current = os.stat(child_name, dir_fd=child_fd, follow_symlinks=False)
                if (current.st_dev, current.st_ino) != (child.st_dev, child.st_ino):
                    raise ValueError("recursive cleanup entry was replaced")
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


def remove_authenticated_directory(
    output_root: Path,
    directory: Path,
    expected_identity: DirectoryIdentity,
    *,
    expected_root_identity: DirectoryIdentity,
) -> None:
    root_fd = os.open(
        output_root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    detached_name = f".cleanup-{directory.name}-{token_hex(8)}"
    try:
        root = os.fstat(root_fd)
        if (root.st_dev, root.st_ino) != (
            expected_root_identity.device,
            expected_root_identity.inode,
        ):
            raise ValueError("output root was replaced before cleanup")
        current = os.stat(directory.name, dir_fd=root_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (
            expected_identity.device,
            expected_identity.inode,
        ):
            raise ValueError("partial workspace was replaced before cleanup")
        os.rename(directory.name, detached_name, src_dir_fd=root_fd, dst_dir_fd=root_fd)
        os.fsync(root_fd)
        try:
            remove_tree_at(root_fd, detached_name, expected_identity=expected_identity)
        except (OSError, ValueError):
            os.rename(detached_name, directory.name, src_dir_fd=root_fd, dst_dir_fd=root_fd)
            os.fsync(root_fd)
            raise
    finally:
        os.close(root_fd)


def atomic_link_bytes(
    directory: Path,
    destination: Path,
    payload: bytes,
    *,
    expected_identity: DirectoryIdentity,
) -> None:
    load_canonical_json_bytes(payload, label="publication payload")
    if destination.parent != directory:
        raise ValueError("atomic destination must be directly inside its anchored directory")
    directory_fd = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    opened = os.fstat(directory_fd)
    if (opened.st_dev, opened.st_ino) != (expected_identity.device, expected_identity.inode):
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
        try:
            os.fsync(directory_fd)
        except OSError as error:
            raise PublishedDurabilityError(
                f"published {destination}; directory fsync failed"
            ) from error
    finally:
        with suppress(OSError):
            os.unlink(temporary_name, dir_fd=directory_fd)
        os.close(directory_fd)


def operational_write(
    path: Path,
    payload: object,
    *,
    expected_parent_identity: DirectoryIdentity,
) -> None:
    if not isinstance(payload, dict):
        raise ValueError("operational receipt must be a top-level JSON object")
    raw = canonical_json_bytes(payload)
    load_canonical_json_bytes(
        raw,
        label="operational receipt",
        maximum_bytes=_MAX_JSON_BYTES,
    )
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
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=parent_fd)
        os.close(parent_fd)


def acquire_lock(path: Path, *, expected_parent_identity: DirectoryIdentity) -> int:
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
        raise ValueError("lock parent directory was replaced")
    descriptor = os.open(
        path.name,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=parent_fd,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("bundle lock must be a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_locked_path(
            path,
            descriptor,
            expected_parent_identity=expected_parent_identity,
            parent_fd=parent_fd,
        )
    except Exception:
        os.close(descriptor)
        os.close(parent_fd)
        raise
    os.close(parent_fd)
    return descriptor


def require_locked_path(
    path: Path,
    descriptor: int,
    *,
    expected_parent_identity: DirectoryIdentity,
    parent_fd: int | None = None,
) -> None:
    """Require the named lock to remain the inode held by ``descriptor``."""

    owned_parent = parent_fd is None
    if parent_fd is None:
        parent_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    try:
        parent = os.fstat(parent_fd)
        opened = os.fstat(descriptor)
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            (parent.st_dev, parent.st_ino)
            != (expected_parent_identity.device, expected_parent_identity.inode)
            or not stat.S_ISREG(named.st_mode)
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise ValueError("named lock path no longer identifies the held lock")
    finally:
        if owned_parent:
            os.close(parent_fd)


def canonical_json_bytes(value: object) -> bytes:
    """Encode one value as finite, canonical JSON."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def canonical_sha256(value: object) -> str:
    """Hash the canonical JSON representation of one value."""

    return sha256(canonical_json_bytes(value)).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def load_canonical_json_bytes(
    raw: bytes,
    *,
    label: str,
    maximum_bytes: int = _MAX_JSON_BYTES,
) -> dict[str, object]:
    """Parse one bounded strict JSON object and require canonical encoding."""

    if maximum_bytes <= 0:
        raise ValueError("maximum canonical JSON size must be positive")
    if not raw or len(raw) > maximum_bytes:
        raise ValueError(f"{label} has an invalid size")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    if canonical_json_bytes(value) != raw:
        raise ValueError(f"{label} must use canonical JSON encoding")
    return cast(dict[str, object], value)


def read_canonical_json(path: Path, *, label: str) -> dict[str, object]:
    """Read one bounded regular non-symlink file as canonical JSON."""

    if not isinstance(path, Path):
        raise ValueError(f"{label} path must be a pathlib.Path")
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
        raw = b""
        while len(raw) < metadata.st_size:
            chunk = os.read(descriptor, min(metadata.st_size - len(raw), 1024 * 1024))
            if not chunk:
                raise ValueError(f"{label} changed while it was read")
            raw += chunk
        if os.read(descriptor, 1):
            raise ValueError(f"{label} changed while it was read")
    finally:
        os.close(descriptor)
    return load_canonical_json_bytes(raw, label=label)


def publish_canonical_json_once(path: Path, value: object) -> Path:
    """Durably publish canonical JSON without replacing an existing path.

    The hard link is the commit point. A crash before it leaves no final artifact;
    a crash after it leaves the complete fsynced payload, possibly with a harmless
    recognizable temporary beside it.
    """

    if not isinstance(path, Path):
        raise ValueError("publication path must be a pathlib.Path")
    if not isinstance(value, dict):
        raise ValueError("published canonical JSON must be a top-level object")
    payload = canonical_json_bytes(value)
    load_canonical_json_bytes(payload, label="publication payload")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"generated artifact already exists: {path}")
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_descriptor = os.open(path.parent, parent_flags)
    try:
        parent_metadata = os.fstat(parent_descriptor)
        if not stat.S_ISDIR(parent_metadata.st_mode):
            raise ValueError("publication parent must be a real directory")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".tmp-{path.name}-",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            current_parent = os.stat(path.parent, follow_symlinks=False)
            if (current_parent.st_dev, current_parent.st_ino) != (
                parent_metadata.st_dev,
                parent_metadata.st_ino,
            ):
                raise ValueError("publication parent was replaced")
            os.link(
                temporary.name,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            try:
                os.fsync(parent_descriptor)
            except OSError as error:
                raise PublishedDurabilityError(
                    f"published {path}; directory fsync failed"
                ) from error
        finally:
            with suppress(OSError):
                temporary.unlink()
    finally:
        os.close(parent_descriptor)
    return path
