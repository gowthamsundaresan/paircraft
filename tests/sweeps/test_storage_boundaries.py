from __future__ import annotations

import os
from pathlib import Path

import pytest

from paircraft.sweeps.storage import (
    DirectoryIdentity,
    acquire_lock,
    atomic_link_bytes,
    canonical_json_bytes,
    directory_identity,
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


def test_directory_boundaries_reject_missing_files_and_replacement(tmp_path: Path) -> None:
    identity = directory_identity(tmp_path, "test directory")

    with pytest.raises(ValueError, match="anchored real directory"):
        directory_identity(tmp_path / "missing", "missing directory")
    with pytest.raises(ValueError, match="was replaced"):
        require_directory_identity(
            tmp_path,
            DirectoryIdentity(identity.device, identity.inode + 1),
            "test directory",
        )
    regular = tmp_path / "regular"
    regular.write_text("not a directory")
    with pytest.raises(ValueError, match="real non-symlink directory"):
        lstat_directory(regular, "regular file")


def test_regular_reader_and_unlinker_fail_closed_on_wrong_file_types(tmp_path: Path) -> None:
    identity = directory_identity(tmp_path, "test directory")
    empty = tmp_path / "empty.json"
    empty.touch()
    directory = tmp_path / "directory"
    directory.mkdir()

    with pytest.raises(ValueError, match="readable regular"):
        read_regular(tmp_path / "missing.json", "missing input")
    with pytest.raises(ValueError, match="invalid size or file type"):
        read_regular(empty, "empty input")
    with pytest.raises(ValueError, match="invalid size or file type"):
        read_regular(directory, "directory input")
    with pytest.raises(ValueError, match="parent directory was replaced"):
        unlink_regular(
            tmp_path,
            empty.name,
            "empty input",
            expected_parent_identity=DirectoryIdentity(identity.device, identity.inode + 1),
        )
    with pytest.raises(ValueError, match="regular non-symlink"):
        unlink_regular(
            tmp_path,
            directory.name,
            "directory input",
            expected_parent_identity=identity,
        )


def test_recursive_cleanup_checks_identity_and_unsafe_entries(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    parent_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        target_identity = directory_identity(target, "cleanup target")
        with pytest.raises(ValueError, match="target was replaced"):
            remove_tree_at(
                parent_fd,
                target.name,
                expected_identity=DirectoryIdentity(
                    target_identity.device,
                    target_identity.inode + 1,
                ),
            )

        fifo = target / "unsafe"
        os.mkfifo(fifo)
        with pytest.raises(ValueError, match="unsafe file type"):
            remove_tree_at(parent_fd, target.name, expected_identity=target_identity)
        fifo.unlink()
        nested = target / "nested"
        nested.mkdir()
        (nested / "value").write_text("ok")
        remove_tree_at(parent_fd, target.name, expected_identity=target_identity)
        assert not target.exists()
    finally:
        os.close(parent_fd)


def test_authenticated_directory_cleanup_rejects_wrong_root_or_target(tmp_path: Path) -> None:
    partial = tmp_path / "partial"
    partial.mkdir()
    root_identity = directory_identity(tmp_path, "output root")
    partial_identity = directory_identity(partial, "partial")

    with pytest.raises(ValueError, match="output root was replaced"):
        remove_authenticated_directory(
            tmp_path,
            partial,
            partial_identity,
            expected_root_identity=DirectoryIdentity(
                root_identity.device,
                root_identity.inode + 1,
            ),
        )
    with pytest.raises(ValueError, match="partial workspace was replaced"):
        remove_authenticated_directory(
            tmp_path,
            partial,
            DirectoryIdentity(partial_identity.device, partial_identity.inode + 1),
            expected_root_identity=root_identity,
        )


def test_atomic_and_operational_writers_reject_wrong_destination_identity(tmp_path: Path) -> None:
    identity = directory_identity(tmp_path, "output root")
    payload = canonical_json_bytes({"value": 1})

    with pytest.raises(ValueError, match="directly inside"):
        atomic_link_bytes(
            tmp_path,
            tmp_path / "nested" / "value.json",
            payload,
            expected_identity=identity,
        )
    with pytest.raises(ValueError, match="directory was replaced"):
        atomic_link_bytes(
            tmp_path,
            tmp_path / "value.json",
            payload,
            expected_identity=DirectoryIdentity(identity.device, identity.inode + 1),
        )
    with pytest.raises(ValueError, match="top-level JSON object"):
        operational_write(
            tmp_path / "receipt.json",
            ["not", "an", "object"],
            expected_parent_identity=identity,
        )
    with pytest.raises(ValueError, match="parent directory was replaced"):
        operational_write(
            tmp_path / "receipt.json",
            {"value": 1},
            expected_parent_identity=DirectoryIdentity(identity.device, identity.inode + 1),
        )


def test_lock_boundaries_reject_wrong_parent_and_named_inode(tmp_path: Path) -> None:
    identity = directory_identity(tmp_path, "lock root")
    lock = tmp_path / "bundle.lock"

    with pytest.raises(ValueError, match="lock parent directory was replaced"):
        acquire_lock(
            lock,
            expected_parent_identity=DirectoryIdentity(identity.device, identity.inode + 1),
        )

    descriptor = acquire_lock(lock, expected_parent_identity=identity)
    try:
        lock.rename(tmp_path / "held.lock")
        lock.write_text("replacement")
        with pytest.raises(ValueError, match="named lock path"):
            require_locked_path(lock, descriptor, expected_parent_identity=identity)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"", "invalid size"),
        (b"not-json", "valid strict JSON"),
        (b'{"value":NaN}', "non-finite JSON constant"),
        (b"[1,2]", "one JSON object"),
        (b'{"value": 1}', "canonical JSON encoding"),
    ],
)
def test_canonical_parser_rejects_invalid_or_noncanonical_payloads(
    raw: bytes,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        load_canonical_json_bytes(raw, label="payload")


def test_canonical_read_boundary_rejects_invalid_path_and_empty_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"pathlib\.Path"):
        read_canonical_json("payload.json", label="payload")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="readable regular"):
        read_canonical_json(tmp_path / "missing.json", label="payload")
    empty = tmp_path / "empty.json"
    empty.touch()
    with pytest.raises(ValueError, match="invalid size"):
        read_canonical_json(empty, label="payload")
    with pytest.raises(ValueError, match="maximum canonical JSON size"):
        load_canonical_json_bytes(b"{}", label="payload", maximum_bytes=0)
