"""Public CLI smoke tests."""

import pytest

from paircraft.cli import main


def test_version_reports_installed_version(capsys: pytest.CaptureFixture[str]) -> None:
    """The version command succeeds and emits one non-empty line."""
    assert main(["--version"]) == 0
    output = capsys.readouterr().out
    assert output.strip()
    assert output.count("\n") == 1


def test_default_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    """The default command exposes its available public behavior."""
    assert main([]) == 0
    assert "usage: paircraft" in capsys.readouterr().out
