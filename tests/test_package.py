"""Package-boundary smoke tests."""

from importlib.metadata import version

import paircraft


def test_import_exposes_installed_version() -> None:
    """The import boundary reports the installed distribution version."""
    assert paircraft.__version__ == version("paircraft")
