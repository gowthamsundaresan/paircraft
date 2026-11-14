"""Command-line interface for Paircraft."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from paircraft import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the public command-line parser."""
    parser = argparse.ArgumentParser(
        prog="paircraft",
        description="Computational design tools for magnetosensitive flavoproteins.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the installed Paircraft version and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Paircraft command-line interface."""
    args = build_parser().parse_args(argv)
    if args.version:
        print(__version__)
        return 0

    build_parser().print_help()
    return 0
