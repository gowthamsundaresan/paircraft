"""Command-line interface for Paircraft."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

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
    subparsers = parser.add_subparsers(dest="command")
    reproduce = subparsers.add_parser(
        "reproduce-oxford-fig1c",
        help="Reproduce the authenticated final Oxford Figure 1c processed-trace analysis.",
    )
    reproduce.add_argument("--processed-npz", required=True, type=Path)
    reproduce.add_argument("--metadata", required=True, type=Path)
    reproduce.add_argument("--notebook", required=True, type=Path)
    reproduce.add_argument("--output-dir", required=True, type=Path)
    corpus = subparsers.add_parser(
        "build-oxford-calibration-corpus",
        help="Build the audited Tier-D Oxford v3 descriptive corpus.",
    )
    corpus.add_argument("--input-root", required=True, type=Path)
    corpus.add_argument("--metadata", required=True, type=Path)
    corpus.add_argument("--output-root", required=True, type=Path)
    corpus.add_argument("--result-name", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Paircraft command-line interface."""
    args = build_parser().parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    if args.command == "reproduce-oxford-fig1c":
        from paircraft.experiments.oxford_fig1c_report import reproduce_oxford_fig1c

        paths = reproduce_oxford_fig1c(
            cast(Path, args.processed_npz),
            cast(Path, args.metadata),
            cast(Path, args.notebook),
            cast(Path, args.output_dir),
        )
        for path in paths:
            print(path)
        return 0
    if args.command == "build-oxford-calibration-corpus":
        from paircraft.experiments.oxford_corpus import build_oxford_calibration_corpus

        destination = build_oxford_calibration_corpus(
            cast(Path, args.input_root),
            cast(Path, args.metadata),
            cast(Path, args.output_root),
            cast(str, args.result_name),
        )
        print(destination)
        return 0

    build_parser().print_help()
    return 0
