"""Public CLI smoke tests."""

from pathlib import Path

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


def test_single_cell_reproduction_command_exposes_required_inputs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The scientific handoff command documents every authenticated input boundary."""

    with pytest.raises(SystemExit) as exit_info:
        main(["reproduce-oxford-fig1c", "--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "--processed-npz" in output
    assert "--metadata" in output
    assert "--notebook" in output
    assert "--output-dir" in output


def test_single_cell_reproduction_command_dispatches_authenticated_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI delegates exact paths and reports every generated artifact."""

    from paircraft.experiments import oxford_fig1c_report

    output = tmp_path / "result.json"

    def fake_reproduction(
        processed: Path, metadata: Path, notebook: Path, output_directory: Path
    ) -> tuple[Path, Path, Path]:
        assert (processed.name, metadata.name, notebook.name) == ("p.npz", "m.json", "n.ipynb")
        assert output_directory == tmp_path
        return output, tmp_path / "result.csv", tmp_path / "result.md"

    monkeypatch.setattr(oxford_fig1c_report, "reproduce_oxford_fig1c", fake_reproduction)
    assert (
        main(
            [
                "reproduce-oxford-fig1c",
                "--processed-npz",
                "p.npz",
                "--metadata",
                "m.json",
                "--notebook",
                "n.ipynb",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.splitlines() == [
        str(output),
        str(tmp_path / "result.csv"),
        str(tmp_path / "result.md"),
    ]


def test_corpus_command_exposes_audited_boundaries(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["build-oxford-calibration-corpus", "--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert all(
        option in output
        for option in ("--input-root", "--metadata", "--output-root", "--result-name")
    )


def test_corpus_command_dispatches_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from paircraft.experiments import oxford_corpus

    destination = tmp_path / "outputs/results/corpus"

    def fake_build(input_root: Path, metadata: Path, output_root: Path, result_name: str) -> Path:
        assert input_root == Path("inputs")
        assert metadata == Path("metadata.json")
        assert output_root == tmp_path / "outputs"
        assert result_name == "corpus"
        return destination

    monkeypatch.setattr(oxford_corpus, "build_oxford_calibration_corpus", fake_build)
    assert (
        main(
            [
                "build-oxford-calibration-corpus",
                "--input-root",
                "inputs",
                "--metadata",
                "metadata.json",
                "--output-root",
                str(tmp_path / "outputs"),
                "--result-name",
                "corpus",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == str(destination)
