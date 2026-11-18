from __future__ import annotations

import json
import os
import stat
from copy import deepcopy
from pathlib import Path

import pytest

import paircraft.sweeps.qa as qa_module
from paircraft.spin import MaedaProteinSystem
from paircraft.sweeps.bundle import (
    MaedaSweepBundle,
    load_authenticated_maeda_sweep_bundle,
    run_maeda_sweep_bundle,
)
from paircraft.sweeps.maeda import (
    IdealFieldPoint,
    MaedaSweepPlan,
    MaedaSweepSpecification,
    maeda_sweep_target,
    materialize_maeda_sweep,
)
from paircraft.sweeps.qa import (
    MaedaSweepQaReport,
    build_maeda_sweep_qa_report,
    maeda_sweep_qa_json,
    write_maeda_sweep_qa_report,
)


@pytest.fixture(scope="module")
def authenticated_report(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, MaedaSweepPlan, MaedaSweepQaReport]:
    output = tmp_path_factory.mktemp("qa-bundle")
    plan = materialize_maeda_sweep(
        MaedaSweepSpecification(
            sweep_id="qa-test",
            targets=(maeda_sweep_target(MaedaProteinSystem.ATCRY1),),
            field_points=(
                IdealFieldPoint("zero", 0.0),
                IdealFieldPoint("ten-ut", 10e-6),
                IdealFieldPoint("fifty-ut", 50e-6),
            ),
            claim_boundary="Deterministic QA test sweep.",
        )
    )
    bundle = run_maeda_sweep_bundle(plan, output)
    return bundle, plan, build_maeda_sweep_qa_report(bundle, expected_plan=plan)


def test_report_preserves_rows_and_explicit_zero_field_comparisons(
    authenticated_report: tuple[Path, MaedaSweepPlan, MaedaSweepQaReport],
) -> None:
    _, plan, report = authenticated_report

    assert report.plan_sha256 == plan.plan_sha256
    assert len(report.rows) == 3
    assert len(report.zero_field_relative_effects) == 3
    zero = report.zero_field_relative_effects[0]
    assert zero.field_magnitude_tesla == 0.0
    assert zero.relative_effect == 0.0
    assert zero.target_job_sha256 == zero.reference_job_sha256
    assert zero.dependency_group_sha256 == report.rows[0].dependency_group_sha256
    assert zero.model_family == report.rows[0].model_family
    summary = report.curve_summaries[0]
    assert summary.point_count == 3
    assert summary.relative_effect_at_10_microtesla is not None
    assert summary.relative_effect_at_20_microtesla is None
    assert summary.relative_effect_at_50_microtesla is not None
    assert summary.maximum_achieved_relative_residual <= 1e-8
    assert summary.dependency_group_sha256 == report.rows[0].dependency_group_sha256
    assert summary.configuration_sha256 == report.rows[0].configuration_sha256


def test_qa_json_is_canonical_and_content_addressed(
    authenticated_report: tuple[Path, MaedaSweepPlan, MaedaSweepQaReport],
) -> None:
    _, _, report = authenticated_report
    encoded = maeda_sweep_qa_json(report)
    document = json.loads(encoded)

    assert document["report_sha256"] == report.report_sha256
    assert (
        encoded
        == json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    assert "not measurements" in document["claim_boundary"]


def test_writer_refuses_overwrite(
    tmp_path: Path,
    authenticated_report: tuple[Path, MaedaSweepPlan, MaedaSweepQaReport],
) -> None:
    bundle, plan, _ = authenticated_report
    path = tmp_path / "qa.json"

    assert write_maeda_sweep_qa_report(bundle, expected_plan=plan, output_path=path) == path
    with pytest.raises(FileExistsError):
        write_maeda_sweep_qa_report(bundle, expected_plan=plan, output_path=path)


def test_report_cannot_be_constructed_outside_authenticated_boundary() -> None:
    with pytest.raises(ValueError, match="authenticated sweep bundle"):
        MaedaSweepQaReport(
            plan_sha256="0" * 64,
            bundle_sha256="1" * 64,
            rows=(),
            zero_field_relative_effects=(),
            curve_summaries=(),
            _construction_key=object(),
        )


def test_published_report_survives_post_link_directory_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authenticated_report: tuple[Path, MaedaSweepPlan, MaedaSweepQaReport],
) -> None:
    bundle, plan, _ = authenticated_report
    output = tmp_path / "qa.json"
    real_fsync = os.fsync

    def fail_directory_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("injected post-link directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_directory_fsync)

    assert write_maeda_sweep_qa_report(bundle, expected_plan=plan, output_path=output) == output
    assert output.is_file()


def test_writer_cleans_temporary_file_when_publication_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authenticated_report: tuple[Path, MaedaSweepPlan, MaedaSweepQaReport],
) -> None:
    bundle, plan, _ = authenticated_report
    output = tmp_path / "qa.json"

    def fail_link(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("injected publication failure")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(OSError, match="injected publication failure"):
        write_maeda_sweep_qa_report(bundle, expected_plan=plan, output_path=output)

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_writer_rejects_untyped_output_path(
    authenticated_report: tuple[Path, MaedaSweepPlan, MaedaSweepQaReport],
) -> None:
    bundle, plan, _ = authenticated_report
    with pytest.raises(ValueError, match=r"pathlib\.Path"):
        write_maeda_sweep_qa_report(
            bundle,
            expected_plan=plan,
            output_path="qa.json",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "corruption, message",
    [
        ("missing-document", "job counts differ"),
        ("prediction-type", "prediction must be an object"),
        ("boolean-yield", "real scalar"),
        ("nonfinite-yield", "must be finite"),
        ("negative-iteration", "nonnegative integer"),
        ("yield-out-of-range", r"lie in \[0, 1\]"),
        ("residual-out-of-range", "residual exceeds"),
        ("mixed-model-family", "cannot mix scientific families"),
        ("zero-reference-yield", "cannot normalize"),
    ],
)
def test_builder_defensively_rejects_corrupted_authenticated_loader_output(
    monkeypatch: pytest.MonkeyPatch,
    authenticated_report: tuple[Path, MaedaSweepPlan, MaedaSweepQaReport],
    corruption: str,
    message: str,
) -> None:
    bundle_path, plan, _ = authenticated_report
    bundle, authenticated = load_authenticated_maeda_sweep_bundle(bundle_path, expected_plan=plan)
    documents = deepcopy(list(authenticated))
    if corruption == "missing-document":
        documents.pop()
    elif corruption == "prediction-type":
        documents[0]["prediction"] = []
    elif corruption == "boolean-yield":
        documents[0]["prediction"]["distribution"]["mean"] = True  # type: ignore[index]
    elif corruption == "nonfinite-yield":
        documents[0]["prediction"]["distribution"]["mean"] = float("inf")  # type: ignore[index]
    elif corruption == "negative-iteration":
        documents[0]["execution"]["iteration_count"] = -1  # type: ignore[index]
    elif corruption == "yield-out-of-range":
        documents[0]["prediction"]["distribution"]["mean"] = 2.0  # type: ignore[index]
    elif corruption == "residual-out-of-range":
        documents[0]["execution"]["achieved_relative_residual"] = 2e-8  # type: ignore[index]
    elif corruption == "mixed-model-family":
        documents[1]["prediction"]["query"]["model_family"] = "fabricated"  # type: ignore[index]
    else:
        documents[0]["prediction"]["distribution"]["mean"] = 0.0  # type: ignore[index]

    def corrupted_loader(
        path: Path, *, expected_plan: MaedaSweepPlan
    ) -> tuple[MaedaSweepBundle, tuple[dict[str, object], ...]]:
        del path, expected_plan
        return bundle, tuple(documents)

    monkeypatch.setattr(
        qa_module,
        "load_authenticated_maeda_sweep_bundle",
        corrupted_loader,
    )
    with pytest.raises(ValueError, match=message):
        build_maeda_sweep_qa_report(bundle_path, expected_plan=plan)
