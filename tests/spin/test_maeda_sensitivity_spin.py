from dataclasses import replace
from hashlib import sha256
from math import exp
from pathlib import Path

import pytest

import paircraft.spin._maeda_kernel as kernel_module
import paircraft.spin.maeda2012 as published_module
import paircraft.spin.maeda_sensitivity as sensitivity_module
from paircraft.spin import (
    MaedaProteinSystem,
    SensitivityDimension,
    SensitivityDirection,
    maeda_2012_specification,
    maeda_sensitivity_specification,
    solve_maeda_secondary_pair_yield,
    solve_maeda_sensitivity_yield,
)


def test_design_exposes_exactly_eight_one_at_a_time_dimensions() -> None:
    assert tuple(SensitivityDimension) == (
        SensitivityDimension.FAD_N5_HYPERFINE,
        SensitivityDimension.FAD_PSEUDO_H_HYPERFINE,
        SensitivityDimension.TRPH_H18_HYPERFINE,
        SensitivityDimension.TRPH_H23_HYPERFINE,
        SensitivityDimension.TRPH_PSEUDO_H_HYPERFINE,
        SensitivityDimension.FORWARD_RATE,
        SensitivityDimension.BACK_RECOMBINATION_RATE,
        SensitivityDimension.ST_DEPHASING_RATE,
    )


@pytest.mark.parametrize(
    ("dimension", "direction", "expected_delta", "baseline_value"),
    [
        (
            SensitivityDimension.BACK_RECOMBINATION_RATE,
            SensitivityDirection.LOWER,
            -0.05,
            4.9e5,
        ),
        (
            SensitivityDimension.FORWARD_RATE,
            SensitivityDirection.BASELINE,
            0.0,
            2.5e5,
        ),
        (
            SensitivityDimension.BACK_RECOMBINATION_RATE,
            SensitivityDirection.HIGHER,
            0.05,
            4.9e5,
        ),
    ],
)
def test_fixed_log_scale_design_keeps_published_evidence_separate(
    dimension: SensitivityDimension,
    direction: SensitivityDirection,
    expected_delta: float,
    baseline_value: float,
) -> None:
    published = maeda_2012_specification(MaedaProteinSystem.ATCRY1)
    specification = maeda_sensitivity_specification(
        MaedaProteinSystem.ATCRY1,
        dimension,
        direction,
    )

    assert specification.baseline is not published
    assert specification.baseline == published
    assert specification.override.baseline_value_si == baseline_value
    assert specification.override.proposed_value_si == pytest.approx(
        baseline_value * exp(expected_delta), rel=1e-15
    )
    assert specification.override.evidence_status == "assumed_hypothetical_design"
    assert specification.override.unit == "s^-1"
    assert specification.override.chemical_state
    assert "270 K" in specification.override.conditions
    assert specification.override.design_source_uri.startswith("paircraft:")
    assert "not an uncertainty" in specification.override.uncertainty_status
    assert "not a reported uncertainty" in specification.override.rationale
    assert specification.baseline_parameter_set_sha256 == published.parameter_set_sha256
    assert specification.parameter_set_sha256 != published.parameter_set_sha256


def test_each_dimension_has_a_distinct_content_addressed_parameter_set() -> None:
    specifications = tuple(
        maeda_sensitivity_specification(
            MaedaProteinSystem.ECPL_E109A,
            dimension,
            SensitivityDirection.HIGHER,
        )
        for dimension in SensitivityDimension
    )

    assert len({item.parameter_set_sha256 for item in specifications}) == 8
    assert all(item.override.unit in {"T", "s^-1"} for item in specifications)
    assert all(item.model_id != item.baseline.model_id for item in specifications)


def test_baseline_design_matches_published_numerics_but_not_identity() -> None:
    published_specification = maeda_2012_specification(MaedaProteinSystem.ATCRY1)
    hypothetical_specification = maeda_sensitivity_specification(
        MaedaProteinSystem.ATCRY1,
        SensitivityDimension.FORWARD_RATE,
        SensitivityDirection.BASELINE,
    )
    field = (0.0, 0.0, 50e-6)

    published = solve_maeda_secondary_pair_yield(published_specification, field)
    hypothetical = solve_maeda_sensitivity_yield(hypothetical_specification, field)

    assert hypothetical.secondary_pair == pytest.approx(published.secondary_pair, abs=1e-15)
    assert hypothetical.singlet_back_recombination == pytest.approx(
        published.singlet_back_recombination, abs=1e-15
    )
    assert hypothetical.model_id != published.model_id
    assert hypothetical.parameter_set_sha256 != published.parameter_set_sha256
    assert hypothetical.baseline_parameter_set_sha256 == published.parameter_set_sha256
    assert "not calculated" in hypothetical.uncertainty_status
    assert "not experimental calibration" in hypothetical.claim_boundary
    assert "mutation or sequence prediction" in hypothetical.claim_boundary
    digest = sha256()
    for module in (sensitivity_module, kernel_module, published_module):
        digest.update(Path(module.__file__ or "").read_bytes())
    assert hypothetical.code_artifact_sha256 == digest.hexdigest()


def test_specification_and_result_reject_identity_or_claim_rewrites() -> None:
    specification = maeda_sensitivity_specification(
        MaedaProteinSystem.ATCRY1,
        SensitivityDimension.FAD_N5_HYPERFINE,
        SensitivityDirection.LOWER,
    )
    override = specification.override
    with pytest.raises(ValueError, match="fixed transform"):
        replace(override, proposed_value_si=override.baseline_value_si)
    with pytest.raises(ValueError, match="provenance boundary"):
        replace(override, evidence_status="measured")
    with pytest.raises(ValueError, match="design identity"):
        replace(specification, model_id=specification.baseline.model_id)
    with pytest.raises(ValueError, match="canonical only"):
        maeda_sensitivity_specification(
            MaedaProteinSystem.ATCRY1,
            SensitivityDimension.ST_DEPHASING_RATE,
            SensitivityDirection.BASELINE,
        )

    result = solve_maeda_sensitivity_yield(specification, (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="cannot claim predictive uncertainty"):
        replace(result, uncertainty_status="calibrated")
    with pytest.raises(ValueError, match="claim boundary"):
        replace(result, claim_boundary="mutation prediction")
    with pytest.raises(ValueError, match="published baseline identity"):
        replace(result, baseline_parameter_set_sha256="0" * 64)
    with pytest.raises(ValueError, match="solver identity"):
        replace(result, solver="different solver")
    with pytest.raises(ValueError, match="solver identity"):
        replace(result, solver_version="numpy-unknown; scipy-unknown")
    with pytest.raises(ValueError, match="code version"):
        replace(result, code_version="different-version")
    with pytest.raises(ValueError, match="code identity"):
        replace(result, code_artifact_sha256="A" * 64)
    with pytest.raises(ValueError, match="code identity"):
        replace(result, code_artifact_sha256="0" * 64)


def test_public_builder_rejects_untyped_or_out_of_domain_inputs() -> None:
    with pytest.raises(ValueError, match="SensitivityDimension"):
        maeda_sensitivity_specification(
            MaedaProteinSystem.ATCRY1,
            "forward",  # type: ignore[arg-type]
            SensitivityDirection.BASELINE,
        )
    specification = maeda_sensitivity_specification(
        MaedaProteinSystem.ATCRY1,
        SensitivityDimension.FORWARD_RATE,
        SensitivityDirection.HIGHER,
    )
    with pytest.raises(ValueError, match="0-50 mT"):
        solve_maeda_sensitivity_yield(specification, (0.0, 0.0, 0.050001))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"dimension": "forward"}, "SensitivityDimension"),
        ({"direction": "higher"}, "SensitivityDirection"),
        ({"baseline_parameter_id": ""}, "context and provenance"),
        ({"baseline_parameter_sha256": "A" * 64}, "lowercase digest"),
        ({"baseline_value_si": float("nan")}, "must be finite"),
        ({"baseline_value_si": 0.0, "proposed_value_si": 0.0}, "nonzero baseline"),
        ({"log_scale_delta": 0.0}, "contradicts its direction"),
    ],
)
def test_override_rejects_malformed_design_inputs(
    changes: dict[str, object],
    message: str,
) -> None:
    specification = maeda_sensitivity_specification(
        MaedaProteinSystem.ATCRY1,
        SensitivityDimension.FORWARD_RATE,
        SensitivityDirection.HIGHER,
    )

    with pytest.raises(ValueError, match=message):
        replace(specification.override, **changes)  # type: ignore[arg-type]


def test_specification_rejects_untyped_or_mismatched_components() -> None:
    specification = maeda_sensitivity_specification(
        MaedaProteinSystem.ATCRY1,
        SensitivityDimension.FORWARD_RATE,
        SensitivityDirection.HIGHER,
    )
    other = maeda_sensitivity_specification(
        MaedaProteinSystem.ECPL_E109A,
        SensitivityDimension.FORWARD_RATE,
        SensitivityDirection.HIGHER,
    )

    with pytest.raises(ValueError, match="protein system"):
        sensitivity_module.MaedaSensitivitySpecification(
            "atcry1",  # type: ignore[arg-type]
            specification.baseline,
            specification.override,
        )
    with pytest.raises(ValueError, match="exact published specification"):
        replace(specification, baseline=other.baseline)
    with pytest.raises(ValueError, match="typed override"):
        sensitivity_module.MaedaSensitivitySpecification(
            specification.protein_system,
            specification.baseline,
            "override",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="inherited baseline"):
        replace(specification, override=other.override)


@pytest.fixture(scope="module")
def valid_hypothetical_result() -> sensitivity_module.HypotheticalSecondaryPairYield:
    specification = maeda_sensitivity_specification(
        MaedaProteinSystem.ATCRY1,
        SensitivityDimension.FORWARD_RATE,
        SensitivityDirection.BASELINE,
    )
    return solve_maeda_sensitivity_yield(specification, (0.0, 0.0, 0.0))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"field_tesla": (0.0, 0.0)}, "three finite"),
        ({"secondary_pair": -0.1}, "lie in"),
        ({"secondary_pair": 0.4, "singlet_back_recombination": 0.4}, "conserve"),
        ({"model_id": "other"}, "model identity"),
        ({"parameter_set_sha256": "0" * 64}, "parameter hash"),
        ({"iterations": 0}, "diagnostics"),
        ({"relative_residual": -1.0}, "diagnostics"),
        ({"requested_relative_tolerance": 1.0}, "diagnostics"),
        ({"restart": 0}, "configuration"),
        ({"maximum_restart_cycles": 0}, "configuration"),
        ({"field_intent": "measured"}, "ideal simulation target"),
    ],
)
def test_result_rejects_invalid_numerics_and_identity(
    valid_hypothetical_result: sensitivity_module.HypotheticalSecondaryPairYield,
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(valid_hypothetical_result, **changes)  # type: ignore[arg-type]


def test_public_builders_reject_untyped_system_direction_and_specification() -> None:
    with pytest.raises(ValueError, match="MaedaProteinSystem"):
        maeda_sensitivity_specification(
            "atcry1",  # type: ignore[arg-type]
            SensitivityDimension.FORWARD_RATE,
            SensitivityDirection.BASELINE,
        )
    with pytest.raises(ValueError, match="SensitivityDirection"):
        maeda_sensitivity_specification(
            MaedaProteinSystem.ATCRY1,
            SensitivityDimension.FORWARD_RATE,
            "baseline",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="MaedaSensitivitySpecification"):
        solve_maeda_sensitivity_yield(
            "specification",  # type: ignore[arg-type]
            (0.0, 0.0, 0.0),
        )
