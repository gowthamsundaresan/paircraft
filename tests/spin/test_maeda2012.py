from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

import paircraft.spin._maeda_kernel as kernel_module
import paircraft.spin.maeda2012 as maeda_module
from paircraft.spin import (
    MaedaProteinSystem,
    ParameterEvidence,
    PublishedHyperfine,
    PublishedParameter,
    SecondaryPairYield,
    maeda_2012_specification,
    solve_maeda_secondary_pair_yield,
)


@pytest.fixture(scope="module")
def curves() -> dict[MaedaProteinSystem, dict[float, SecondaryPairYield]]:
    return {
        system: {
            field_mt: solve_maeda_secondary_pair_yield(
                maeda_2012_specification(system),
                (0.0, 0.0, field_mt * 1e-3),
            )
            for field_mt in (0.0, 1.0, 10.0, 30.0)
        }
        for system in MaedaProteinSystem
    }


def _relative(target: SecondaryPairYield, reference: SecondaryPairYield) -> float:
    return (target.secondary_pair - reference.secondary_pair) / reference.secondary_pair


def test_published_specifications_retain_values_units_and_evidence() -> None:
    atcry = maeda_2012_specification(MaedaProteinSystem.ATCRY1)
    ecpl = maeda_2012_specification(MaedaProteinSystem.ECPL_E109A)

    assert [item.equivalent_field.value_si for item in atcry.hyperfine_parameters] == [
        523.3e-6,
        1106.5e-6,
        1604.6e-6,
        -598.3e-6,
        915.7e-6,
    ]
    assert [item.spin_quantum_number for item in atcry.hyperfine_parameters] == [
        1.0,
        0.5,
        0.5,
        0.5,
        0.5,
    ]
    assert atcry.forward_rate.value_si == 2.5e5
    assert atcry.forward_rate.evidence_status is ParameterEvidence.ASSUMED
    assert atcry.back_recombination_rate.value_si == 4.9e5
    assert ecpl.back_recombination_rate.value_si == 1.2e5
    assert atcry.st_dephasing_rate.value_si == 1.1e7
    assert ecpl.st_dephasing_rate.value_si == 2.7e7
    assert atcry.electron_gyromagnetic_ratio.value_si == pytest.approx(
        175_882_000_837.07785,
        rel=1e-15,
    )
    assert "270 K" in atcry.forward_rate.conditions
    assert "260 K" in ecpl.forward_rate.conditions
    assert "PMC3323948/pdf" in atcry.back_recombination_rate.source_uri
    assert atcry.electron_gyromagnetic_ratio.uncertainty_status == "not_propagated"
    assert len(atcry.hyperfine_parameters[0].equivalent_field.citation_chain) == 3
    assert all(item.equivalent_field.unit == "T" for item in atcry.hyperfine_parameters)
    assert len(atcry.parameter_set_sha256) == 64
    assert atcry.parameter_set_sha256 != ecpl.parameter_set_sha256


def test_zero_hyperfine_limit_matches_competing_first_order_channels() -> None:
    specification = maeda_2012_specification(MaedaProteinSystem.ATCRY1)
    zero_hyperfine = replace(
        specification,
        hyperfine_parameters=tuple(
            replace(
                item,
                equivalent_field=replace(item.equivalent_field, value_si=0.0),
            )
            for item in specification.hyperfine_parameters
        ),
    )

    result = maeda_module._solve_maeda_secondary_pair_yield(
        zero_hyperfine,
        (0.0, 0.0, 10e-3),
    )
    expected_secondary = specification.forward_rate.value_si / (
        specification.forward_rate.value_si + specification.back_recombination_rate.value_si
    )

    assert result.secondary_pair == pytest.approx(expected_secondary, abs=1e-12)
    assert result.singlet_back_recombination == pytest.approx(1 - expected_secondary, abs=1e-12)


def test_dephasing_damps_only_singlet_triplet_coherences() -> None:
    singlet = np.diag([1.0, 0.0]).astype(np.complex128)
    triplet = np.diag([0.0, 1.0]).astype(np.complex128)
    density = np.array(((2.0, 3.0j), (-4.0j, 5.0)), dtype=np.complex128)

    derivative = maeda_module._st_dephasing(singlet, triplet, density, 2.0)

    assert derivative == pytest.approx(np.array(((0.0, -6.0j), (8.0j, 0.0))))


def test_published_rate_sets_reproduce_paper_level_curve_topology(
    curves: dict[MaedaProteinSystem, dict[float, SecondaryPairYield]],
) -> None:
    atcry = curves[MaedaProteinSystem.ATCRY1]
    ecpl = curves[MaedaProteinSystem.ECPL_E109A]
    atcry_effects = {field: _relative(result, atcry[0.0]) for field, result in atcry.items()}
    ecpl_effects = {field: _relative(result, ecpl[0.0]) for field, result in ecpl.items()}

    assert atcry[0.0].secondary_pair == pytest.approx(0.6466039595709816, abs=1e-15)
    assert ecpl[0.0].secondary_pair == pytest.approx(0.8825586735769018, abs=1e-15)
    assert 0.01 < atcry_effects[1.0] < 0.03
    assert -0.16 < atcry_effects[10.0] < -0.10
    assert -0.25 < atcry_effects[30.0] < -0.18
    assert 0.003 < ecpl_effects[1.0] < 0.01
    assert -0.05 < ecpl_effects[10.0] < -0.02
    assert -0.10 < ecpl_effects[30.0] < -0.05
    assert abs(atcry_effects[30.0]) > abs(ecpl_effects[30.0])
    assert all(
        result.total == pytest.approx(1.0, abs=1e-7)
        for curve in curves.values()
        for result in curve.values()
    )


def test_isotropic_model_is_invariant_to_field_direction(
    curves: dict[MaedaProteinSystem, dict[float, SecondaryPairYield]],
) -> None:
    specification = maeda_2012_specification(MaedaProteinSystem.ATCRY1)
    x_directed = solve_maeda_secondary_pair_yield(specification, (1e-3, 0.0, 0.0))

    assert x_directed.secondary_pair == pytest.approx(
        curves[MaedaProteinSystem.ATCRY1][1.0].secondary_pair,
        abs=1e-12,
    )


def test_parameter_hash_commits_to_coupling_sign_and_provenance() -> None:
    specification = maeda_2012_specification(MaedaProteinSystem.ATCRY1)
    first = specification.hyperfine_parameters[0]
    changed_hyperfine = replace(
        first,
        equivalent_field=replace(
            first.equivalent_field,
            value_si=-first.equivalent_field.value_si,
        ),
    )
    changed = replace(
        specification,
        hyperfine_parameters=(changed_hyperfine, *specification.hyperfine_parameters[1:]),
    )

    assert changed.parameter_set_sha256 != specification.parameter_set_sha256
    with pytest.raises(ValueError, match="exact published specification"):
        solve_maeda_secondary_pair_yield(changed, (0.0, 0.0, 1e-3))


def test_nonconvergence_fails_without_a_numeric_result(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_gmres(*args: object, **kwargs: object) -> tuple[np.ndarray, int]:
        del args, kwargs
        return np.zeros(maeda_module._LIOUVILLE_DIM, dtype=np.complex128), 1

    monkeypatch.setattr(kernel_module, "gmres", fail_gmres)

    with pytest.raises(ArithmeticError, match="failed to converge"):
        solve_maeda_secondary_pair_yield(
            maeda_2012_specification(MaedaProteinSystem.ATCRY1),
            (0.0, 0.0, 0.0),
        )


def test_invalid_public_inputs_are_rejected() -> None:
    specification = maeda_2012_specification(MaedaProteinSystem.ATCRY1)
    parameter = specification.forward_rate

    with pytest.raises(ValueError, match="64 hexadecimal"):
        replace(parameter, source_artifact_sha256="not-a-digest")
    with pytest.raises(ValueError, match="spin-1/2 and spin-1"):
        PublishedHyperfine("FAD radical anion", "X", 1.5, parameter)
    with pytest.raises(ValueError, match="three finite"):
        solve_maeda_secondary_pair_yield(
            maeda_2012_specification(MaedaProteinSystem.ATCRY1),
            (0.0, 0.0, float("nan")),
        )
    with pytest.raises(ValueError, match="relative tolerance"):
        solve_maeda_secondary_pair_yield(
            specification,
            (0.0, 0.0, 0.0),
            relative_tolerance=0.0,
        )
    with pytest.raises(ValueError, match="strictly positive"):
        replace(specification, forward_rate=replace(parameter, value_si=0.0))
    with pytest.raises(ValueError, match="published 0-50 mT domain"):
        solve_maeda_secondary_pair_yield(specification, (0.0, 0.0, 0.050001))


def test_result_cannot_contradict_its_physics_identity(
    curves: dict[MaedaProteinSystem, dict[float, SecondaryPairYield]],
) -> None:
    result = curves[MaedaProteinSystem.ATCRY1][0.0]

    with pytest.raises(ValueError, match="lie in"):
        replace(result, secondary_pair=-0.1, singlet_back_recombination=1.1)
    with pytest.raises(ValueError, match="model identity"):
        replace(result, model_id="another-model")
    with pytest.raises(ValueError, match="parameter hash"):
        replace(result, parameter_set_sha256="0" * 64)
    with pytest.raises(ValueError, match="solver diagnostics"):
        replace(result, relative_residual=float("nan"))
    with pytest.raises(ValueError, match="solver configuration"):
        replace(result, restart=0)
    with pytest.raises(ValueError, match="ideal simulation target"):
        replace(result, field_intent="measured_field")
    with pytest.raises(ValueError, match="code identity"):
        replace(result, code_artifact_sha256="not-a-digest")


def test_exact_result_artifact_identity_includes_neutral_kernel() -> None:
    result = solve_maeda_secondary_pair_yield(
        maeda_2012_specification(MaedaProteinSystem.ATCRY1),
        (0.0, 0.0, 0.0),
    )
    digest = sha256()
    for module in (maeda_module, kernel_module):
        digest.update(Path(module.__file__ or "").read_bytes())

    assert result.code_artifact_sha256 == digest.hexdigest()


def test_published_parameter_rejects_missing_provenance() -> None:
    with pytest.raises(ValueError, match="provenance fields"):
        PublishedParameter(
            parameter_id="",
            value_si=1.0,
            unit="T",
            chemical_state="state",
            conditions="conditions",
            evidence_status=ParameterEvidence.ASSUMED,
            uncertainty_status="not_reported",
            uncertainty_note="note",
            source_uri="source",
            exact_source_location="location",
            source_artifact_sha256=None,
            citation_chain=(),
            license="unknown",
        )
