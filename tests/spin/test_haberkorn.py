from math import inf, nan

import pytest

from paircraft.spin.haberkorn import (
    _ReactionYields,
    _ReducedIsotropicHaberkornSystem,
    _solve_reaction_yields,
)


def _system(
    *,
    hyperfine_hz: float = 12_330_000.0,
    singlet_rate: float = 1_000_000.0,
    triplet_rate: float = 1_000_000.0,
) -> _ReducedIsotropicHaberkornSystem:
    return _ReducedIsotropicHaberkornSystem(
        hyperfine_hz=hyperfine_hz,
        singlet_rate_per_second=singlet_rate,
        triplet_rate_per_second=triplet_rate,
        electron_gyromagnetic_ratio_rad_per_second_tesla=176_085_963_023.0,
    )


@pytest.mark.parametrize(
    ("field_tesla", "expected_singlet"),
    [
        ((0.0, 0.0, 0.0), 0.6250624702183424),
        ((0.0, 0.0, 50e-6), 0.3891794521481556),
        ((0.0, 0.0, 1e-3), 0.4800936038635648),
    ],
)
def test_yields_match_pinned_radicalpy_reference(
    field_tesla: tuple[float, float, float],
    expected_singlet: float,
) -> None:
    """Check RadicalPy 3f547bd5 with nuclear Zeeman explicitly omitted."""

    result = _solve_reaction_yields(_system(), field_tesla)

    assert result.singlet == pytest.approx(expected_singlet, abs=1e-14)
    assert result.total == pytest.approx(1.0, abs=1e-12)


def test_zero_hyperfine_preserves_the_initial_singlet() -> None:
    result = _solve_reaction_yields(_system(hyperfine_hz=0.0), (0.0, 0.0, 50e-6))

    assert result.singlet == pytest.approx(1.0, abs=1e-12)
    assert result.triplet == pytest.approx(0.0, abs=1e-12)


def test_unequal_rates_still_produce_bounded_conserved_yields() -> None:
    result = _solve_reaction_yields(
        _system(singlet_rate=3e5, triplet_rate=2e6),
        (10e-6, -20e-6, 30e-6),
    )

    assert 0.0 <= result.singlet <= 1.0
    assert 0.0 <= result.triplet <= 1.0
    assert result.total == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hyperfine_hz", nan),
        ("singlet_rate_per_second", 0.0),
        ("triplet_rate_per_second", -1.0),
        ("electron_gyromagnetic_ratio_rad_per_second_tesla", inf),
    ],
)
def test_invalid_system_parameters_are_rejected(field: str, value: float) -> None:
    parameters = {
        "hyperfine_hz": 12_330_000.0,
        "singlet_rate_per_second": 1_000_000.0,
        "triplet_rate_per_second": 1_000_000.0,
        "electron_gyromagnetic_ratio_rad_per_second_tesla": 176_085_963_023.0,
    }
    parameters[field] = value

    with pytest.raises(ValueError):
        _ReducedIsotropicHaberkornSystem(**parameters)


def test_nonfinite_field_is_rejected_before_solving() -> None:
    with pytest.raises(ValueError, match="three finite tesla components"):
        _solve_reaction_yields(_system(), (0.0, 0.0, nan))


def test_internal_result_rejects_nonphysical_yields() -> None:
    with pytest.raises(ValueError, match="lie in"):
        _ReactionYields(singlet=-1.0, triplet=2.0)
