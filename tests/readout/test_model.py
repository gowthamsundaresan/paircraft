from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm

from paircraft.readout import (
    CitationRelation,
    FourStateModel,
    FourStateRates,
    PopulationState,
    RateConstant,
    ReferenceInputs,
    SourceLocationStatus,
    Uncertainty,
    UncertaintyStatus,
    ground_state,
    ross_2026_reference_inputs,
)


def test_numerical_evolution_conserves_population_and_reaches_analytical_steady_state() -> None:
    reference = ross_2026_reference_inputs()
    model = FourStateModel(reference.rates, reference.field_on_yield)

    states = model.evolve(ground_state(), (0.0, 1e-9, 1.0, 2_000.0))
    analytical = model.steady_state()

    assert all(state.total == pytest.approx(sum(state.as_tuple()), abs=1e-10) for state in states)
    assert states[-1].as_tuple() == pytest.approx(analytical.as_tuple(), rel=1e-7, abs=1e-10)


def test_zero_excitation_is_valid_but_singular_steady_state_is_rejected() -> None:
    reference = ross_2026_reference_inputs()
    zero_excitation = replace(
        reference.rates,
        excitation=_rate_like(reference.rates.excitation, 0.0),
    )
    dark_model = FourStateModel(zero_excitation, reference.field_off_yield)

    assert dark_model.steady_state() == ground_state()
    assert dark_model.steady_state_emission_rate() == 0.0

    no_loss = replace(
        reference.rates,
        fluorescence=_rate_like(reference.rates.fluorescence, 0.0),
        electron_transfer=_rate_like(reference.rates.electron_transfer, 0.0),
    )
    with pytest.raises(ValueError, match="non-unique or singular"):
        FourStateModel(no_loss, reference.field_off_yield).steady_state()


def test_contrast_vanishes_at_low_excitation_and_approaches_high_intensity_plateau() -> None:
    reference = ross_2026_reference_inputs()
    low = FourStateModel(
        _with_excitation(reference.rates, 1e-9), reference.field_off_yield
    ).singlet_triplet_contrast()
    high = FourStateModel(
        _with_excitation(reference.rates, 1e9), reference.field_off_yield
    ).singlet_triplet_contrast()
    higher = FourStateModel(
        _with_excitation(reference.rates, 1e12), reference.field_off_yield
    ).singlet_triplet_contrast()

    assert low < 1e-6
    assert high > 0.1
    assert higher == pytest.approx(high, rel=0.02)


def test_recovery_rate_shifts_threshold_and_competing_recovery_suppresses_contrast() -> None:
    reference = ross_2026_reference_inputs()
    slow_rates = _with_excitation(reference.rates, 1.0)
    slow_recovery = FourStateModel(slow_rates, reference.field_off_yield).singlet_triplet_contrast()
    faster_rates = replace(
        slow_rates,
        recovery=_rate_like(reference.rates.recovery, 1.0),
    )
    faster_recovery = FourStateModel(
        faster_rates, reference.field_off_yield
    ).singlet_triplet_contrast()
    competing_rates = replace(
        _with_excitation(reference.rates, 1e12),
        recovery=_rate_like(reference.rates.recovery, 2e7),
    )
    competing_recovery = FourStateModel(
        competing_rates, reference.field_off_yield
    ).singlet_triplet_contrast()

    assert faster_recovery < slow_recovery
    assert competing_recovery < 0.1


def test_swapping_imposed_yield_order_reverses_fluorescence_change_sign() -> None:
    reference = ross_2026_reference_inputs()
    off_model = FourStateModel(reference.rates, reference.field_off_yield)
    on_model = FourStateModel(reference.rates, reference.field_on_yield)

    assert off_model.relative_steady_state_change(reference.field_on_yield) < 0
    assert on_model.relative_steady_state_change(reference.field_off_yield) > 0


def test_steady_state_change_matches_independently_evaluated_equation_22() -> None:
    reference = ross_2026_reference_inputs()
    rates = reference.rates
    k_exc = rates.excitation.value_per_second
    k_f = rates.fluorescence.value_per_second
    k_et = rates.electron_transfer.value_per_second
    k_sbet = rates.singlet_back_electron_transfer.value_per_second
    k_rec = rates.recovery.value_per_second
    phi_off = reference.field_off_yield.value
    phi_on = reference.field_on_yield.value
    a = k_exc / (k_f + k_et)

    def gamma(phi_s: float) -> float:
        return k_et * (k_rec + (1.0 - phi_s) * k_sbet) / (k_rec * (k_rec + k_sbet))

    equation_22 = a * (gamma(phi_off) - gamma(phi_on)) / (1.0 + a * (1.0 + gamma(phi_on)))
    observed = FourStateModel(rates, reference.field_off_yield).relative_steady_state_change(
        reference.field_on_yield
    )

    assert observed == pytest.approx(equation_22, rel=1e-6, abs=1e-12)


def test_moderate_rate_transient_matches_independent_matrix_exponential() -> None:
    reference = ross_2026_reference_inputs()
    rates = FourStateRates(
        excitation=_rate_like(reference.rates.excitation, 0.6),
        fluorescence=_rate_like(reference.rates.fluorescence, 2.0),
        electron_transfer=_rate_like(reference.rates.electron_transfer, 3.0),
        singlet_back_electron_transfer=_rate_like(
            reference.rates.singlet_back_electron_transfer, 0.4
        ),
        recovery=_rate_like(reference.rates.recovery, 0.1),
    )
    model = FourStateModel(rates, reference.field_off_yield)
    times = (0.0, 0.25, 1.0, 3.0)
    observed = model.evolve(ground_state(), times)
    initial = np.asarray(ground_state().as_tuple())

    for time, state in zip(times, observed, strict=True):
        independent = expm(model.transition_matrix() * time) @ initial
        assert state.as_tuple() == pytest.approx(independent, rel=1e-7, abs=1e-10)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda reference: FourStateModel(
                replace(
                    reference.rates,
                    recovery=_rate_like(reference.rates.recovery, -1.0),
                ),
                reference.field_off_yield,
            ),
            "finite and nonnegative",
        ),
        (
            lambda reference: FourStateModel(
                reference.rates,
                replace(reference.field_off_yield, value=1.1),
            ),
            "lie in",
        ),
        (lambda _reference: PopulationState(0.5, 0.0, 0.0, 0.0), "sum to"),
    ],
)
def test_invalid_physical_inputs_are_rejected(
    mutator: Callable[[ReferenceInputs], object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        mutator(ross_2026_reference_inputs())


def test_scientific_metadata_rejects_missing_or_inconsistent_provenance() -> None:
    reference = ross_2026_reference_inputs()
    metadata = reference.rates.excitation.metadata

    invalid_uncertainty_builders: tuple[Callable[[], Uncertainty], ...] = (
        lambda: replace(metadata.uncertainty, unit=""),
        lambda: Uncertainty(UncertaintyStatus.REPORTED, "s^-1", note="missing value"),
        lambda: Uncertainty(
            UncertaintyStatus.UNKNOWN,
            "s^-1",
            value=1.0,
            note="contradiction",
        ),
        lambda: replace(metadata.uncertainty, note=""),
    )
    for build_uncertainty in invalid_uncertainty_builders:
        with pytest.raises(ValueError):
            build_uncertainty()

    with pytest.raises(ValueError, match="source must be explicit"):
        replace(metadata, source="")
    with pytest.raises(ValueError, match="chemical_state must be explicit"):
        replace(metadata, chemical_state="")
    with pytest.raises(ValueError, match="citation chain"):
        replace(metadata, citation_chain=())
    with pytest.raises(ValueError, match="direct model source"):
        replace(metadata, source_location="different location")
    with pytest.raises(ValueError, match="units must match"):
        replace(metadata, uncertainty=replace(metadata.uncertainty, unit="1"))

    dimensionless = reference.field_off_yield.metadata
    with pytest.raises(ValueError, match="rate constants must use"):
        RateConstant(1.0, dimensionless)
    with pytest.raises(ValueError, match="singlet yield must be dimensionless"):
        replace(reference.field_off_yield, metadata=metadata)


def test_reference_citation_chains_preserve_origins_and_verification_status() -> None:
    reference = ross_2026_reference_inputs()
    inherited_or_motivating = (
        reference.rates.fluorescence,
        reference.rates.electron_transfer,
        reference.rates.singlet_back_electron_transfer,
        reference.rates.recovery,
    )

    for rate in inherited_or_motivating:
        direct, origin = rate.metadata.citation_chain
        assert direct.relation is CitationRelation.MODEL_SOURCE
        assert direct.location_status is SourceLocationStatus.VERIFIED
        assert origin.relation is CitationRelation.ORIGINATING_SOURCE
        assert origin.location_status is SourceLocationStatus.UNVERIFIED

    _, yield_origin = reference.field_off_yield.metadata.citation_chain
    assert yield_origin.identifier == "https://arxiv.org/abs/2604.18608"
    assert yield_origin.location_status is SourceLocationStatus.UNVERIFIED

    changed = reference.rates.with_excitation(_rate_like(reference.rates.excitation, 2.0))
    assert (
        changed.excitation.metadata.citation_chain
        == reference.rates.excitation.metadata.citation_chain
    )


def test_population_and_time_boundaries_are_explicit() -> None:
    reference = ross_2026_reference_inputs()
    model = FourStateModel(reference.rates, reference.field_off_yield)

    with pytest.raises(ValueError, match="finite and positive"):
        ground_state(0.0)
    with pytest.raises(ValueError, match="nonnegative"):
        PopulationState(1.1, -0.1, 0.0, 0.0)
    with pytest.raises(ValueError, match="at least one"):
        model.evolve(ground_state(), ())
    with pytest.raises(ValueError, match="finite and nonnegative"):
        model.evolve(ground_state(), (-1.0,))
    with pytest.raises(ValueError, match="strictly increasing"):
        model.evolve(ground_state(), (1.0, 0.5))
    assert model.evolve(ground_state(), (0.0,)) == (ground_state(),)
    with pytest.raises(ValueError, match="finite and positive"):
        model.steady_state(0.0)


def test_relative_observables_reject_a_dark_reference() -> None:
    reference = ross_2026_reference_inputs()
    dark = FourStateModel(_with_excitation(reference.rates, 0.0), reference.field_off_yield)

    with pytest.raises(ValueError, match="zero reference emission"):
        dark.relative_steady_state_change(reference.field_on_yield)
    with pytest.raises(ValueError, match="zero pure-singlet emission"):
        dark.singlet_triplet_contrast()


def _with_excitation(rates: FourStateRates, value: float) -> FourStateRates:
    return rates.with_excitation(_rate_like(rates.excitation, value))


def _rate_like(rate: RateConstant, value: float) -> RateConstant:
    return replace(rate, value_per_second=value)
