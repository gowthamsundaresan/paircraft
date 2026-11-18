from dataclasses import replace
from math import nan

import pytest

from paircraft.coupling import (
    CalculatedObservable,
    CoordinateFrame,
    MagneticFieldMetadata,
    MagneticFluxDensity,
    ReadoutProjectionPoint,
    ReadoutRunContract,
    SpinRunContract,
    SpinYieldCurve,
    SpinYieldPoint,
    SteadyStateReadoutProjection,
    VectorFieldUncertainty,
    project_steady_state_readout,
)
from paircraft.readout import (
    Citation,
    CitationRelation,
    EvidenceStatus,
    ParameterMetadata,
    RateConstant,
    SingletYield,
    SourceLocationStatus,
    Uncertainty,
    UncertaintyStatus,
    ross_2026_reference_inputs,
)

SOURCE = "https://example.test/toy-spin-run"
LOCATION = "toy fixture, record 1"
CHEMICAL_STATE = "Toy flavin radical pair in a declared redox and protonation state."
YIELD_CONDITIONS = "Toy calculation at 298.15 K with fixed parameter set."
FIELD_CONDITIONS = "Applied Cartesian B vector used by the toy calculation."


def _citation() -> Citation:
    return Citation(
        SOURCE,
        LOCATION,
        CitationRelation.MODEL_SOURCE,
        SourceLocationStatus.VERIFIED,
    )


def _frame() -> CoordinateFrame:
    return CoordinateFrame(
        name="toy-lab-frame",
        definition="Origin at sample center; axes fixed to the simulated sample geometry.",
        axis_convention="+z follows the coil axis; +x points toward the camera.",
        handedness="right",
    )


def _unknown_field_uncertainty() -> VectorFieldUncertainty:
    return VectorFieldUncertainty(
        status=UncertaintyStatus.UNKNOWN,
        component_standard_uncertainty_tesla=None,
        covariance_tesla_squared=None,
        source=SOURCE,
        source_location=LOCATION,
        citation_chain=(_citation(),),
        note="The toy spin calculation does not report field uncertainty.",
    )


def _reported_field_uncertainty() -> VectorFieldUncertainty:
    return VectorFieldUncertainty(
        status=UncertaintyStatus.REPORTED,
        component_standard_uncertainty_tesla=(1e-9, 2e-9, 3e-9),
        covariance_tesla_squared=(
            (1e-18, 0.5e-18, 0.0),
            (0.5e-18, 4e-18, 0.0),
            (0.0, 0.0, 9e-18),
        ),
        source=SOURCE,
        source_location=LOCATION,
        citation_chain=(_citation(),),
        note="Toy reported standard uncertainties and covariance.",
    )


def _run(frame: CoordinateFrame | None = None) -> SpinRunContract:
    return SpinRunContract(
        radical_pair_identity="toy flavin/tryptophan radical pair",
        chemical_state=CHEMICAL_STATE,
        computational_method="exact toy Hilbert-space calculation",
        engine_name="toy-independent-solver",
        engine_version="1.0",
        parameter_set_id="toy-parameter-set-v1",
        parameter_set_sha256="a" * 64,
        temperature_kelvin=298.15,
        solvent="water",
        viscosity_pascal_seconds=0.00089,
        orientation_regime="fixed molecular orientation in toy-lab-frame",
        yield_conditions=YIELD_CONDITIONS,
        field_conditions=FIELD_CONDITIONS,
        uncertainty_assumptions="No joint distribution or cross-field covariance reported.",
        coordinate_frame=frame or _frame(),
        source=SOURCE,
        source_location=LOCATION,
        citation_chain=(_citation(),),
    )


def _yield(value: float, *, conditions: str = YIELD_CONDITIONS) -> SingletYield:
    return SingletYield(
        value=value,
        metadata=ParameterMetadata(
            unit="1",
            evidence_status=EvidenceStatus.CALCULATED,
            source=SOURCE,
            source_location=LOCATION,
            citation_chain=(_citation(),),
            chemical_state=CHEMICAL_STATE,
            conditions=conditions,
            uncertainty=Uncertainty(
                status=UncertaintyStatus.REPORTED,
                unit="1",
                value=0.002,
                note="Marginal standard uncertainty only; cross-field covariance is unknown.",
            ),
        ),
    )


def _field(
    z_tesla: float,
    *,
    frame: CoordinateFrame | None = None,
    conditions: str = FIELD_CONDITIONS,
    uncertainty: VectorFieldUncertainty | None = None,
) -> MagneticFluxDensity:
    return MagneticFluxDensity(
        x_tesla=0.0,
        y_tesla=0.0,
        z_tesla=z_tesla,
        metadata=MagneticFieldMetadata(
            coordinate_frame=frame or _frame(),
            evidence_status=EvidenceStatus.CALCULATED,
            source=SOURCE,
            source_location=LOCATION,
            citation_chain=(_citation(),),
            conditions=conditions,
            uncertainty=uncertainty or _unknown_field_uncertainty(),
        ),
    )


def _curve(*, reverse: bool = False) -> SpinYieldCurve:
    points = (
        SpinYieldPoint("reference", _field(0.0), _yield(0.208)),
        SpinYieldPoint(
            "target",
            _field(50e-6, uncertainty=_reported_field_uncertainty()),
            _yield(0.167),
        ),
    )
    return SpinYieldCurve(_run(), tuple(reversed(points)) if reverse else points)


def _independent_emission(phi_s: float) -> float:
    rates = ross_2026_reference_inputs().rates
    k_exc = rates.excitation.value_per_second
    k_f = rates.fluorescence.value_per_second
    k_et = rates.electron_transfer.value_per_second
    k_sbet = rates.singlet_back_electron_transfer.value_per_second
    k_rec = rates.recovery.value_per_second
    a = k_exc / (k_f + k_et)
    gamma = k_et * (k_rec + (1.0 - phi_s) * k_sbet) / (k_rec * (k_rec + k_sbet))
    excited = a / (1.0 + a * (1.0 + gamma))
    return k_f * excited


def _readout_run() -> ReadoutRunContract:
    return ReadoutRunContract(ross_2026_reference_inputs().rates)


def test_projection_preserves_exact_scientific_inputs() -> None:
    curve = _curve()
    readout_run = _readout_run()

    projection = project_steady_state_readout(curve, readout_run, "reference")

    assert projection.input_curve is curve
    assert projection.readout_run is readout_run
    assert projection.readout_rates is readout_run.rates
    for input_point, output_point in zip(curve.points, projection.points, strict=True):
        assert output_point.input_point is input_point
        assert output_point.input_point.field is input_point.field
        assert output_point.input_point.field.metadata is input_point.field.metadata
        assert (
            output_point.input_point.field.metadata.uncertainty
            is input_point.field.metadata.uncertainty
        )
        assert output_point.input_point.singlet_yield is input_point.singlet_yield
        assert (
            output_point.input_point.singlet_yield.metadata.uncertainty
            is input_point.singlet_yield.metadata.uncertainty
        )


def test_reference_normalization_matches_independent_equation() -> None:
    projection = project_steady_state_readout(_curve(), _readout_run(), "reference")
    by_id = {point.input_point.point_id: point for point in projection.points}
    reference = _independent_emission(0.208)
    target = _independent_emission(0.167)

    assert by_id["reference"].emission_rate.value == pytest.approx(reference, rel=1e-12)
    assert by_id["reference"].relative_change.value == pytest.approx(0.0, abs=1e-15)
    assert by_id["target"].emission_rate.value == pytest.approx(target, rel=1e-12)
    assert by_id["target"].relative_change.value == pytest.approx(
        (target - reference) / reference, rel=1e-12
    )


def test_rate_change_changes_readout_without_mutating_curve() -> None:
    curve = _curve()
    reference_inputs = ross_2026_reference_inputs()
    changed_excitation = RateConstant(
        1.2,
        replace(
            reference_inputs.rates.excitation.metadata,
            conditions="Deliberately changed toy illumination for an invariance test.",
        ),
    )
    changed_rates = reference_inputs.rates.with_excitation(changed_excitation)

    first = project_steady_state_readout(
        curve, ReadoutRunContract(reference_inputs.rates), "reference"
    )
    second = project_steady_state_readout(curve, ReadoutRunContract(changed_rates), "reference")

    assert first.input_curve is curve
    assert second.input_curve is curve
    assert second.input_curve == first.input_curve
    assert second.points[0].emission_rate.value != first.points[0].emission_rate.value


def test_yield_change_changes_readout_without_refitting_rates() -> None:
    curve = _curve()
    rates = ross_2026_reference_inputs().rates
    changed_target = replace(curve.points[1], singlet_yield=_yield(0.18))
    changed_curve = SpinYieldCurve(curve.run, (curve.points[0], changed_target))

    first = project_steady_state_readout(curve, ReadoutRunContract(rates), "reference")
    second = project_steady_state_readout(changed_curve, ReadoutRunContract(rates), "reference")

    assert first.readout_rates is rates
    assert second.readout_rates is rates
    assert first.points[1].emission_rate.value != second.points[1].emission_rate.value


def test_point_order_does_not_change_id_associated_results() -> None:
    rates = ross_2026_reference_inputs().rates
    forward = project_steady_state_readout(_curve(), ReadoutRunContract(rates), "reference")
    reverse = project_steady_state_readout(
        _curve(reverse=True), ReadoutRunContract(rates), "reference"
    )

    forward_values = {
        point.input_point.point_id: (point.emission_rate.value, point.relative_change.value)
        for point in forward.points
    }
    reverse_values = {
        point.input_point.point_id: (point.emission_rate.value, point.relative_change.value)
        for point in reverse.points
    }
    assert reverse_values.keys() == forward_values.keys()
    for point_id in forward_values:
        assert reverse_values[point_id] == pytest.approx(forward_values[point_id], rel=1e-12)


def test_field_and_covariance_validation_rejects_invalid_data() -> None:
    with pytest.raises(ValueError, match="finite"):
        _field(nan)
    with pytest.raises(ValueError, match="symmetric"):
        VectorFieldUncertainty(
            status=UncertaintyStatus.REPORTED,
            component_standard_uncertainty_tesla=(1.0, 1.0, 1.0),
            covariance_tesla_squared=((1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            source=SOURCE,
            source_location=LOCATION,
            citation_chain=(_citation(),),
            note="Intentionally asymmetric invalid fixture.",
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        VectorFieldUncertainty(
            status=UncertaintyStatus.REPORTED,
            component_standard_uncertainty_tesla=(1.0, 1.0, 1.0),
            covariance_tesla_squared=((1.0, 2.0, 0.0), (2.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            source=SOURCE,
            source_location=LOCATION,
            citation_chain=(_citation(),),
            note="Intentionally indefinite invalid fixture.",
        )


def test_field_uncertainty_requires_a_complete_consistent_report() -> None:
    with pytest.raises(ValueError, match="requires components and covariance"):
        replace(
            _unknown_field_uncertainty(),
            status=UncertaintyStatus.REPORTED,
        )
    with pytest.raises(ValueError, match="diagonal"):
        replace(
            _reported_field_uncertainty(),
            covariance_tesla_squared=(
                (2e-18, 0.5e-18, 0.0),
                (0.5e-18, 4e-18, 0.0),
                (0.0, 0.0, 9e-18),
            ),
        )
    with pytest.raises(ValueError, match="unreported"):
        replace(
            _reported_field_uncertainty(),
            status=UncertaintyStatus.UNKNOWN,
        )
    with pytest.raises(ValueError, match="immutable tuples"):
        replace(
            _reported_field_uncertainty(),
            covariance_tesla_squared=[  # type: ignore[arg-type]
                (1e-18, 0.5e-18, 0.0),
                (0.5e-18, 4e-18, 0.0),
                (0.0, 0.0, 9e-18),
            ],
        )
    with pytest.raises(ValueError, match="exactly three"):
        replace(
            _reported_field_uncertainty(),
            component_standard_uncertainty_tesla=(1e-9,),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="diagonal"):
        replace(
            _reported_field_uncertainty(),
            component_standard_uncertainty_tesla=(1e-16, 1e-16, 1e-16),
            covariance_tesla_squared=(
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
        )


def test_run_and_coordinate_frame_require_complete_identity() -> None:
    with pytest.raises(ValueError, match="handedness"):
        replace(_frame(), handedness="clockwise")
    with pytest.raises(ValueError, match="64-character"):
        replace(_run(), parameter_set_sha256="not-a-content-hash")
    with pytest.raises(ValueError, match="temperature"):
        replace(_run(), temperature_kelvin=0.0)


def test_curve_identity_and_run_compatibility_are_enforced() -> None:
    point = SpinYieldPoint("same", _field(0.0), _yield(0.2))
    with pytest.raises(ValueError, match="immutable tuple"):
        SpinYieldCurve(_run(), [point])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unique"):
        SpinYieldCurve(_run(), (point, point))
    with pytest.raises(ValueError, match="yield conditions"):
        SpinYieldCurve(_run(), (replace(point, singlet_yield=_yield(0.2, conditions="other")),))
    with pytest.raises(ValueError, match="coordinate frame"):
        other_frame = replace(_frame(), name="other-frame")
        SpinYieldCurve(_run(), (replace(point, field=_field(0.0, frame=other_frame)),))


def test_missing_reference_and_zero_reference_emission_are_rejected() -> None:
    curve = _curve()
    rates = ross_2026_reference_inputs().rates
    with pytest.raises(ValueError, match="not present"):
        project_steady_state_readout(curve, ReadoutRunContract(rates), "missing")

    dark_excitation = RateConstant(
        0.0,
        replace(rates.excitation.metadata, conditions="Explicit zero-excitation test condition."),
    )
    dark_rates = rates.with_excitation(dark_excitation)
    with pytest.raises(ValueError, match="zero reference emission"):
        project_steady_state_readout(curve, ReadoutRunContract(dark_rates), "reference")


def test_calculated_output_metadata_is_explicit_and_does_not_invent_uncertainty() -> None:
    rates = ross_2026_reference_inputs().rates
    readout_run = ReadoutRunContract(rates)
    projection = project_steady_state_readout(_curve(), readout_run, "reference")

    for point in projection.points:
        for observable, expected_unit in (
            (point.emission_rate, "population s^-1"),
            (point.relative_change, "1"),
        ):
            metadata = observable.metadata
            assert metadata.evidence_status is EvidenceStatus.CALCULATED
            assert metadata.unit == expected_unit
            assert metadata.conditions == readout_run.conditions
            assert metadata.readout_rates is rates
            assert metadata.reference_point_id == "reference"
            assert metadata.population_total == 1.0
            assert metadata.uncertainty.status is UncertaintyStatus.UNKNOWN
            assert metadata.uncertainty.value is None
            assert "Not calculated" in metadata.uncertainty.note


def test_readout_conditions_are_derived_from_exact_rates() -> None:
    reference_rates = ross_2026_reference_inputs().rates
    illuminated = ReadoutRunContract(reference_rates)
    dark_metadata = replace(
        reference_rates.excitation.metadata,
        conditions="Explicit zero-excitation condition for the dark readout probe.",
    )
    dark = ReadoutRunContract(reference_rates.with_excitation(RateConstant(0.0, dark_metadata)))

    assert illuminated.conditions.startswith(
        f"excitation={reference_rates.excitation.value_per_second:.17g} s^-1"
    )
    assert "excitation=0 s^-1" not in illuminated.conditions
    assert dark.conditions.startswith("excitation=0 s^-1")
    illuminated_projection = project_steady_state_readout(_curve(), illuminated, "reference")
    assert (
        illuminated_projection.points[0].emission_rate.metadata.conditions == illuminated.conditions
    )


def test_projection_constructor_rejects_mutable_or_unrelated_points() -> None:
    projection = project_steady_state_readout(_curve(), _readout_run(), "reference")
    with pytest.raises(ValueError, match="immutable tuple"):
        SteadyStateReadoutProjection(
            projection.input_curve,
            projection.readout_run,
            projection.reference_point_id,
            list(projection.points),  # type: ignore[arg-type]
        )

    unrelated = SpinYieldPoint("unrelated", _field(1e-6), _yield(0.19))
    unrelated_result = ReadoutProjectionPoint(
        unrelated,
        projection.points[0].emission_rate,
        projection.points[0].relative_change,
    )
    with pytest.raises(ValueError, match="exact input point identities"):
        SteadyStateReadoutProjection(
            projection.input_curve,
            projection.readout_run,
            projection.reference_point_id,
            (unrelated_result, projection.points[1]),
        )


def test_projection_constructor_rejects_reference_and_metadata_mismatches() -> None:
    projection = project_steady_state_readout(_curve(), _readout_run(), "reference")
    with pytest.raises(ValueError, match="reference must exist"):
        SteadyStateReadoutProjection(
            projection.input_curve,
            projection.readout_run,
            "absent",
            projection.points,
        )

    first = projection.points[0]
    other_excitation_metadata = replace(
        projection.readout_rates.excitation.metadata,
        conditions="Deliberately changed excitation condition for a constructor probe.",
    )
    other_rates = projection.readout_rates.with_excitation(
        RateConstant(1.2, other_excitation_metadata)
    )
    wrong_rates_emission = CalculatedObservable(
        first.emission_rate.value,
        replace(first.emission_rate.metadata, readout_rates=other_rates),
    )
    wrong_rates_point = ReadoutProjectionPoint(
        first.input_point,
        wrong_rates_emission,
        first.relative_change,
    )
    with pytest.raises(ValueError, match="exact readout rates"):
        SteadyStateReadoutProjection(
            projection.input_curve,
            projection.readout_run,
            projection.reference_point_id,
            (wrong_rates_point, projection.points[1]),
        )

    wrong_reference_change = CalculatedObservable(
        first.relative_change.value,
        replace(first.relative_change.metadata, reference_point_id="target"),
    )
    wrong_reference_point = ReadoutProjectionPoint(
        first.input_point,
        first.emission_rate,
        wrong_reference_change,
    )
    with pytest.raises(ValueError, match="metadata reference"):
        SteadyStateReadoutProjection(
            projection.input_curve,
            projection.readout_run,
            projection.reference_point_id,
            (wrong_reference_point, projection.points[1]),
        )


def test_repeat_field_vectors_are_allowed_for_distinct_ensemble_members() -> None:
    shared_field = _field(50e-6)
    curve = SpinYieldCurve(
        _run(),
        (
            SpinYieldPoint("ensemble-a", shared_field, _yield(0.20)),
            SpinYieldPoint("ensemble-b", shared_field, _yield(0.21)),
        ),
    )

    result = project_steady_state_readout(curve, _readout_run(), "ensemble-a")
    assert [point.input_point.point_id for point in result.points] == ["ensemble-a", "ensemble-b"]


def test_citation_and_field_metadata_fail_closed_on_provenance_mutations() -> None:
    uncertainty = _unknown_field_uncertainty()
    metadata = _field(0.0).metadata

    with pytest.raises(ValueError, match="immutable tuple"):
        replace(uncertainty, citation_chain=[_citation()])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least the direct source"):
        replace(uncertainty, citation_chain=())
    with pytest.raises(ValueError, match="declared direct source"):
        replace(uncertainty, source="https://example.test/different")
    with pytest.raises(ValueError, match="tesla internally"):
        replace(metadata, unit="mT")
    with pytest.raises(ValueError, match="field conditions"):
        SpinYieldCurve(
            _run(),
            (SpinYieldPoint("point", _field(0.0, conditions="different"), _yield(0.2)),),
        )


@pytest.mark.parametrize("components", [(nan, 1.0, 1.0), (-1.0, 1.0, 1.0)])
def test_reported_field_uncertainty_rejects_invalid_components(
    components: tuple[float, float, float],
) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        replace(
            _reported_field_uncertainty(),
            component_standard_uncertainty_tesla=components,
        )


def test_spin_run_and_curve_require_physical_identity() -> None:
    point = SpinYieldPoint("point", _field(0.0), _yield(0.2))
    with pytest.raises(ValueError, match="viscosity"):
        replace(_run(), viscosity_pascal_seconds=-1.0)
    with pytest.raises(ValueError, match="at least one point"):
        SpinYieldCurve(_run(), ())
    with pytest.raises(ValueError, match="chemical state"):
        SpinYieldCurve(
            _run(),
            (
                replace(
                    point,
                    singlet_yield=replace(
                        point.singlet_yield,
                        metadata=replace(
                            point.singlet_yield.metadata,
                            chemical_state="different state",
                        ),
                    ),
                ),
            ),
        )
    with pytest.raises(ValueError, match="point ID"):
        replace(point, point_id=" ")


def test_calculated_observable_metadata_rejects_scientific_claim_mutations() -> None:
    point = project_steady_state_readout(_curve(), _readout_run(), "reference").points[0]
    metadata = point.emission_rate.metadata

    with pytest.raises(ValueError, match="marked calculated"):
        replace(metadata, evidence_status=EvidenceStatus.ASSUMED)
    with pytest.raises(ValueError, match="population total"):
        replace(metadata, population_total=0.99)
    with pytest.raises(ValueError, match="units must match"):
        replace(metadata, uncertainty=replace(metadata.uncertainty, unit="1"))
    with pytest.raises(ValueError, match="must be finite"):
        replace(point.emission_rate, value=nan)


def test_projection_rejects_missing_and_contradictory_outputs() -> None:
    projection = project_steady_state_readout(_curve(), _readout_run(), "reference")
    with pytest.raises(ValueError, match="exactly one result"):
        replace(projection, points=projection.points[:-1])

    first = projection.points[0]
    wrong_conditions = CalculatedObservable(
        first.emission_rate.value,
        replace(first.emission_rate.metadata, conditions="different rates"),
    )
    with pytest.raises(ValueError, match="conditions do not match"):
        replace(
            projection,
            points=(
                ReadoutProjectionPoint(
                    first.input_point,
                    wrong_conditions,
                    first.relative_change,
                ),
                projection.points[1],
            ),
        )
