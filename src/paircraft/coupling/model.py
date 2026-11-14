"""Engine-neutral coupling of imposed spin yields to steady-state readout kinetics."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from string import hexdigits
from typing import Final

import numpy as np
from numpy.typing import NDArray

from paircraft import __version__
from paircraft.readout import (
    Citation,
    CitationRelation,
    EvidenceStatus,
    FourStateModel,
    FourStateRates,
    SingletYield,
    Uncertainty,
    UncertaintyStatus,
)

_MODEL_IDENTIFIER: Final = "paircraft.readout.FourStateModel"
_MODEL_VERSION: Final = __version__
_MODEL_SOURCE: Final = "https://doi.org/10.64898/2026.07.08.737352"
_PSD_RELATIVE_TOLERANCE: Final = 1e-12

type Vector3 = tuple[float, float, float]
type Matrix3 = tuple[Vector3, Vector3, Vector3]


def _require_text(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be explicit")


def _validate_direct_citation(
    source: str, source_location: str, citation_chain: tuple[Citation, ...]
) -> None:
    if not isinstance(citation_chain, tuple):
        raise ValueError("citation chain must be an immutable tuple")
    if not citation_chain:
        raise ValueError("citation chain must retain at least the direct source")
    direct = citation_chain[0]
    if (
        direct.identifier != source
        or direct.location != source_location
        or direct.relation is not CitationRelation.MODEL_SOURCE
    ):
        raise ValueError("citation chain must begin with the declared direct source")


def _scale_relative_equal(left: NDArray[np.float64], right: NDArray[np.float64]) -> bool:
    """Compare values relatively while treating exact zero as a meaningful report."""

    if left.shape != right.shape:
        return False
    for left_value, right_value in zip(left.flat, right.flat, strict=True):
        if left_value == right_value:
            continue
        if left_value == 0.0 or right_value == 0.0:
            return False
        scale = max(abs(float(left_value)), abs(float(right_value)))
        if abs(float(left_value - right_value)) > scale * _PSD_RELATIVE_TOLERANCE:
            return False
    return True


@dataclass(frozen=True, slots=True)
class CoordinateFrame:
    """A complete Cartesian coordinate-frame definition for vector fields."""

    name: str
    definition: str
    axis_convention: str
    handedness: str

    def __post_init__(self) -> None:
        for name in ("name", "definition", "axis_convention", "handedness"):
            _require_text(getattr(self, name), name)
        if self.handedness not in {"left", "right"}:
            raise ValueError("handedness must be 'left' or 'right'")


@dataclass(frozen=True, slots=True)
class VectorFieldUncertainty:
    """Component uncertainty and covariance for one magnetic-flux-density vector."""

    status: UncertaintyStatus
    component_standard_uncertainty_tesla: Vector3 | None
    covariance_tesla_squared: Matrix3 | None
    source: str
    source_location: str
    citation_chain: tuple[Citation, ...]
    note: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.source, "uncertainty source"),
            (self.source_location, "uncertainty source location"),
            (self.note, "uncertainty note"),
        ):
            _require_text(value, name)
        _validate_direct_citation(self.source, self.source_location, self.citation_chain)

        if self.status is not UncertaintyStatus.REPORTED:
            if (
                self.component_standard_uncertainty_tesla is not None
                or self.covariance_tesla_squared is not None
            ):
                raise ValueError("unreported field uncertainty cannot carry numeric values")
            return

        components = self.component_standard_uncertainty_tesla
        covariance = self.covariance_tesla_squared
        if components is None or covariance is None:
            raise ValueError("reported field uncertainty requires components and covariance")
        if (
            not isinstance(components, tuple)
            or not isinstance(covariance, tuple)
            or any(not isinstance(row, tuple) for row in covariance)
        ):
            raise ValueError("field uncertainty components and covariance must be immutable tuples")
        if len(components) != 3:
            raise ValueError("field uncertainty requires exactly three component values")
        if any(not isfinite(value) or value < 0 for value in components):
            raise ValueError("component field uncertainties must be finite and nonnegative")

        matrix = np.asarray(covariance, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("field covariance must be a finite 3x3 matrix")
        if not _scale_relative_equal(matrix, matrix.T):
            raise ValueError("field covariance must be symmetric")
        expected_diagonal = np.square(np.asarray(components, dtype=np.float64))
        if not _scale_relative_equal(np.diag(matrix), expected_diagonal):
            raise ValueError("field covariance diagonal must match component uncertainties")
        eigenvalues = np.linalg.eigvalsh(matrix)
        scale = float(np.max(np.abs(matrix)))
        if scale != 0.0 and float(np.min(eigenvalues)) < -scale * _PSD_RELATIVE_TOLERANCE:
            raise ValueError("field covariance must be positive semidefinite")


@dataclass(frozen=True, slots=True)
class MagneticFieldMetadata:
    """Auditable metadata for a magnetic flux density B in tesla."""

    coordinate_frame: CoordinateFrame
    evidence_status: EvidenceStatus
    source: str
    source_location: str
    citation_chain: tuple[Citation, ...]
    conditions: str
    uncertainty: VectorFieldUncertainty
    unit: str = "T"

    def __post_init__(self) -> None:
        if self.unit != "T":
            raise ValueError("magnetic flux density B must use tesla internally")
        for value, name in (
            (self.source, "field source"),
            (self.source_location, "field source location"),
            (self.conditions, "field conditions"),
        ):
            _require_text(value, name)
        _validate_direct_citation(self.source, self.source_location, self.citation_chain)


@dataclass(frozen=True, slots=True)
class MagneticFluxDensity:
    """A full Cartesian magnetic flux density B with auditable metadata."""

    x_tesla: float
    y_tesla: float
    z_tesla: float
    metadata: MagneticFieldMetadata

    def __post_init__(self) -> None:
        if any(not isfinite(value) for value in self.components_tesla):
            raise ValueError("magnetic flux density components must be finite")

    @property
    def components_tesla(self) -> Vector3:
        """Return signed Cartesian components in tesla."""

        return (self.x_tesla, self.y_tesla, self.z_tesla)


@dataclass(frozen=True, slots=True)
class SpinRunContract:
    """Scientific identity shared by every point in one spin-yield curve."""

    radical_pair_identity: str
    chemical_state: str
    computational_method: str
    engine_name: str
    engine_version: str
    parameter_set_id: str
    parameter_set_sha256: str
    temperature_kelvin: float
    solvent: str
    viscosity_pascal_seconds: float
    orientation_regime: str
    yield_conditions: str
    field_conditions: str
    uncertainty_assumptions: str
    coordinate_frame: CoordinateFrame
    source: str
    source_location: str
    citation_chain: tuple[Citation, ...]

    def __post_init__(self) -> None:
        text_fields = (
            "radical_pair_identity",
            "chemical_state",
            "computational_method",
            "engine_name",
            "engine_version",
            "parameter_set_id",
            "parameter_set_sha256",
            "solvent",
            "orientation_regime",
            "yield_conditions",
            "field_conditions",
            "uncertainty_assumptions",
            "source",
            "source_location",
        )
        for name in text_fields:
            _require_text(getattr(self, name), name)
        digest = self.parameter_set_sha256.lower()
        if len(digest) != 64 or any(character not in hexdigits for character in digest):
            raise ValueError("parameter_set_sha256 must be a 64-character hexadecimal digest")
        if not isfinite(self.temperature_kelvin) or self.temperature_kelvin <= 0:
            raise ValueError("temperature must be finite and positive in kelvin")
        if not isfinite(self.viscosity_pascal_seconds) or self.viscosity_pascal_seconds < 0:
            raise ValueError("viscosity must be finite and nonnegative in Pa s")
        _validate_direct_citation(self.source, self.source_location, self.citation_chain)


@dataclass(frozen=True, slots=True)
class SpinYieldPoint:
    """One externally calculated singlet yield at one identified field sample."""

    point_id: str
    field: MagneticFluxDensity
    singlet_yield: SingletYield

    def __post_init__(self) -> None:
        _require_text(self.point_id, "point ID")


@dataclass(frozen=True, slots=True)
class SpinYieldCurve:
    """A coherent set of imposed singlet yields from one spin calculation run."""

    run: SpinRunContract
    points: tuple[SpinYieldPoint, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.points, tuple):
            raise ValueError("spin-yield curve points must be an immutable tuple")
        if not self.points:
            raise ValueError("spin-yield curve requires at least one point")
        point_ids = [point.point_id for point in self.points]
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("spin-yield curve point IDs must be unique")
        for point in self.points:
            yield_metadata = point.singlet_yield.metadata
            field_metadata = point.field.metadata
            if yield_metadata.chemical_state != self.run.chemical_state:
                raise ValueError("yield chemical state contradicts the spin-run contract")
            if yield_metadata.conditions != self.run.yield_conditions:
                raise ValueError("yield conditions contradict the spin-run contract")
            if field_metadata.coordinate_frame != self.run.coordinate_frame:
                raise ValueError("field coordinate frame contradicts the spin-run contract")
            if field_metadata.conditions != self.run.field_conditions:
                raise ValueError("field conditions contradict the spin-run contract")


@dataclass(frozen=True, slots=True)
class CalculatedObservableMetadata:
    """Audit metadata for a value calculated by the coupling operation."""

    unit: str
    conditions: str
    readout_rates: FourStateRates
    reference_point_id: str
    uncertainty: Uncertainty
    evidence_status: EvidenceStatus = EvidenceStatus.CALCULATED
    model_identifier: str = _MODEL_IDENTIFIER
    model_version: str = _MODEL_VERSION
    model_source: str = _MODEL_SOURCE
    population_total: float = 1.0

    def __post_init__(self) -> None:
        for value, name in (
            (self.unit, "observable unit"),
            (self.conditions, "readout conditions"),
            (self.reference_point_id, "reference point ID"),
            (self.model_identifier, "model identifier"),
            (self.model_version, "model version"),
            (self.model_source, "model source"),
        ):
            _require_text(value, name)
        if self.evidence_status is not EvidenceStatus.CALCULATED:
            raise ValueError("coupled observables must be marked calculated")
        if self.population_total != 1.0:
            raise ValueError("the coupling interface uses normalized population total 1.0")
        if self.uncertainty.unit != self.unit:
            raise ValueError("observable and uncertainty units must match")


@dataclass(frozen=True, slots=True)
class CalculatedObservable:
    """One finite calculated observable and its scientific metadata."""

    value: float
    metadata: CalculatedObservableMetadata

    def __post_init__(self) -> None:
        if not isfinite(self.value):
            raise ValueError("calculated observable must be finite")


@dataclass(frozen=True, slots=True)
class ReadoutRunContract:
    """One exact kinetic configuration for a steady-state readout projection."""

    rates: FourStateRates

    @property
    def conditions(self) -> str:
        """Derive conditions from exact values and provenance-bearing rate metadata."""

        named_rates = (
            ("excitation", self.rates.excitation),
            ("fluorescence", self.rates.fluorescence),
            ("electron_transfer", self.rates.electron_transfer),
            ("singlet_back_electron_transfer", self.rates.singlet_back_electron_transfer),
            ("recovery", self.rates.recovery),
        )
        return "; ".join(
            f"{name}={rate.value_per_second:.17g} s^-1 [{rate.metadata.conditions}]"
            for name, rate in named_rates
        )


@dataclass(frozen=True, slots=True)
class ReadoutProjectionPoint:
    """One unchanged spin-yield input and its steady-state readout projection."""

    input_point: SpinYieldPoint
    emission_rate: CalculatedObservable
    relative_change: CalculatedObservable


@dataclass(frozen=True, slots=True)
class SteadyStateReadoutProjection:
    """A coupled toy projection, not a calibrated protein prediction."""

    input_curve: SpinYieldCurve
    readout_run: ReadoutRunContract
    reference_point_id: str
    points: tuple[ReadoutProjectionPoint, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.points, tuple):
            raise ValueError("readout projection points must be an immutable tuple")
        input_ids = tuple(point.point_id for point in self.input_curve.points)
        if self.reference_point_id not in input_ids:
            raise ValueError("projection reference must exist in the input curve")
        if len(self.points) != len(self.input_curve.points):
            raise ValueError("projection must contain exactly one result per input point")
        expected_conditions = self.readout_run.conditions
        for input_point, projected_point in zip(self.input_curve.points, self.points, strict=True):
            if projected_point.input_point is not input_point:
                raise ValueError("projection points must retain exact input point identities")
            for observable in (
                projected_point.emission_rate,
                projected_point.relative_change,
            ):
                metadata = observable.metadata
                if metadata.readout_rates is not self.readout_run.rates:
                    raise ValueError("projection metadata must retain exact readout rates")
                if metadata.reference_point_id != self.reference_point_id:
                    raise ValueError("projection metadata reference does not match the projection")
                if metadata.conditions != expected_conditions:
                    raise ValueError("projection metadata conditions do not match exact rates")

    @property
    def readout_rates(self) -> FourStateRates:
        """Return the exact provenance-bearing rates used by the projection."""

        return self.readout_run.rates


def project_steady_state_readout(
    curve: SpinYieldCurve,
    readout_run: ReadoutRunContract,
    reference_point_id: str,
) -> SteadyStateReadoutProjection:
    """Project imposed spin yields through the reproduced steady-state readout model.

    This operation performs no spin dynamics, fitting, interpolation, extrapolation, or
    numerical uncertainty propagation.
    """

    _require_text(reference_point_id, "reference point ID")
    point_by_id = {point.point_id: point for point in curve.points}
    if reference_point_id not in point_by_id:
        raise ValueError("reference point ID is not present in the spin-yield curve")

    emissions = {
        point.point_id: FourStateModel(
            readout_run.rates, point.singlet_yield
        ).steady_state_emission_rate(total=1.0)
        for point in curve.points
    }
    reference_emission = emissions[reference_point_id]
    if reference_emission <= 0:
        raise ValueError("relative readout projection is undefined for zero reference emission")

    projected_points = tuple(
        _project_point(
            point=point,
            emission=emissions[point.point_id],
            reference_emission=reference_emission,
            readout_run=readout_run,
            reference_point_id=reference_point_id,
        )
        for point in curve.points
    )
    return SteadyStateReadoutProjection(
        input_curve=curve,
        readout_run=readout_run,
        reference_point_id=reference_point_id,
        points=projected_points,
    )


def _project_point(
    point: SpinYieldPoint,
    emission: float,
    reference_emission: float,
    readout_run: ReadoutRunContract,
    reference_point_id: str,
) -> ReadoutProjectionPoint:
    emission_unit = "population s^-1"
    relative_unit = "1"
    uncertainty_reason = (
        "Not calculated: singlet-yield uncertainty lacks a declared joint distribution and "
        "cross-field covariance model. The input uncertainty is retained unchanged."
    )
    emission_metadata = CalculatedObservableMetadata(
        unit=emission_unit,
        conditions=readout_run.conditions,
        readout_rates=readout_run.rates,
        reference_point_id=reference_point_id,
        uncertainty=Uncertainty(
            status=UncertaintyStatus.UNKNOWN,
            unit=emission_unit,
            note=uncertainty_reason,
        ),
    )
    relative_metadata = CalculatedObservableMetadata(
        unit=relative_unit,
        conditions=readout_run.conditions,
        readout_rates=readout_run.rates,
        reference_point_id=reference_point_id,
        uncertainty=Uncertainty(
            status=UncertaintyStatus.UNKNOWN,
            unit=relative_unit,
            note=uncertainty_reason,
        ),
    )
    return ReadoutProjectionPoint(
        input_point=point,
        emission_rate=CalculatedObservable(emission, emission_metadata),
        relative_change=CalculatedObservable(
            (emission - reference_emission) / reference_emission,
            relative_metadata,
        ),
    )
