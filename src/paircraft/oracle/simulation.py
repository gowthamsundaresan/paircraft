"""Canonical non-training records for individual simulator-oracle predictions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Any, cast

from paircraft.oracle.model import (
    CalibrationTier,
    FieldIntent,
    NumericalExecution,
    OraclePrediction,
)

_SCHEMA_VERSION = "paircraft-simulation-datum-v1"


class EvidenceOrigin(StrEnum):
    """Origin of evidence carried by a datum."""

    SIMULATION = "simulation"


class CorpusMembership(StrEnum):
    """Current corpus role; no training role exists before a manifest assigns one."""

    REFERENCE_ONLY_NOT_TRAINING = "reference_only_not_training"


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SimulationDatum:
    """One complete simulator prediction, explicitly not assigned as training data."""

    prediction: OraclePrediction
    execution: NumericalExecution
    schema_version: str = _SCHEMA_VERSION
    evidence_origin: EvidenceOrigin = EvidenceOrigin.SIMULATION
    membership: CorpusMembership = CorpusMembership.REFERENCE_ONLY_NOT_TRAINING

    def __post_init__(self) -> None:
        if not isinstance(self.prediction, OraclePrediction):
            raise ValueError("simulation datum requires an OraclePrediction")
        if not isinstance(self.execution, NumericalExecution):
            raise ValueError("simulation datum requires a NumericalExecution")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("simulation datum schema version is unsupported")
        if self.evidence_origin is not EvidenceOrigin.SIMULATION:
            raise ValueError("simulation datum evidence origin must be simulation")
        if self.membership is not CorpusMembership.REFERENCE_ONLY_NOT_TRAINING:
            raise ValueError("simulation datum membership must remain reference-only")
        if self.prediction.query.observable.is_relative:
            raise ValueError(
                "relative predictions require composite target/reference execution provenance"
            )
        if self.prediction.execution != self.execution:
            raise ValueError("simulation datum execution must be the prediction's exact execution")
        if self.execution.model_family is not self.prediction.query.model_family:
            raise ValueError("simulation execution model family contradicts the prediction")
        if self.prediction.executed_model_family is not self.execution.model_family:
            raise ValueError("prediction and execution model families must match")
        if self.prediction.executed_model_id != self.execution.model_id:
            raise ValueError("prediction and execution model IDs must match")
        protocols = (
            self.prediction.query.target_protocol,
            self.prediction.query.reference_protocol,
        )
        if any(
            protocol is not None
            and protocol.field.intent is not FieldIntent.IDEAL_SIMULATION_TARGET
            for protocol in protocols
        ):
            raise ValueError("simulation data require ideal simulation target fields")
        if self.prediction.oracle.calibration_tier is not CalibrationTier.SIMULATOR_ONLY:
            raise ValueError("simulation data require a simulator-only oracle")

    @property
    def query_sha256(self) -> str:
        """Hash the complete resolved query."""

        return _canonical_sha256(asdict(self.prediction.query))

    @property
    def prediction_sha256(self) -> str:
        """Hash the complete oracle prediction without reinterpreting its scalar."""

        return _canonical_sha256(asdict(self.prediction))

    @property
    def dependency_group_sha256(self) -> str:
        """Group fields sharing one model, catalog, parameter set, and oracle artifact."""

        system = self.prediction.query.system
        payload = {
            "system_id": system.system_id,
            "parameter_set_id": system.parameter_set_id,
            "parameter_set_sha256": system.parameter_set_sha256,
            "parameter_catalog_version": system.parameter_catalog_version,
            "parameter_catalog_sha256": system.parameter_catalog_sha256,
            "model_family": self.prediction.query.model_family,
            "oracle_artifact_sha256": self.prediction.oracle.artifact_sha256,
        }
        return _canonical_sha256(payload)

    @property
    def datum_sha256(self) -> str:
        """Hash datum identity, outcome, execution, origin, and membership."""

        payload = {
            "schema_version": self.schema_version,
            "query_sha256": self.query_sha256,
            "prediction_sha256": self.prediction_sha256,
            "execution_sha256": self.execution.execution_sha256,
            "dependency_group_sha256": self.dependency_group_sha256,
            "evidence_origin": self.evidence_origin,
            "membership": self.membership,
        }
        return _canonical_sha256(payload)


def simulation_datum_document(datum: SimulationDatum) -> dict[str, object]:
    """Return a complete JSON-compatible document with derived content hashes."""

    document: dict[str, Any] = {
        "schema_version": datum.schema_version,
        "evidence_origin": datum.evidence_origin,
        "membership": datum.membership,
        "prediction": asdict(datum.prediction),
        "execution": asdict(datum.execution),
        "hashes": {
            "query_sha256": datum.query_sha256,
            "prediction_sha256": datum.prediction_sha256,
            "configuration_sha256": datum.execution.configuration_sha256,
            "execution_sha256": datum.execution.execution_sha256,
            "dependency_group_sha256": datum.dependency_group_sha256,
            "datum_sha256": datum.datum_sha256,
        },
    }
    normalized = json.loads(
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    if not isinstance(normalized, dict):
        raise AssertionError("simulation datum normalization did not produce an object")
    return cast(dict[str, object], normalized)


def simulation_datum_json(datum: SimulationDatum) -> bytes:
    """Return canonical compact JSON bytes; no persistence policy is implied."""

    return json.dumps(
        simulation_datum_document(datum),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
