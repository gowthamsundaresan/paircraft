import json
from dataclasses import replace

import pytest

from paircraft.oracle import (
    CalibrationTier,
    Maeda2012SimulatorOracle,
    ObservableKind,
    OraclePrediction,
    SimulationDatum,
    SolverSetting,
    maeda_2012_query,
    simulation_datum_document,
    simulation_datum_json,
)
from paircraft.spin import MaedaProteinSystem


@pytest.fixture(scope="module")
def predictions() -> tuple[OraclePrediction, OraclePrediction]:
    oracle = Maeda2012SimulatorOracle()
    results = tuple(
        oracle.predict(
            maeda_2012_query(
                MaedaProteinSystem.ATCRY1,
                (0.0, 0.0, field_tesla),
                query_id=f"field-{field_tesla}",
            )
        )
        for field_tesla in (1e-3, 10e-3)
    )
    assert all(isinstance(result, OraclePrediction) for result in results)
    return results  # type: ignore[return-value]


def test_simulation_datum_emits_canonical_complete_nontraining_json(
    predictions: tuple[OraclePrediction, OraclePrediction],
) -> None:
    prediction = predictions[0]
    assert prediction.execution is not None
    datum = SimulationDatum(prediction, prediction.execution)

    first = simulation_datum_json(datum)
    second = simulation_datum_json(datum)
    document = json.loads(first)

    assert first == second
    assert document == simulation_datum_document(datum)
    assert document["evidence_origin"] == "simulation"
    assert document["membership"] == "reference_only_not_training"
    assert document["prediction"]["distribution"]["unit"] == "1"
    assert document["prediction"]["lineage"]
    assert document["execution"]["model_family"] == prediction.query.model_family
    assert document["hashes"]["datum_sha256"] == datum.datum_sha256
    assert document["hashes"]["configuration_sha256"] == datum.execution.configuration_sha256
    assert len(datum.query_sha256) == 64
    assert len(datum.prediction_sha256) == 64


def test_fields_share_dependency_group_but_not_datum_identity(
    predictions: tuple[OraclePrediction, OraclePrediction],
) -> None:
    first_prediction, second_prediction = predictions
    assert first_prediction.execution is not None
    assert second_prediction.execution is not None
    first = SimulationDatum(first_prediction, first_prediction.execution)
    second = SimulationDatum(second_prediction, second_prediction.execution)

    assert first.dependency_group_sha256 == second.dependency_group_sha256
    assert first.query_sha256 != second.query_sha256
    assert first.datum_sha256 != second.datum_sha256


def test_execution_mismatch_and_nonsimulator_oracle_are_rejected(
    predictions: tuple[OraclePrediction, OraclePrediction],
) -> None:
    prediction = predictions[0]
    other = predictions[1]
    assert prediction.execution is not None
    assert other.execution is not None

    with pytest.raises(ValueError, match="exact execution"):
        SimulationDatum(prediction, other.execution)

    with pytest.raises(ValueError, match="schema version"):
        SimulationDatum(prediction, prediction.execution, schema_version="unknown-schema")

    with pytest.raises(ValueError, match="evidence origin"):
        SimulationDatum(
            prediction,
            prediction.execution,
            evidence_origin="measurement",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="reference-only"):
        SimulationDatum(
            prediction,
            prediction.execution,
            membership="training",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="OraclePrediction"):
        SimulationDatum("not-a-prediction", prediction.execution)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="NumericalExecution"):
        SimulationDatum(prediction, None)  # type: ignore[arg-type]

    retrospective_identity = replace(
        prediction.oracle,
        calibration_tier=CalibrationTier.PUBLIC_RETROSPECTIVE,
    )
    retrospective = replace(prediction, oracle=retrospective_identity)
    with pytest.raises(ValueError, match="simulator-only"):
        SimulationDatum(retrospective, prediction.execution)

    with pytest.raises(ValueError, match="executed model ID"):
        replace(prediction, executed_model_id="unrelated-model")


def test_meaningful_solver_setting_changes_execution_and_datum_hashes(
    predictions: tuple[OraclePrediction, OraclePrediction],
) -> None:
    prediction = predictions[0]
    assert prediction.execution is not None
    baseline = SimulationDatum(prediction, prediction.execution)
    changed_settings = tuple(
        SolverSetting(setting.name, "99", setting.unit, setting.definition)
        if setting.name == "restart"
        else setting
        for setting in prediction.execution.settings
    )
    changed_execution = replace(prediction.execution, settings=changed_settings)
    changed_prediction = replace(prediction, execution=changed_execution)
    changed = SimulationDatum(changed_prediction, changed_execution)

    assert baseline.execution.configuration_sha256 != changed.execution.configuration_sha256
    assert baseline.execution.execution_sha256 != changed.execution.execution_sha256
    assert baseline.datum_sha256 != changed.datum_sha256


def test_relative_prediction_requires_composite_execution_provenance() -> None:
    result = Maeda2012SimulatorOracle().predict(
        maeda_2012_query(
            MaedaProteinSystem.ATCRY1,
            (0.0, 0.0, 1e-3),
            observable=ObservableKind.SECONDARY_RADICAL_YIELD_MFE,
            reference_field_tesla=(0.0, 0.0, 0.0),
        )
    )
    assert isinstance(result, OraclePrediction)
    assert result.execution is not None

    with pytest.raises(ValueError, match="composite target/reference"):
        SimulationDatum(result, result.execution)

    with pytest.raises(ValueError, match="cannot be less than -1"):
        replace(result, distribution=replace(result.distribution, lower=-1.1))


def test_numerical_execution_rejects_ambiguous_or_false_run_identity(
    predictions: tuple[OraclePrediction, OraclePrediction],
) -> None:
    prediction = predictions[0]
    execution = prediction.execution
    assert execution is not None

    with pytest.raises(ValueError, match="at least one solver dependency"):
        replace(execution, solver_versions=())
    with pytest.raises(ValueError, match="unique sorted keys"):
        replace(execution, solver_versions=(("scipy", "1"), ("scipy", "2")))
    with pytest.raises(ValueError, match="solver dependency name"):
        replace(execution, solver_versions=(("", "1"),))
    with pytest.raises(ValueError, match="unique sorted names"):
        replace(execution, settings=(execution.settings[0], execution.settings[0]))
    with pytest.raises(ValueError, match="convergence diagnostics"):
        replace(execution, achieved_relative_residual=-1.0)
    with pytest.raises(ValueError, match="convergence diagnostics"):
        replace(execution, iteration_count=0)
    with pytest.raises(ValueError, match="cannot carry a random seed"):
        replace(execution, random_seed=7)
    with pytest.raises(ValueError, match="requires an explicit random seed"):
        replace(execution, deterministic=False)
    with pytest.raises(ValueError, match="without execution cannot claim"):
        replace(prediction, execution=None, executed_model_id=execution.model_id)
